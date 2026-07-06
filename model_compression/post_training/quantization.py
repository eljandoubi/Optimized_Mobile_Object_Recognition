"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm


class QuantizableMobileNetV3_Household(nn.Module):
    """
    Wraps a MobileNetV3_Household model with QuantStub/DeQuantStub so that the
    model can go through PyTorch's eager-mode static quantization workflow
    (prepare -> calibrate -> convert).

    The wrapped model's activations are quantized right after the input
    (via `quant`) and de-quantized right before the output (via `dequant`);
    everything in between runs through the original model's layers, which
    get swapped for their quantized counterparts during `convert`.
    """

    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.model = original_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        """
        Fuse conv, bn, relu layers for better quantization results

        Args:
            model: Model to fuse
        """
        print("Fusing layers...")

        # Get list of modules to fuse
        modules_to_fuse = []

        # MobileNetV3 is built out of nn.Sequential blocks (e.g. the
        # ConvNormActivation blocks used in the stem and inverted-residual
        # blocks). Eager-mode quantization fusion only supports the
        # Conv+BN, Conv+BN+ReLU and Conv+ReLU patterns out of the box, so we
        # walk every Sequential submodule looking for those patterns.
        # Note: MobileNetV3 also uses Hardswish/Hardsigmoid activations,
        # which are *not* fusable via `fuse_modules`, so those are skipped.
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential):
                continue

            child_names = list(module._modules.keys())
            i = 0
            while i < len(child_names):
                current = module._modules[child_names[i]]

                if isinstance(current, nn.Conv2d):
                    group = [f"{name}.{child_names[i]}" if name else child_names[i]]
                    j = i + 1

                    if j < len(child_names) and isinstance(
                        module._modules[child_names[j]], nn.BatchNorm2d
                    ):
                        group.append(
                            f"{name}.{child_names[j]}" if name else child_names[j]
                        )
                        j += 1

                        if j < len(child_names) and isinstance(
                            module._modules[child_names[j]], (nn.ReLU, nn.ReLU6)
                        ):
                            group.append(
                                f"{name}.{child_names[j]}" if name else child_names[j]
                            )
                            j += 1

                    if len(group) >= 2:
                        modules_to_fuse.append(group)

                    i = j
                else:
                    i += 1

        if modules_to_fuse:
            print(f"Found {len(modules_to_fuse)} fusable pattern(s).")
            tq.fuse_modules(self.model, modules_to_fuse, inplace=True)
        else:
            print("No fusable Conv-BN(-ReLU) patterns found; skipping fusion.")


def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")
    
    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")


def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    print("Applying dynamic quantization...")

    # Dynamic quantization gives the biggest benefit on layers whose cost is
    # dominated by large weight matrices/matmuls -- Linear layers here.
    # (Conv2d dynamic quantization is supported too, but tends to help far
    # less for MobileNet-style architectures and is more prone to accuracy
    # loss, so we stick to the standard Linear-only recipe.)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model
                

def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    device = next(model.parameters()).device
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)

    # Make sure the requested backend is actually used for quantized kernels.
    torch.backends.quantized.engine = backend

    # Wrap the model with Quant/DeQuant stubs if it isn't already wrapped,
    # so activations get quantized/dequantized at the model boundaries.
    if not isinstance(model, QuantizableMobileNetV3_Household):
        model = QuantizableMobileNetV3_Household(model)

    model.eval()

    # Fuse Conv+BN(+ReLU) patterns for better accuracy and speed before
    # inserting observers.
    model.fuse_model()

    # Attach the qconfig (observers + fake-quant settings) for the chosen
    # backend, then insert observers throughout the model.
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    
    torch.quantization.prepare(model, inplace=True)

    # Calibrate: run representative data through the model so the inserted
    # observers can record activation statistics (min/max ranges).
    print(f"Calibrating with up to {calibration_num_batches} batch(es)...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_data_loader, total=calibration_num_batches)):
            if i >= calibration_num_batches:
                break
            # Support both (inputs, labels) batches and plain-tensor batches.
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(inputs.to(device))

    # Convert the observed/fake-quantized model into a truly quantized model
    # (replaces float modules with their quantized int8 counterparts).
    print("Convert model")
    quantized_model = torch.quantization.convert(model, inplace=False)

    return quantized_model