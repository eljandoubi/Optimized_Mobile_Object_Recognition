"""
Graph optimization utilities for PyTorch models.
Supports TorchScript and TorchFX optimizations with unified evaluation pipeline.
"""
import os
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import (
    fuse,
    remove_dropout, 
    optimize_for_inference
)
from typing import Dict, Any, Optional, Tuple, Literal, List, Union, Callable
import warnings
import json
import time

def optimize_model(
    model: nn.Module,
    optimization_method: Literal["torchscript", "torch_fx"] = "torchscript",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cuda'),
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize a model using PyTorch graph optimization techniques.
    
    Args:
        model: Model to optimize
        optimization_method: Optimization method to use
        input_shape: Input shape for tracing (e.g., (1, 3, 224, 224) for a single RGB image)
        device: Device to set the input and model on
        custom_options: Custom optimization options (optional)
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a sample input tensor for tracing
    dummy_input = torch.randn(input_shape).to(device)
    model = model.to(device)
    
    # Apply optimization based on method
    if optimization_method == "torchscript":
        optimized_model = _optimize_with_torchscript(model, dummy_input, custom_options)    
    elif optimization_method == "torch_fx":
        optimized_model = _optimize_with_torch_fx(model, dummy_input, custom_options)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")
        
    return optimized_model


def _optimize_with_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> torch.jit.ScriptModule:
    """
    Optimize model with TorchScript (JIT).
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        
    Returns:
        TorchScript optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}

    # `use_script` lets the caller opt into torch.jit.script instead of
    # torch.jit.trace. Tracing is simpler and works for most CNNs (no
    # data-dependent control flow), but scripting is more faithful when
    # the model has conditionals/loops that depend on tensor values.
    use_script = custom_options.get("use_script", False)
    do_freeze = custom_options.get("freeze", True)
    do_optimize_for_inference = custom_options.get("optimize_for_inference", True)

    model.eval()

    with torch.no_grad():
        if use_script:
            scripted_model = torch.jit.script(model)
        else:
            with warnings.catch_warnings():
                # Tracing commonly emits TracerWarnings for things like
                # Python control flow that won't be captured dynamically;
                # for a standard eval-mode CNN forward pass these are
                # expected/benign, so we suppress the noise here.
                warnings.simplefilter("ignore")
                scripted_model = torch.jit.trace(model, dummy_input, check_trace=False)

    scripted_model.eval()

    # Freezing inlines the module's parameters/buffers as constants in the
    # graph, which unlocks additional graph-level optimizations (constant
    # folding, Conv+BN fusion, etc.) that aren't safe on a "live" module
    # whose parameters could still change.
    if do_freeze:
        scripted_model = torch.jit.freeze(scripted_model)

    # Applies TorchScript's inference-time optimization passes (operator
    # fusion, removing no-ops, etc.) on top of the frozen graph.
    if do_optimize_for_inference:
        scripted_model = torch.jit.optimize_for_inference(scripted_model)

    return scripted_model


def _optimize_with_torch_fx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize model with Torch FX.
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        device: str
        
    Returns:
        FX-optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}

    pass_config = custom_options.get("pass_config", None)

    model.eval()

    with torch.no_grad():
        # Dropout is a no-op in eval mode anyway, but stripping the nodes
        # out of the FX graph avoids unnecessary graph complexity/overhead.
        optimized_model = remove_dropout(model)

        # Fuse fusible Conv+BN patterns (and similar) directly on the
        # nn.Module before the inference-optimization pass below -- this
        # is done explicitly here for clarity, even though
        # `optimize_for_inference` performs its own fusion pass internally
        # as well (a second pass over already-fused modules is a no-op).
        optimized_model = fuse(optimized_model)

        # Run FX's broader inference-optimization pipeline: additional
        # operator fusions, and (where applicable/supported on this
        # hardware) MKLDNN conversion for faster CPU inference.
        optimized_model = optimize_for_inference(optimized_model, pass_config=pass_config)

    optimized_model.eval()

    return optimized_model


def verify_model_equivalence(
    original_model: nn.Module,
    optimized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cpu'),
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> bool:
    """
    Verify equivalence between original and optimized models.
    
    Args:
        original_model: Original PyTorch model
        optimized_model: Optimized model
        input_shape: Shape of input tensor
        device: Device to run verification on
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if models are equivalent, False otherwise
    """
    # Set models to evaluation mode
    original_model.eval()
    optimized_model.eval()
    
    # Create a random input tensor
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.randn(input_shape, device=device)
    
    # Run inference with both models
    with torch.no_grad():
        original_output = original_model(input_tensor)
        optimized_output = optimized_model(input_tensor)
    
    # Compare outputs
    if isinstance(original_output, tuple):
        original_output = original_output[0]  # Use first output if model returns multiple
    
    if isinstance(optimized_output, tuple):
        optimized_output = optimized_output[0]
    
    # Check if the outputs are close
    is_close = torch.allclose(original_output, optimized_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("Original and optimized models produce equivalent outputs.")
    else:
        max_diff = torch.max(torch.abs(original_output - optimized_output))
        mean_diff = torch.mean(torch.abs(original_output - optimized_output))
        print(f"Models differ. Max difference: {max_diff.item():.6f}, Mean difference: {mean_diff.item():.6f}")
    
    return is_close