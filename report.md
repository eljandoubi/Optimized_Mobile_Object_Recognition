# UdaciSense: Model Optimization Technical Report

## Executive Summary
UdaciSense's household-object recognition model needed to run well on mobile devices without giving up accuracy. Starting from a MobileNetV3-based baseline (5.96 MB, 227.94 ms per inference on CPU, 88.20% accuracy), we tested five families of compression techniques individually, then combined the most effective ones into a multi-stage pipeline. The final mobile-ready model is 0.59 MB — a 90% size reduction — runs in roughly 76-100 ms per inference on CPU (a 55-67% latency cut), and holds accuracy at 85.30%, a 2.9-point drop that stays well inside the 5% tolerance the CTO's team set. In practical terms: the app can now ship a model small enough to bundle directly into the mobile install, recognize objects noticeably faster on-device, and do it without a meaningful hit to the user experience's accuracy. All three of the CTO's technical requirements — 30% smaller, 40% faster, accuracy within 5% — were met with room to spare.

## 1. Baseline Model Analysis

### 1.1 Model Architecture
The baseline model wraps a torchvision MobileNetV3-Small backbone (ImageNet-pretrained) with a custom classifier head (`Linear → Hardswish → Dropout(0.2) → Linear`) sized for the 10 household-object classes. The backbone relies heavily on depthwise-separable convolutions, squeeze-and-excitation (SE) blocks, and Hardswish/Hardsigmoid activations rather than plain ReLU — all deliberate choices from the original MobileNetV3 architecture search aimed at mobile efficiency. The model has 1,528,106 total parameters and was trained for 34 epochs (early-stopped from a 100-epoch budget, best weights from epoch 19) on a 5,000-image training set / 1,000-image test set drawn from CIFAR-100's household categories.

### 1.2 Performance Metrics
| Metric | Value |
|--------|-------|
| Model Size (MB) | 5.96 |
| Inference Time - CPU (ms) | 227.94 |
| Inference Time - GPU (ms) | 6.15 |
| Accuracy (%) | 88.20 |
| Total Parameters | 1,528,106 |
| Peak Memory (GPU, MB) | 48.59 |

### 1.3 Optimization Challenges
Two things stood out before any optimization work started. First, CPU inference (227.94 ms) is roughly 37x slower than GPU (6.15 ms) for the same model, which tells us this model's real deployment target is CPU/edge, not GPU — so CPU latency is the number that actually matters even though the CTO's targets apply to both. Second, MobileNetV3-Small has already been through neural architecture search specifically for efficiency, meaning there's less obvious redundancy sitting around for pruning to remove than in an older, hand-designed architecture — and its SE blocks and Hardswish/Hardsigmoid activations fall outside the standard Conv+BN+ReLU pattern that PyTorch's `fuse_modules` fuses automatically, limiting how much benefit layer fusion alone provides. On the positive side, there was about 4.4 accuracy points of slack available (88.20% down to the 83.79% floor) to spend on compression.

## 2. Compression Techniques

### 2.1 Overview

#### Technique 1: Post-Training Dynamic Quantization
##### Implementation Approach
Applied `torch.quantization.quantize_dynamic` to the trained baseline, quantizing only the `Linear` layers to int8 (weights quantized ahead of time, activations quantized dynamically per batch). Tested both the fbgemm (x86) and qnnpack (ARM) backends.

##### Results
| Metric | Baseline | After Dynamic Quantization (fbgemm) | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96 | 4.24 | -28.8% |
| Inference Time - CPU (ms) | 227.94 | 197.03 | -13.6% |
| Accuracy (%) | 88.20 | 88.20 | +0.1% |

##### Analysis
Accuracy was essentially untouched, since only the classifier's Linear layers were quantized. Size reduction landed just short of the 30% target, and CPU speed barely moved, because the Conv2d layers — where most of the model's compute actually lives — stayed fp32. The qnnpack backend produced nearly identical results (4.24 MB, 203.01 ms, 88.20%), suggesting the backend choice doesn't matter much for this particular (Linear-only) quantization scope.

#### Technique 2: Quantization-Aware Training (QAT)
##### Implementation Approach
Fused Conv+BN(+ReLU) patterns, inserted fake-quantization observers via `prepare_qat`, and fine-tuned for 100 epochs (QAT activated after epoch 10, BatchNorm stats frozen 5 epochs later, observers disabled partway through), then converted to a fully int8 model. This quantizes both Conv2d and Linear layers, unlike dynamic quantization.

##### Results
| Metric | Baseline | After QAT (fbgemm) | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96 | 1.76 | -70.5% |
| Inference Time - CPU (ms) | 227.94 | 151.04 | -33.7% |
| Accuracy (%) | 88.20 | 76.70 | -13.0% |

##### Analysis
QAT delivered by far the largest size reduction of any single technique, since it quantizes the whole network rather than just the classifier — but accuracy dropped well outside the 5% tolerance, likely because the fine-tuning didn't fully recover from quantization noise interacting with the SE blocks/Hardswish activations. The qnnpack variant was considerably worse — 12.30% accuracy and 6567.95 ms (actually *slower* than baseline) — which points to a backend problem specific to running qnnpack on this x86 development machine rather than a fundamental issue with quantization itself; this needs to be re-validated on real ARM hardware before drawing conclusions about qnnpack.

#### Technique 3: Pruning (Post-Training and In-Training)
##### Implementation Approach
Tested both one-shot post-training L1 unstructured pruning (no retraining) at 10%/30%/50% sparsity, and gradual magnitude pruning during training (cubic schedule, global ranking) targeting 30%/50%/70% final sparsity with fine-tuning across the schedule.

##### Results
| Metric | Baseline | One-Shot Pruning (10%) | Gradual Pruning (30%, w/ fine-tuning) | Change (%) |
|--------|----------|------------------------|----------------------------------------|------------|
| Model Size (MB) | 5.96 | 5.96 | 5.96 | 0.0% |
| Inference Time - CPU (ms) | 227.94 | 183.03 | 193.99 | -19.7% / -14.9% |
| Accuracy (%) | 88.20 | 87.10 | 88.90 | -1.2% / +0.8% |

##### Analysis
The most important finding here wasn't about accuracy — it was that **model size never changed at all**, regardless of sparsity level or method. `torch.nn.utils.prune` only zeroes out weights; it doesn't remove them from the serialized tensor, so a "70% sparse" model saves to disk exactly the same size as the dense original, and dense CPU kernels don't skip the zeros either, so speed barely moves. Accuracy told a clear story on its own: one-shot pruning without retraining degrades fast (down to 7.90% accuracy at 50% sparsity), while gradual pruning with fine-tuning during training was essentially free — accuracy even improved slightly (+0.8 to +1.2% across the 30/50/70% sparsity runs tested), likely acting as a mild regularizer. Getting real size/speed benefit from pruning would require structured (channel-level) pruning that actually changes tensor shapes, or a sparsity-aware runtime — neither of which this implementation used.

#### Technique 4: Knowledge Distillation
##### Implementation Approach
Trained a narrower MobileNetV3-Small student (width_mult=0.6) against the trained baseline as teacher, using a temperature-scaled KL-divergence + cross-entropy loss. Swept temperature (1, 4, 5, 10) and alpha (0.1, 0.3, 0.5).

##### Results
| Metric | Baseline | Best Distillation Run (temp=10, alpha=0.3) | Change (%) |
|--------|----------|----------------------------------------------|------------|
| Model Size (MB) | 5.96 | 1.81 | -69.7% |
| Inference Time - CPU (ms) | 227.94 | 135.96 | -40.4% |
| Accuracy (%) | 88.20 | 71.50 | -18.9% |

##### Analysis
Distillation gave the strongest combination of size reduction and speedup of any technique except graph optimization — but training accuracy was unstable across almost every hyperparameter combination tried. Several runs collapsed to ~10% accuracy (chance level for 10 classes), and even the best-performing configuration only reached 71.50%, far outside the 5% tolerance. This looks like a training-stability problem (learning rate, epoch budget, or the width reduction being too aggressive) rather than a fundamental limit of distillation as a technique.

#### Technique 5: Graph Optimization (TorchScript)
##### Implementation Approach
Applied `torch.jit.trace`, `torch.jit.freeze`, and `torch.jit.optimize_for_inference` to the unmodified fp32 baseline — pure graph-level optimization with no change to weight precision.

##### Results
| Metric | Baseline | After Graph Optimization (CPU) | Change (%) |
|--------|----------|-------------------------------|------------|
| Model Size (MB) | 5.96 | 2.30 | -61.4% |
| Inference Time - CPU (ms) | 227.94 | 113.03 | -50.4% |
| Accuracy (%) | 88.20 | 88.20 | +0.0% |

##### Analysis
This was the only single technique that met all three CTO requirements on its own — over 60% smaller, over 2x faster, with zero accuracy cost, since it never touches numerical precision, only the execution graph (constant folding, operator fusion, dead-code elimination). The CUDA-targeted variant, by contrast, barely reduced size (2.0%) and slightly increased parameter count, since GPU-focused graph optimization mostly restructures execution rather than compacting the model.

### 2.2 Comparative Analysis
| Technique | Size Reduction | CPU Speedup | Accuracy Change | Meets All 3 Targets Alone? |
|---|---|---|---|---|
| Dynamic Quantization | 28.8% | ~1.2x | +0.1% | No (size, speed short) |
| QAT (static quantization) | 70.5% | ~1.5x | -13.0% | No (accuracy fails) |
| Pruning (any variant) | 0.0% | ~1.0-1.3x | +1.2% to -80.3% | No (size, speed unmoved) |
| Distillation (best config) | 69.7% | ~1.7x | -18.9% | No (accuracy fails) |
| Graph Optimization | 61.4% | ~2.0x+ | 0.0% | **Yes** |

Quantization and distillation are the strongest levers for size, but come with real accuracy risk when pushed hard enough to hit the size target. Pruning, as implemented here, contributes nothing to size or speed by itself — its actual value turned out to be that it's accuracy-safe (even accuracy-positive) when paired with fine-tuning, making it a good "free" first stage rather than a standalone solution. Graph optimization stood apart as the only technique that's simultaneously effective and risk-free on accuracy, since it operates on the execution graph rather than the weights. That combination — pruning as an accuracy-neutral base layer, quantization for size, graph optimization for speed — is what motivated the multi-stage pipeline design.

## 3. Multi-Stage Compression Pipeline

### 3.1 Pipeline Design
Since no single technique cleared both the size and speed bars together, the pipeline chains three stages: global unstructured pruning (40% target) as an accuracy-neutral base, dynamic quantization for size reduction, and TorchScript graph optimization for latency — anchored on graph optimization since it was the only technique proven to hit all three targets alone. Two variants were built to test whether the *order* of quantization and graph optimization matters: Pipeline A applies them as two separate sequential steps (quantize, then graph-optimize); Pipeline B applies them as one combined step (script + freeze + optimize, then quantize the resulting graph directly via PyTorch's JIT-native dynamic quantization).

### 3.2 Implementation
Built a reusable `OptimizationPipeline` class that chains step functions (each taking a model in and returning a modified model out), checkpointing and evaluating the model after every step, and saving the final result as a TorchScript (`.pt`) file. Both pipelines started from the same pruned baseline (40% global unstructured sparsity) before diverging in how quantization and graph optimization were combined.

### 3.3 Results
| Metric | Baseline | Final Optimized Model (Pipeline B) | Change (%) | Requirement Met? |
|--------|----------|------------------------|------------|----------|
| Model Size (MB) | 5.96 | 2.31 | -61.27% | ✅ (30% target) |
| Inference Time CPU (ms) | 227.94 | 114.00 | -49.99% | ✅ (40% target) |
| Accuracy (%) | 88.20 | 85.20 | -3.40% | ✅ (within 5%) |
| Total Parameters | 1,528,106 | 605,104 | -60.4% | - |

*(Pipeline A, for comparison, reached 4.13 MB / 111.02 ms / 85.30% accuracy — also passing all three official targets, but with a larger final model than Pipeline B for essentially the same accuracy cost.)*

### 3.4 Analysis
Both pipelines cleared all three CTO requirements, but Pipeline B was strictly better — smaller (2.31 MB vs 4.13 MB) at the same speed and accuracy cost. Looking at what each stage actually contributed: the pruning step alone already brought accuracy down to ~85% with zero measurable change in size or speed (consistent with the earlier finding that unstructured pruning doesn't shrink the serialized model on its own) — meaning essentially all of the size and speed improvement in both pipelines came from the quantization and graph-optimization stages, not from pruning. The more interesting technical insight is that *how* those two stages combine matters: running graph optimization and quantization as one integrated step (Pipeline B) produced a meaningfully smaller final artifact than running them as two separate sequential steps (Pipeline A), even though both started from the same pruned model and reached similar speed and accuracy. This suggests the graph-level fusion pass interacts differently with quantized operators depending on whether it's applied before, after, or interleaved with quantization — worth exploring further. On trade-offs: both pipelines spent roughly 3 of the 4.4 available accuracy points to buy their size/speed wins, leaving some accuracy budget unused, and both used a single fixed pruning level (40%) that wasn't swept — an easy next experiment.

## 4. Mobile Deployment

### 4.1 Export Process
The pipeline's TorchScript output was converted for mobile via `torch.jit.trace` followed by `torch.utils.mobile_optimizer.optimize_for_mobile`, then saved with `torch.jit.save`. Note: the deployment notebook loaded Pipeline A's result (4.13 MB, 85.30% accuracy) rather than the objectively smaller Pipeline B result (2.31 MB, 85.20% accuracy) — worth revisiting, since Pipeline B would likely produce an even smaller final mobile artifact for essentially the same accuracy.

### 4.2 Mobile-Specific Considerations
The most important open issue is the quantization backend: this model was quantized with fbgemm, which targets x86 CPUs, while real ARM phones require the qnnpack backend. That's not a minor detail — the one qnnpack run attempted in this project (QAT, notebook 2) failed badly on this dev machine (12.30% accuracy, 6567.95 ms), so the qnnpack path is effectively unvalidated and needs to be re-run and confirmed working, ideally on actual ARM hardware, before this model ships to real devices. Beyond that, device fragmentation (a wide range of CPU capability across the Android install base) and thermal throttling under sustained inference are both realistic risks that a single-CPU dev-machine benchmark can't surface. The model was also saved via plain `torch.jit.save` rather than the lite-interpreter (`_save_for_lite_interpreter`, `.ptl`) format that PyTorch Mobile's production runtime actually loads — switching to that format is a straightforward next step.

### 4.3 Performance Verification
An output-consistency check (`torch.allclose`, rtol/atol = 1e-3) between the pre-mobile and mobile-converted models passed, confirming the conversion is functionally lossless. Mobile packaging alone shrank the model further, from 4.13 MB to 0.59 MB (an additional 85.64% reduction) with accuracy completely unchanged (85.30% → 85.30%, a 0.00-point change) and CPU inference dropping further, from ~100.93 ms (pre-mobile) to 75.98 ms (mobile-optimized) — roughly another 1.3x on top of the pipeline's own gains. Measured end-to-end against the original baseline, the final mobile model represents about a 90% size reduction and 55-67% CPU speed improvement for a 2.9-point accuracy cost, comfortably clearing all three CTO targets. Since no physical ARM device was available in this environment, real-hardware benchmarking wasn't performed; the proposed approach is to use PyTorch Mobile's `speed_benchmark_torch` tool or a minimal on-device Android/iOS test harness, profiled with Android Studio's or Xcode's built-in profilers, measuring cold-start latency, warm-inference p50/p95/p99 (not just the mean, since mobile CPUs throttle under sustained load), peak memory, and energy draw per inference — run across a flagship, mid-range, and low-end device tier with a fixed thread count and a proper warmup period (discarding the first several runs) to keep the comparison fair.

## 5. Conclusion and Recommendations

### 6.1 Summary of Achievements
Starting from a 5.96 MB / 227.94 ms / 88.20%-accuracy baseline, five compression technique families were evaluated across roughly 20 configurations, identifying graph optimization as the only individually sufficient technique and pruning-with-fine-tuning as an accuracy-safe (if size/speed-neutral) complement. Two multi-stage pipelines were built and validated, both clearing all three CTO requirements, with the best (Pipeline B) reaching 2.31 MB (61.27% smaller), 114.00 ms (49.99% faster), and 85.20% accuracy (-3.40%). The deployed model was further converted to a mobile-ready format, landing at 0.59 MB with unchanged accuracy and additional speedup — roughly 90% smaller and 55-67% faster than baseline overall, for a well-tolerated ~2.9-point accuracy cost.

### 6.2 Key Insights
PyTorch's built-in unstructured pruning API doesn't reduce serialized model size or measured inference speed by itself, no matter how high the sparsity — a real tooling limitation worth knowing before relying on pruning as a compression technique in its own right. Quantization's benefit depends heavily on matching the backend to the target hardware, and "it ran without an error" isn't the same as "it's validated" — the qnnpack results here were a clear warning sign that needs following up on real hardware. The order in which quantization and graph optimization are combined changes the final model's size, not just its speed, which wasn't obvious going in. And mobile-specific packaging (`optimize_for_mobile`) delivered a substantial size win essentially for free on top of an already-optimized model.

### 6.3 Recommendations for Future Work
Before shipping to production: validate the qnnpack backend properly on real ARM hardware, since the only data point gathered so far for it was a failure. Deploy Pipeline B's model (2.31 MB) instead of Pipeline A's (4.13 MB), since it's strictly smaller for the same accuracy cost. Sweep the pruning amount (fixed at 40% in both pipelines tested) to see if a different level improves the size/speed/accuracy balance further. Revisit knowledge distillation layered on top of an already-pruned, regularized model — standalone distillation was unstable here, but the underlying approach (a genuinely smaller architecture, not just sparser/quantized weights) is attractive if training stability can be fixed. Finally, switch the mobile export to the lite-interpreter (`.ptl`) format, since that's what the production PyTorch Mobile runtime actually loads.

### 6.4 Business Impact
A model that's roughly 90% smaller means it can be bundled directly into the UdaciSense mobile app without a meaningful download or update-size penalty for users. Faster on-device inference (55-67% quicker) translates directly into a snappier, more responsive object-recognition experience — the difference between a scan feeling instant versus feeling laggy. Because inference runs on-device rather than requiring a server round-trip, this also avoids ongoing cloud inference costs and network-latency dependence for a core product feature. All of this comes at a controlled, well-understood accuracy cost (2.9 points, within the agreed 5% tolerance), so the trade-off was made deliberately and is documented, not accidental.

## Optional: References
- PyTorch quantization documentation (eager-mode static/dynamic quantization and QAT APIs)
- PyTorch TorchScript / `torch.jit` documentation (tracing, scripting, freezing, `optimize_for_inference`)
- `torch.utils.mobile_optimizer` and PyTorch Mobile lite-interpreter documentation
- Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (knowledge distillation loss formulation)
- Zhu & Gupta — "To prune, or not to prune" (cubic sparsity schedule used for gradual magnitude pruning)
- Howard et al. — "Searching for MobileNetV3" (baseline architecture)