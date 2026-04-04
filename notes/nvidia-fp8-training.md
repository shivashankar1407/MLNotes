# Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training

**Source:** NVIDIA Developer Blog | **Date:** 2026

## Summary

This blog post explores FP8 (Floating-Point 8) training for efficient LLM training. FP8 comes in two variants: E4M3 (4 exponent, 3 mantissa bits) for forward passes where precision matters, and E5M2 (5 exponent, 2 mantissa bits) for backward passes where wider dynamic range is needed. Unlike INT8's fixed-point nature, FP8's floating-point format handles the wide dynamic range of activations and gradients in transformers through per-number implicit scaling via the exponent.

## Technical Details

### FP8 Format Variants
- **E4M3**: 4 exponent + 3 mantissa bits, range ±448, handles most layer outputs without overflow
- **E5M2**: 5 exponent + 2 mantissa bits, range ±57,344, better for gradients in backward passes

### Why FP8 > INT8 for LLMs
- INT8 requires predefined scaling factors that struggle with unpredictable dynamic ranges
- Attention mechanism exponentiated scores (near-zero to thousands) cause errors with INT8's fixed scaling
- Deep neural network gradient propagation produces extreme values that FP8's exponents handle well

### NVIDIA Blackwell Microscaling (MXFP8)
- Standard FP8: single FP32 scaling factor per tensor
- MXFP8: block-level scaling (32 values per block), each with distinct scaling factor
- Better accommodates variations within tensors, more accurate representation
- Enables wider adoption of higher-precision E4M3 format

### Scaling Strategies (FP8 Recipes)

**Tensor Scaling:**
- **Delayed Scaling**: Uses history of max absolute values (amax) over previous steps, not current values
- **Per-tensor Current Scaling**: Determines scale factor based on current forward/backward pass statistics

**Block Scaling:**
- **MXFP8** (Blackwell): Fixed block size of 32 values, each with power-of-2 scaling factor (E8M0)
- **General FP8**: Configurable block dimensions (e.g., 1×128, 128×128)

### Memory Implications
- Standard FP8: one FP32 scaling factor per tensor
- MXFP8: one E8M0 scaling factor per 32-element block
- Scaling factors managed automatically by Transformer Engine

## Key Insights

1. FP8 without proper scaling can cause training divergence due to outlier tensor values
2. NVIDIA Transformer Engine handles FP8 in Ada Lovelace/Hopper and MXFP8 in Blackwell
3. MXFP8 (8B LLM) achieves comparable perplexity to BF16 during pretraining
4. Real-world success: iGenius achieved 82.04% MMLU accuracy with FP8 on 355B model

## Learn More
- [FP8 Recipes in Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [iGenius Case Study](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)
- GTC 2025: Building LLMs with FP8 Pretraining

---
Source: https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/