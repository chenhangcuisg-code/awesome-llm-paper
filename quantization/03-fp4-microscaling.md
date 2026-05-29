# Trend 3 — FP4 / MXFP4 / NVFP4 / Microscaling

**The fastest-growing 2025 trend.** Blackwell-era hardware (NVIDIA B100/B200) and AMD MI355X expose native FP4 / microscaling formats. Both *inference* (PTQ for MXFP4/NVFP4) and *pretraining* (Quartet I/II, NVIDIA NVFP4 recipe) papers exploded.

## FP4 pretraining

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **FP4 All the Way: Fully Quantized Training of LLMs** | arXiv | 2025-05 | [arXiv:2505.19115](https://arxiv.org/abs/2505.19115) | Trains 7B LLM end-to-end in FP4 on 1T tokens with 256 Intel Gaudi2 accelerators. Brief BF16→FP4 QAT phase, matches BF16 quality. First non-NVIDIA fully-FP4 GEMM at pretrain scale. |
| 2 | **Optimizing LLM Training Using FP4 Quantization** (Microsoft) | arXiv | 2025-01 | [arXiv:2501.17116](https://arxiv.org/abs/2501.17116) | First systematic FP4 LLM training framework: differentiable quantization estimator for weights + outlier clamping & compensation for activations. Up to 13B / 100B tokens matching BF16/FP8. |
| 3 | **Towards Efficient Pre-training: Exploring FP4 Precision** | arXiv | 2025-02 | [arXiv:2502.11458](https://arxiv.org/abs/2502.11458) | Mixed-precision FP4 pretraining strategies per module/training stage. Identifies failure modes of naive FP4 training. |
| 4 | **Quartet: Native FP4 Training Can Be Optimal for LLMs** | arXiv | 2025-05 | [arXiv:2505.14669](https://arxiv.org/abs/2505.14669) | Shows MXFP4 with all three GEMMs in FP4 is near-lossless and Pareto-optimal across FLOPs vs accuracy under fixed compute budget. |
| 5 | **Quartet II: Accurate LLM Pre-Training in NVFP4 via Unbiased Gradient Estimation** | arXiv | 2025 | [arXiv:2601.22813](https://arxiv.org/abs/2601.22813) | Successor with provably unbiased gradient estimator. Shrinks residual gap to BF16 on >12B models. |
| 6 | **Pretraining LLMs with NVFP4** (NVIDIA) | arXiv | 2025-09 | [arXiv:2509.25149](https://arxiv.org/abs/2509.25149) | Recipe for pretraining in NVFP4 (4-bit float, FP8 scale, smaller micro-block than MXFP4). Lossless ≥12B training on Blackwell. |
| 7 | **Recipes for Pre-training LLMs with MXFP8** (NVIDIA) | arXiv | 2025-06 | [arXiv:2506.08027](https://arxiv.org/abs/2506.08027) | Validated MXFP8 recipe: E4M3 throughout, weight-only stochastic rounding, BF16-equivalent loss curves. |
| 8 | **Pretraining LLMs with MXFP4 on Native FP4 Hardware (AMD MI355X)** | arXiv | 2025 | [arXiv:2605.09825](https://arxiv.org/abs/2605.09825) | First non-emulated MXFP4 pretraining on AMD Instinct MI355X. Identifies gradient paths where micro-scaling error dominates. |

## FP4 / Microscaling inference (PTQ)

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 9 | **Bridging the Gap Between Promise and Performance for Microscaling FP4** | arXiv | 2025-09 | [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) | First comprehensive MXFP4 + NVFP4 PTQ study. Introduces **MR-GPTQ (Micro-Rotated GPTQ)** with block-wise Hadamard. Reveals kernel inefficiencies preventing theoretical 2× speedup. |
| 10 | **Block Rotation Is All You Need for MXFP4 Quantization** | arXiv | 2025-11 | [arXiv:2511.04214](https://arxiv.org/abs/2511.04214) | Block-level rotations alone close most of the MXFP4 gap. |
| 11 | **RaZeR: Pushing the Limits of NVFP4 with Redundant Zero Remapping** | arXiv | 2025-01 | [arXiv:2501.04052](https://arxiv.org/abs/2501.04052) | Reclaims unused zero codes in NVFP4 codebook to extend dynamic range. |
| 12 | **ARCQuant: Boosting NVFP4 with Augmented Residual Channels** | arXiv | 2025 | [arXiv:2601.07475](https://arxiv.org/abs/2601.07475) | Residual augmented channels for NVFP4 outlier handling. |
| 13 | **MicroMix: Mixed-Precision Quantization with Microscaling Formats** | arXiv | 2025-08 | [arXiv:2508.02343](https://arxiv.org/abs/2508.02343) | Mixed MXFP4/6/8 per channel with Blackwell-tailored kernel. |
| 14 | **AMXFP4: Asymmetric Microscaling FP for 4-bit LLM Inference** | arXiv | 2024-11 anchor | [arXiv:2411.09909](https://arxiv.org/abs/2411.09909) | Asymmetric MXFP4 variant fitting skewed activation distributions better. |
| 15 | **Dissecting Outlier Dynamics in LLM NVFP4 Pretraining** | arXiv | 2025 | [arXiv:2602.02047](https://arxiv.org/abs/2602.02047) | Tracks per-channel kurtosis evolution during NVFP4 pretraining; proposes targeted clipping. |

### Related scaling-law paper

- **Scaling Laws for Floating Point Quantization Training** — [arXiv:2501.02423](https://arxiv.org/abs/2501.02423), Jan 2025 — Optimal exponent-mantissa split per bit width; cost-optimal precision lies in 4–8 bits.
