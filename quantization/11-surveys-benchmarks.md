# Trend 11 — Surveys & Benchmarks

The 2025 literature has matured to the point where systematic **benchmarks (cross-family / cross-bit-width / cross-hardware)** and **survey papers** define the methodology. The recurring finding: **FP8 W8A8 is essentially lossless; INT4 weight-only is competitive; sub-4-bit needs QAT or rotations.**

## Surveys

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **A Survey of Low-bit LLMs: Basics, Systems, and Algorithms** | arXiv | 2024-09 / Nov 2025 rev. | [arXiv:2409.16694](https://arxiv.org/abs/2409.16694) | Algorithm + system survey of low-bit LLMs. |
| 2 | **A Survey of Quantization in LLMs: Unlocking Hardware Efficiency** | JCST 41(1) | 2026-01 | [DOI:10.1007/s11390-026-5979-1](https://doi.org/10.1007/s11390-026-5979-1) | Covers PTQ, QAT, and inference-time quantization. |
| 3 | **Resource-Efficient Language Models: Quantization for Fast and Accessible Inference** | arXiv | 2025-05 | [arXiv:2505.08620](https://arxiv.org/abs/2505.08620) | Tutorial-style survey. |
| 4 | **Mixed-Precision Quantization for Language Models: Techniques and Prospects** | arXiv | 2025-10 | [arXiv:2510.16805](https://arxiv.org/abs/2510.16805) | Survey of mixed-precision PTQ. |
| 5 | **Efficient Reasoning Models: A Survey** | TMLR 2025 | 2025-04 | [arXiv:2504.10903](https://arxiv.org/abs/2504.10903) | Surveys compression incl. quantization for reasoning LLMs. |

## Cross-family benchmarks

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 6 | **Benchmarking Post-Training Quantization in LLMs** | arXiv | 2025-02 | [arXiv:2502.13178](https://arxiv.org/abs/2502.13178) | Cross-family PTQ benchmark across tasks and bit-widths. |
| 7 | **A Comprehensive Evaluation on Quantization Techniques for LLMs** | arXiv | 2025-07 | [arXiv:2507.17417](https://arxiv.org/abs/2507.17417) | Wide eval of AWQ/GPTQ/QuaRot/SpinQuant/SmoothQuant. |
| 8 | **Systematic Characterization of LLM Quantization: Performance, Energy, Quality** | arXiv | 2025-08 | [arXiv:2508.16712](https://arxiv.org/abs/2508.16712) | Performance/energy/quality characterization on commodity hardware. |
| 9 | **"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs** | arXiv | 2024-11 / 2025 | [arXiv:2411.02355](https://arxiv.org/abs/2411.02355) | **Finds FP8 W8A8 essentially lossless, INT4 weight-only competitive.** Widely cited benchmark anchor. |
| 10 | **Quantized but Deceptive? Multi-Dimensional Truthfulness Evaluation** | arXiv | 2025-08 | [arXiv:2508.19432](https://arxiv.org/abs/2508.19432) | Reveals quantization-induced truthfulness regressions on reasoning prompts. |

## Scaling laws

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 11 | **Scaling Laws for Floating Point Quantization Training** | arXiv | 2025-01 | [arXiv:2501.02423](https://arxiv.org/abs/2501.02423) | Optimal exponent-mantissa split per bit width; cost-optimal precision lies in 4–8 bits. |
| 12 | **Low-Bit Quantization Favors Undertrained LLMs (to 100T tokens)** | arXiv | 2024-11 (v2 2025) | [arXiv:2411.17691](https://arxiv.org/abs/2411.17691) | Studies 1500+ checkpoints. **Degradation grows with training tokens.** |
| 13 | **Scaling Law for Quantization-Aware Training** | arXiv | 2025-05 | [arXiv:2505.14302](https://arxiv.org/abs/2505.14302) | Empirical QAT scaling laws over model size, tokens, and bit-width. |
