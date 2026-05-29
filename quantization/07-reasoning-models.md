# Trend 7 — Reasoning-Model Quantization

**Emerging 2025 subfield.** After DeepSeek-R1 / o1-style long-CoT models, the community discovered that standard PTQ **degrades reasoning before perplexity moves**. Calibration with CoT traces and reasoning-preserving compression are the two main responses.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **When Reasoning Meets Compression: Effects of LLM Compression on Large Reasoning Models** | arXiv | 2025-04 | [arXiv:2504.02010](https://arxiv.org/abs/2504.02010) | Benchmarks compressed DeepSeek-R1 / distilled variants on AIME/GSM/MATH across AWQ, GPTQ, GPTAQ, SparseGPT, AlphaPruning, ANY4/3. Shows reasoning collapses faster than PPL. |
| 2 | **Reasoning Models Can Be Accurately Pruned via Chain-of-Thought Reconstruction** | arXiv | 2025-09 | [arXiv:2509.12464](https://arxiv.org/abs/2509.12464) | Uses CoT traces as calibration to preserve reasoning under compression. |
| 3 | **LAQuant: Overhead-Free Large-Reasoning-Model Quantization by Layer-wise Lookahead Loss** | arXiv | 2025 | [arXiv:2605.08755](https://arxiv.org/abs/2605.08755) | Layer-wise lookahead loss tailored for reasoning models. |
| 4 | **When Reasoning Meets Compression: Quantized Keys Steal Attention** | arXiv | 2025 | [arXiv:2605.26266](https://arxiv.org/abs/2605.26266) | Identifies attention-mass bias from quantized keys; introduces bias correction for reasoning models. |
| 5 | **Quantized but Deceptive? Multi-Dimensional Truthfulness Evaluation of Quantized LLMs** | arXiv | 2025-08 | [arXiv:2508.19432](https://arxiv.org/abs/2508.19432) | Reveals quantization-induced truthfulness regressions on reasoning prompts. |
| 6 | **What Makes Low-Bit QAT Work for Reasoning LLMs? A Systematic Study** | arXiv | 2025 | [arXiv:2601.14888](https://arxiv.org/abs/2601.14888) | Ablation across teacher, distillation loss, LR, and bit budget for QAT on math/code reasoning models. Finds **reasoning ability collapses earlier than perplexity under low-bit QAT**. |

### Related survey

- **Efficient Reasoning Models: A Survey** — TMLR 2025, [arXiv:2504.10903](https://arxiv.org/abs/2504.10903) — Surveys compression (incl. quantization) for reasoning LLMs.
