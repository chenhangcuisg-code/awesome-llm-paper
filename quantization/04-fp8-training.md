# Trend 4 — FP8 Large-Scale Pretraining

DeepSeek-V3 demonstrated that **native FP8 training at 671B-parameter scale** is feasible and cost-effective, triggering an industry-wide FP8 wave through 2025. The trend is now toward *fully* FP8 GEMM (including backward) and FP8 optimizer states.

| # | Title | Authors | Venue | Year | Link | Summary |
|---|---|---|---|---|---|---|
| 1 | **DeepSeek-V3 Technical Report** ⚓ | DeepSeek-AI | arXiv | 2024-12 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | First production-scale ~671B MoE LLM trained natively in FP8. Per-group activation + per-tile weight quantization; 2.664M H800 GPU-hours on 14.8T tokens. Anchors the 2025 FP8 wave. |
| 2 | **Insights into DeepSeek-V3: Scaling Challenges & Hardware Reflections** | DeepSeek-AI | arXiv | 2025-05 | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) | Hardware-software co-design retrospective: FP8 accumulator precision, all-to-all overlap, recommendations for next-gen FP8/FP6 tensor cores. |
| 3 | **Towards Fully FP8 GEMM LLM Training at Scale** | — | arXiv | 2025-05 | [arXiv:2505.20524](https://arxiv.org/abs/2505.20524) | Pushes both forward and backward GEMMs entirely to FP8 (no BF16 master GEMM) via tensor-wise dynamic scaling for gradients. Validated at 70B scale. |
| 4 | **μnit Scaling: Simple and Scalable FP8 LLM Training** | Cerebras / Bloomberg | arXiv | 2025-02 | [arXiv:2502.05967](https://arxiv.org/abs/2502.05967) | Replaces per-tensor scaling with μP-style unit scaling so FP8 training "just works" across model sizes without delayed-scaling overhead. |
| 5 | **COAT: Compressing Optimizer & Activations for Memory-Efficient FP8 Training** | — | arXiv | 2024-10 / 2025 follow-ups | [arXiv:2410.19313](https://arxiv.org/abs/2410.19313) | Quantizes Adam optimizer states and activations to FP8 on top of GEMM path; ~1.5× total memory reduction over FP8-LM. |
| 6 | **MOSS: Efficient and Accurate FP8 LLM Training** | — | arXiv | 2025-11 | [arXiv:2511.05811](https://arxiv.org/abs/2511.05811) | Outlier-resilient block-wise FP8 scaling + stochastic rounding. Closes remaining gap with BF16 at trillion-token scale. |
| 7 | **LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs** | — | arXiv | 2025-12 | [arXiv:2512.15306](https://arxiv.org/abs/2512.15306) | Low-precision pretraining recipes for RTX 4090-class GPUs. 4-bit training more numerically stable than 16-bit Adam under tight memory. |

⚓ = late-2024 anchor that every 2025 FP8 paper builds on.
