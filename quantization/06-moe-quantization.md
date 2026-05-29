# Trend 6 — MoE Quantization

The rise of DeepSeek-V3 (671B) / Mixtral-8x22B / Qwen-MoE drove a focused subfield: **per-expert calibration imbalance**, **expert-aware Hessians**, and **mixed-precision per expert**.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **MoEQuant: Enhancing Quantization for MoE LLMs** | arXiv | 2025-05 | [arXiv:2505.03804](https://arxiv.org/abs/2505.03804) | Expert-balance-aware calibration sampling + expert-weighted Hessian for MoE PTQ. +10 HumanEval points on DeepSeekMoE-16B at INT4. Targets Mixtral / DeepSeek-V3. |
| 2 | **EAQuant: Expert-Aware Optimization for MoE PTQ** | arXiv | 2025-06 | [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Expert-aware reconstruction; +1.15–1.37% W4A4 / +1.33–2.28% W3A4 over baselines on DeepSeek-MoE-16B and Mixtral-8x7B. |
| 3 | **MxMoE: Mixed-Precision MoE Quantization with Co-Design** | arXiv | 2025-05 | [arXiv:2505.05799](https://arxiv.org/abs/2505.05799) | Allocates per-expert precision. Targets DeepSeek-V3 671B inference on 8×H100. |
| 4 | **MoBiE: Mixture of Binary Experts under PTQ** | arXiv | 2025 | [arXiv:2604.06798](https://arxiv.org/abs/2604.06798) | Binary-expert PTQ outperforming SOTA binary methods on MoE LLMs (incl. DeepSeek-V3). |
| 5 | **Efficient MoE Quantization with Theoretical Generalization Guarantees** | arXiv | 2025 | [arXiv:2604.06515](https://arxiv.org/abs/2604.06515) | Theory-backed expert quantization on Switch Transformer + Mixtral. |
| 6 | **EAC-MoE: Expert-Selection Aware Compressor for MoE LLMs** | arXiv | 2025-08 | [arXiv:2508.01625](https://arxiv.org/abs/2508.01625) | Quantization sensitivity weighted by per-expert routing frequency; preserves accuracy on hot experts. |
| 7 | **PuzzleMoE: Compression via Sparse Expert Merging + Bit-packed Inference** | arXiv | 2025-11 | [arXiv:2511.04805](https://arxiv.org/abs/2511.04805) | Bit-packed inference kernel for compressed MoE. Outperforms prior expert pruning by 16.7% under 50% compression. |
| 8 | **Collaborative Compression for Large-Scale MoE on Edge** | arXiv | 2025-09 | [arXiv:2509.25689](https://arxiv.org/abs/2509.25689) | Joint pruning + quantization for edge MoE. |
| 9 | **VEQ: Modality-Adaptive Quantization for MoE VLMs** | arXiv | 2025 | [arXiv:2602.01037](https://arxiv.org/abs/2602.01037) | Modality-aware expert quantization for MoE vision-language models. |
