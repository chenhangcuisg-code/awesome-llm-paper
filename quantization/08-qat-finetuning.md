# Trend 8 — QAT, QLoRA Successors & Quantized Fine-Tuning

2025 trend: **QAT becomes cheap** (EfficientQAT runs 2-bit 70B in a day on one A100), and **PEFT × quantization converges** with learnable scales merged back into quantized weights.

## QAT — efficient & on-device

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **EfficientQAT: Efficient QAT for LLMs** | ACL 2025 | 2025 | [arXiv:2407.11062](https://arxiv.org/abs/2407.11062) | Decoupled block-wise QAT + end-to-end finetune; enables **2-bit Llama-2-70B in 41 hours on a single A100-80GB**. Code: [`OpenGVLab/EfficientQAT`](https://github.com/OpenGVLab/EfficientQAT). |
| 2 | **End-to-End On-Device QAT for LLMs at Inference Cost** | ICLR 2025 | 2025 | [arXiv:2509.00031](https://arxiv.org/abs/2509.00031) | On-device QAT at inference-memory footprint (no BF16 shadow copy). Targets smartphone-class devices. |
| 3 | **Unifying Block-wise PTQ and Distillation-based QAT for Progressive 2-bit Instruct LLMs** | arXiv | 2025-06 | [arXiv:2506.09104](https://arxiv.org/abs/2506.09104) | FP16 → INT4 → INT2 cascade combining block-wise PTQ + distillation-based QAT to recover instruction-following at 2-bit. |
| 4 | **Meta's Quantized Llama 3.2 (1B/3B): QAT + LoRA + SpinQuant** | Blog + model release | 2024-10 anchor | [Meta blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/) | Production QAT pipeline: BF16 SFT → QAT SFT → frozen-backbone LoRA SFT. SpinQuant offered as data-free alternative. |

## QLoRA / PEFT × quantization

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 5 | **CLoQ: Calibrated LoRA Initialization for Quantized LLMs** | arXiv | 2025-01 | [arXiv:2501.18475](https://arxiv.org/abs/2501.18475) | Closed-form LoRA initialization that aligns the quantized model with FP counterpart. |
| 6 | **QuAILoRA: Quantization-Aware Initialization for LoRA** | arXiv | 2024-10 / 2025 | [arXiv:2410.14713](https://arxiv.org/abs/2410.14713) | LoftQ-style init that minimizes quantization-induced **output drift of W+BA jointly**, not weight-error only. |
| 7 | **ApiQ: Fine-tuning of 2-Bit Quantized LLMs** | arXiv (v3) | 2025 | [arXiv:2402.05147](https://arxiv.org/abs/2402.05147) | 2-bit QLoRA-style finetuning beating LoftQ / GPTQ-LoRA by >10%. |
| 8 | **L4Q: Parameter-Efficient QAT Fine-Tuning on LLMs** | arXiv | 2025 | [arXiv:2402.04902](https://arxiv.org/abs/2402.04902) | Learnable quantization scales + LoRA adapters merged back into quantized weights post-training. Zero inference overhead. |
| 9 | **ARB-LoRA: Fidelity-Plasticity Mixed-Precision Fine-Tuning** | arXiv | 2025-05 | [arXiv:2505.03802](https://arxiv.org/abs/2505.03802) | Joint search over LoRA rank and per-layer bit-width during fine-tuning. Outperforms fixed-rank QLoRA at same memory budget. |
| 10 | **LoRDS: Continuous Low-Rank Decomposed Scaling** | arXiv | 2025 | [arXiv:2601.22716](https://arxiv.org/abs/2601.22716) | Unifies block scaling with low-rank decomposition; supports both PTQ and QAT. |
| 11 | **LoRAQuant: Mixed-Precision Quantization of LoRA to Ultra-Low Bits** | arXiv | 2025-10 | [arXiv:2510.26690](https://arxiv.org/abs/2510.26690) | Pushes LoRA adapters themselves down to ultra-low bits. |
| 12 | **Q-BLoRA: Better Fine-Tuning Balance via Increased Adaptation Rank** | arXiv | 2025 | — | Non-parametric balance between rank and quantization error. |

## Training-time outlier handling (cross-cutting)

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 13 | **Outlier-Safe Pre-Training for Robust 4-Bit Quantization** | ACL 2025 | 2025-06 | [arXiv:2506.19697](https://arxiv.org/abs/2506.19697) | **Muon optimizer + Single-Scale RMSNorm + learnable embedding projection** prevent privileged-basis outliers from emerging. 1.4B/1T-token retains 35.7 avg vs 26.5 (Adam) under W4A4. Code: [`dmis-lab/Outlier-Safe-Pre-Training`](https://github.com/dmis-lab/Outlier-Safe-Pre-Training). |
