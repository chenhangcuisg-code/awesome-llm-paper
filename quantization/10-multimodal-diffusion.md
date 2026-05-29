# Trend 10 — Multimodal / Diffusion / VLM Quantization

2025 saw the first systematic PTQ studies of **diffusion LLMs (LLaDA-style)**, **vision-language models (LLaVA / Qwen-VL / LLaVA-OneVision)**, and **image diffusion transformers (DiT / FLUX / Sana)**.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **Diffusion Model Quantization: A Review** | arXiv | 2025-05 | [arXiv:2505.05215](https://arxiv.org/abs/2505.05215) | Survey of U-Net and DiT quantization (Stable Diffusion 1–3, FLUX), W-only vs WA, timestep-aware schemes. |
| 2 | **Quantization Meets dLLMs: A Systematic PTQ Study for Diffusion LLMs** | arXiv | 2025-08 | [arXiv:2508.14896](https://arxiv.org/abs/2508.14896) | **First PTQ framework for diffusion language models** (LLaDA-style). Addresses dynamic masking ratios and mixed stable/stochastic tokens. |
| 3 | **VLMQ: PTQ for Large Vision-Language Models via Hessian Augmentation** | arXiv | 2025-08 | [arXiv:2508.03351](https://arxiv.org/abs/2508.03351) | Vision-aware Hessian correction for PTQ on LLaVA-OneVision and Qwen2/2.5-VL up to 32B. Outperforms standard GPTQ on multimodal benchmarks. |
| 4 | **LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal LLMs** | arXiv | 2025-09 | [arXiv:2509.23729](https://arxiv.org/abs/2509.23729) | Per-layer bit selection on LLaVA-1.5 and Qwen-2.5-VL. **40%/31% memory reduction vs 4-bit baseline** with <10% MME degradation. |
| 5 | **ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers** | arXiv | 2025-12 | [arXiv:2512.03673](https://arxiv.org/abs/2512.03673) | Convolution-friendly rotation matrices for W4A4 on DiT/FLUX-style models without retraining. |
| 6 | **Q-MLLM: Vector Quantization for Robust Multimodal LLM Security** | arXiv | 2025-11 | [arXiv:2511.16229](https://arxiv.org/abs/2511.16229) | Discretizes visual features via VQ as a defense against gradient attacks; doubles as memory-efficiency mechanism. |
| 7 | **Sana: Efficient High-Resolution Image Synthesis with Linear DiT** | NVIDIA / arXiv | 2024-10 / 2025 release | [arXiv:2410.10629](https://arxiv.org/abs/2410.10629) | Released **INT8** (per-token symmetric activations, per-channel weights) and **4-bit SVDQuant** variant with the Nunchaku inference engine. |
| 8 | **VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models** | arXiv | 2025 | [arXiv:2602.01037](https://arxiv.org/abs/2602.01037) | Modality-aware expert quantization for MoE VLMs. |
