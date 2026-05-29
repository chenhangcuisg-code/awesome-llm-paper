# Trend 1 — Rotation-based PTQ & Algorithmic Refinements

PTQ method papers from 2025+, covering both weight-only (GPTQ/AWQ successors) and weight+activation (SpinQuant/QuaRot successors). The dominant theme is **cheaper / smarter rotations** + better Hessian / rounding objectives.

## Weight-only PTQ

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **LeanQuant: Accurate and Scalable LLM Quantization with Loss-error-aware Grid** | ICLR 2025 | 2025 | [arXiv:2407.10032](https://arxiv.org/abs/2407.10032) | Replaces uniform grid with a loss-error-aware non-uniform grid from inverse-Hessian diagonal. Outperforms GPTQ/AWQ at 3/4-bit; scales to Llama-3.1-405B. |
| 2 | **CBQ: Cross-Block Quantization for LLMs** | ICLR 2025 | 2025 | [arXiv:2312.07950](https://arxiv.org/abs/2312.07950) | Joint cross-block reconstruction + adaptive LoRA rounding; first PTQ to keep W2A16 accuracy across multiple LLM families. |
| 3 | **GuidedQuant: LLM Quantization via End-Loss Guidance** | ICML 2025 | 2025 | [arXiv:2505.07004](https://arxiv.org/abs/2505.07004) | Uses end-loss gradient as guidance for weight rounding; SOTA for layer-wise output PTQ. |
| 4 | **SliM-LLM: Salience-Driven Mixed-Precision Quantization** | ICML 2025 | 2025 | [arXiv:2405.14917](https://arxiv.org/abs/2405.14917) | Group-wise mixed-precision allocation by salience score; SOTA mixed-precision PTQ in 2–4 bit. |
| 5 | **ResQ: Mixed-Precision Quantization of LLMs with Low-Rank Residuals** | ICML 2025 | 2025 | [arXiv:2412.14363](https://arxiv.org/abs/2412.14363) | Keeps 1/8 high-variance channels (found via activation PCA) at 8-bit, pushes the rest to 4-bit for W/A/KV. |
| 6 | **PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers** | ICLR 2025 | 2025 | [arXiv:2410.05265](https://arxiv.org/abs/2410.05265) | Prefixes high-frequency outlier tokens into the KV cache so static per-tensor quant matches dynamic per-token. Code: [`ChenMnZ/PrefixQuant`](https://github.com/ChenMnZ/PrefixQuant). |
| 7 | **Quantization Error Propagation: Revisiting Layer-Wise PTQ** | arXiv | 2025-04 | [arXiv:2504.09629](https://arxiv.org/abs/2504.09629) | Models and corrects error propagation between layers; plug-in for GPTQ/AWQ. |
| 8 | **D2Quant: Accurate Low-bit PTQ for LLMs** | NeurIPS 2025 | 2025 | [arXiv:2502.02546](https://arxiv.org/abs/2502.02546) | Dual-direction quantization with improved Hessian estimate; beats GPTQ and GPTAQ. |
| 9 | **LRQ: Optimizing PTQ via Learning Low-Rank Weight-Scaling Matrices** | arXiv | 2025 | [arXiv:2407.11534](https://arxiv.org/abs/2407.11534) | Replaces channel-wise scales with low-rank scaling matrices. |
| 10 | **Radio: Rate-Distortion Optimization for LLM Compression** | arXiv | 2025-05 | [arXiv:2505.03031](https://arxiv.org/abs/2505.03031) | Formulates PTQ as rate-distortion optimization for principled bit allocation. |
| 11 | **Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation** | arXiv | 2025-09 | [arXiv:2509.20214](https://arxiv.org/abs/2509.20214) | Fractional-bit quantizer palettes for efficient deployment. |
| 12 | **SingleQuant: Efficient Quantization of LLMs in a Single Pass** | arXiv | 2025 | [arXiv:2511.22316](https://arxiv.org/abs/2511.22316) | One-pass PTQ much faster than GPTQ while matching accuracy. |

## Weight + Activation PTQ (rotation-based, SmoothQuant successors)

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 13 | **SpinQuant: LLM Quantization with Learned Rotations** ⚓ | ICLR 2025 | 2024 | [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) | Learns rotation matrices via Cayley optimization to flatten activations; W4A4KV4 within 2.9 pts of FP on Llama-2-7B. Shipped with Meta's quantized Llama 3.2. Code: [`facebookresearch/SpinQuant`](https://github.com/facebookresearch/SpinQuant). |
| 14 | **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** ⚓ | NeurIPS 2024 | 2024 | [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Hadamard rotations enable W4A4KV4 near-lossless at 6/8-bit RTN. Foundation for most 2025 rotation work. |
| 15 | **OSTQuant: Orthogonal & Scaling Transformation-based Quantization** | ICLR 2025 | 2025 | [arXiv:2501.13987](https://arxiv.org/abs/2501.13987) | Learnable orthogonal + scaling transforms + KL-Top loss; retains 99.5% FP at W4, closes 32% of W4A4KV4 gap on Llama-3-8B. |
| 16 | **FlatQuant: Flatness Matters for LLM Quantization** | arXiv | 2025 | [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Reduces affine-transform overhead via Kronecker decomposition; learned affine transform per linear. |
| 17 | **DuQuant: Distributing Outliers via Dual Transformation** | NeurIPS 2024 | 2024 | [arXiv:2406.01721](https://arxiv.org/abs/2406.01721) | Block-rotation + permutation dual transform for outlier redistribution; strong W4A4. |
| 18 | **DFRot: Outlier-Free and Massive-Activation-Free Rotated LLMs** | arXiv | 2025 | [arXiv:2412.00648](https://arxiv.org/abs/2412.00648) | Refined rotation directly removing massive activations. |
| 19 | **DartQuant: Efficient Rotational Distribution Calibration** | arXiv | 2025-11 | [arXiv:2511.04063](https://arxiv.org/abs/2511.04063) | Calibration-only rotation learning at SpinQuant-level accuracy + QuaRot-level compute. |
| 20 | **ReSpinQuant: Subspace Residual Rotation Approximation** | arXiv | 2025 | [arXiv:2604.11080](https://arxiv.org/abs/2604.11080) | Decomposes SpinQuant rotation into low-rank + residual, slashing rotation overhead. |
| 21 | **ParoQuant: Pairwise Rotation Quantization** | NeurIPS 2025 era | 2025-11 | [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Sparse pairwise Givens rotations instead of dense Hadamard, matching QuaRot accuracy at lower cost. |
| 22 | **HALO: Hadamard-Assisted Lower-Precision Optimization** | NeurIPS 2025 | 2025 | — | Hadamard rotation for both training & inference, low-precision optimizer. |
| 23 | **QAM-W: Joint 2D Codebook Quantization via Hadamard + AWQ Scaling** | arXiv | 2025 | — | Combines Hadamard rotation with activation-aware scaling and a 2D codebook. |
| 24 | **InfoQuant: Shaping Activation Distributions for Low-Bit Quantization** | arXiv | 2025 | — | Reshapes activation distribution to be quantization-friendly. |

⚓ = late-2024 anchor included because every 2025 rotation paper cites it.
