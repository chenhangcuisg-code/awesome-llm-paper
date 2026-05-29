# Trend 2 — Sub-4-bit & BitNet Ternary

The 2025 line on **1-bit / 1.58-bit / 2-bit LLMs**. The big news: BitNet b1.58 became a real open-weights 2B model trained on 4T tokens, and ParetoQ established 2→3 bit as the representational phase transition.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **BitNet b1.58 2B4T Technical Report** | arXiv | 2025-04 | [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) | First open-sourced 2B-param native 1.58-bit LLM trained on 4T tokens. Matches BF16 open models of similar scale across reasoning/coding while drastically reducing inference memory. Code: [`microsoft/BitNet`](https://github.com/microsoft/BitNet). |
| 2 | **BitNet v2: Native 4-bit Activations via Hadamard Transformation** | arXiv | 2025-04 | [arXiv:2504.18415](https://arxiv.org/abs/2504.18415) | Extends b1.58 to native W1.58A4 by inserting online Hadamard before each linear projection. Lets models fully exploit FP4/INT4 tensor cores on Blackwell. |
| 3 | **Bitnet.cpp: Efficient Edge Inference for Ternary LLMs** | ACL 2025 | 2025-02 | [arXiv:2502.11880](https://arxiv.org/abs/2502.11880) | CPU/edge inference engine for 1.58-bit BitNet with custom LUT kernels. 5–7× speedup over llama.cpp Q4_0 on ARM / Apple Silicon. |
| 4 | **BitNet Distillation** | arXiv | 2025-10 | [arXiv:2510.13998](https://arxiv.org/abs/2510.13998) | Distills full-precision Llama-3-class models into 1.58-bit BitNet students without pretraining from scratch. Enables retrofitting BitNet to existing open models. |
| 5 | **An Extra RMSNorm Is All You Need for Fine-Tuning to 1.58 Bits** | arXiv | 2025-05 | [arXiv:2505.08823](https://arxiv.org/abs/2505.08823) | Simple recipe to QAT-finetune existing FP LLMs into 1.58-bit. |
| 6 | **Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity** | arXiv | 2025 | [arXiv:2603.05168](https://arxiv.org/abs/2603.05168) | Ternary BitNet weights admit 2:4 sparsity with minimal loss — compounded 1.58-bit + 50% sparse inference on Hopper/Blackwell. |
| 7 | **ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization** | arXiv | 2025-02 | [arXiv:2502.02631](https://arxiv.org/abs/2502.02631) | Unified framework comparing 1 / 1.58 / 2 / 3 / 4-bit. Reveals **2→3 bit representational phase transition**; W2 within 3.4 pts of FP. Identifies ternary as Pareto-optimal. |
| 8 | **QuEST: Stable Training of LLMs with 1-Bit Weights and Activations** | arXiv | 2025-02 | [arXiv:2502.05003](https://arxiv.org/abs/2502.05003) | Hadamard-rotation + trust-estimator gradients enable stable native training from 1-bit up to 8-bit. Pareto-optimal: 4-bit. |
| 9 | **PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression** | NeurIPS 2024 ext. | 2025 | [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) | Pareto-optimal vector quantization at ~2 bits/param for Llama-2. |
| 10 | **BitDistiller: Sub-4-Bit LLMs via Self-Distillation** | ACL 2024/25 | 2025 | [arXiv:2402.10631](https://arxiv.org/abs/2402.10631) | Self-distillation + asymmetric clipping for sub-4-bit. |
| 11 | **Bit-by-Bit: Progressive QAT with Outlier Channel Splitting** | arXiv | 2025 | [arXiv:2604.07888](https://arxiv.org/abs/2604.07888) | Progressive low-bit QAT for stability. |
| 12 | **SignRoundV2: Closing the Gap in Extremely Low-Bit PTQ** | arXiv | 2025-12 | [arXiv:2512.04746](https://arxiv.org/abs/2512.04746) | Improved signed rounding for sub-4-bit weight PTQ; closes most of the gap to QAT at a fraction of compute. |
| 13 | **When Are 1.58 Bits Enough? Bottom-up Exploration of BitNet Quantization** | arXiv | 2024-11 / 2025 | [arXiv:2411.05882](https://arxiv.org/abs/2411.05882) | Identifies regimes where ternary is sufficient. |
| 14 | **Scaling Law for Quantization-Aware Training** | arXiv | 2025-05 | [arXiv:2505.14302](https://arxiv.org/abs/2505.14302) | Empirical QAT scaling laws over model size, tokens, and bit-width. Predicts QAT-recovery vs PTQ degradation. |
