# Trend 5 — KV Cache & Long-Context Quantization

In 128K+ context deployments, the **KV cache dominates inference memory**. 2025 work converged on (i) rotation-based 2-bit KV, (ii) vector-quantized KV that commutes with RoPE, (iii) asymmetric K vs V precision, and (iv) attention-sink / outlier preservation.

## KV cache quantization

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **KVQuant: Towards 10 Million Context Length LLM Inference** ⚓ | NeurIPS 2024 / v6 2025 | 2024-01 | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) | Pre-RoPE per-channel keys + attention-sink-aware quantization + non-uniform datatypes. Demonstrates **10M-context inference on a single A100**. |
| 2 | **KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache** | arXiv | 2025-02 | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) | Layer-wise mixed precision from theoretical sensitivity analysis; nearly lossless 3.25-bit KV on Llama-3.1-8B-Instruct. |
| 3 | **RotateKV: 2-Bit KV Cache Quantization via Outlier-Aware Adaptive Rotations** | IJCAI 2025 | 2025-01 | [arXiv:2501.16383](https://arxiv.org/abs/2501.16383) | FWHT-based outlier redistribution before 2-bit KV. <0.3 PPL drop, 3.97× memory reduction, 5.75× batch. |
| 4 | **CommVQ: Commutative Vector Quantization for KV Cache** | arXiv | 2025-06 | [arXiv:2506.18879](https://arxiv.org/abs/2506.18879) | Vector quantization with a codebook that **commutes with RoPE**, avoiding double-quantization of position. Near-lossless 2-bit / viable 1-bit at long context. |
| 5 | **Cache Me If You Must: Adaptive KV Quantization** | arXiv | 2025-01 | [arXiv:2501.19392](https://arxiv.org/abs/2501.19392) | Adaptive per-token KV precision. |
| 6 | **Quantize What Counts: More for Keys, Less for Values** | arXiv | 2025-02 | [arXiv:2502.15075](https://arxiv.org/abs/2502.15075) | Theoretical + empirical evidence: K-cache deserves higher precision than V-cache. |
| 7 | **AsymKV: Homogeneous Keys, Heterogeneous Values** | NeurIPS 2025 | 2025-06 | [arXiv:2506.05410](https://arxiv.org/abs/2506.05410) | Layer-wise asymmetric K vs V quantization for long-context. Code: [`the-scale-lab/Asymkv`](https://github.com/the-scale-lab/Asymkv). |
| 8 | **BitDecoding: Unlocking Tensor Cores for Long-Context with Low-Bit KV** | arXiv | 2025-03 | [arXiv:2503.18773](https://arxiv.org/abs/2503.18773) | Maps low-bit KV decode onto tensor cores. |
| 9 | **KITTY: Accurate and Efficient 2-Bit KV Cache Quantization** | NeurIPS 2025 | 2025-11 | [arXiv:2511.18643](https://arxiv.org/abs/2511.18643) | Token-grouped 2-bit KV with outlier handling. |
| 10 | **InnerQ: Hardware-Aware Tuning-Free KV Cache Quantization** | arXiv | 2025 | — | Hardware-aware grouping for KV without tuning. |
| 11 | **PolyKV: Shared Asymmetrically-Compressed KV Pool for Multi-Agent Inference** | arXiv | 2025 | — | Cross-request shared KV pool with asymmetric compression. |
| 12 | **Don't Waste Bits! Adaptive KV Quantization for On-Device LLMs** | arXiv | 2025 | — | Per-token adaptive KV bit allocation for edge devices. |

## Long-context (whole-pipeline)

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 13 | **MixKVQ: Query-Aware Mixed-Precision KV for Long-Context Reasoning** | arXiv | 2025-12 | [arXiv:2512.19206](https://arxiv.org/abs/2512.19206) | Per-token mixed precision driven by query saliency computed **post-RoPE**; preserves reasoning under 2-bit average. |
| 14 | **TTKV: Temporal-Tiered KV Cache for Long-Context** | arXiv | 2025 | [arXiv:2604.19769](https://arxiv.org/abs/2604.19769) | Two-tier cache: full-precision recent/hot tokens + quantized+sparsified cold tokens. 128K-context eval. |
| 15 | **FIER: Fine-Grained and Efficient KV Retrieval for Long-Context** | arXiv | 2025-08 | [arXiv:2508.08256](https://arxiv.org/abs/2508.08256) | Combines token-level retrieval with quantization-aware scoring under attention-sink preservation. |
| 16 | **AccLLM: Algorithm-Hardware Co-Design for Long-Context Inference** | arXiv | 2025-05 | [arXiv:2505.03745](https://arxiv.org/abs/2505.03745) | Joint W2A8KV4 + Λ-shaped attention + reconfigurable FPGA engine for 128K-context. |

⚓ = 2024 anchor every 2025 KV paper cites.
