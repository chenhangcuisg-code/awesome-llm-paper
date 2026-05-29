# Trend 9 — Systems, Kernels & Serving

Where quantization meets the actual deployment stack: **W4A8 kernels, FP8/MXFP4 tensor cores, vLLM / TensorRT-LLM / SGLang / llama.cpp** integration, and hardware co-design (FPGA, custom systolic arrays, energy-aware quantization).

## GPU kernels & serving

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **QServe: W4A8KV4 Quantization & System Co-design for Efficient LLM Serving** ⚓ | MLSys 2025 | 2024 | [arXiv:2405.04532](https://arxiv.org/abs/2405.04532) | QoQ algorithm + GPU kernels; **1.2–3.5× throughput over TensorRT-LLM**. Integrated into vLLM. Anchors most 2025 W4A8 work. |
| 2 | **LiquidGEMM: Hardware-Efficient W4A8 GEMM for High-Performance Serving** | arXiv | 2025-09 | [arXiv:2509.01229](https://arxiv.org/abs/2509.01229) | Open-source W4A8 GEMM beating QServe and Atom on A100/H100 across decode and prefill. |
| 3 | **FireQ: Fast INT4-FP8 Kernel + RoPE-aware Quantization** | arXiv | 2025-05 | [arXiv:2505.20839](https://arxiv.org/abs/2505.20839) | W4A8 kernel fusing RoPE rotation into the attention quantization path. Outperforms QServe Marlin kernels on Hopper. |
| 4 | **FlashInfer: Customizable Attention Engine for LLM Serving** | arXiv | 2025-01 | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) | Modular attention library supporting FP8/INT8 KV cache. Now the canonical attention backend for vLLM and SGLang. Code: [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer). |
| 5 | **LLMEasyQuant: Scalable Quantization for Distributed Inference** | arXiv (v5) | 2025 | [arXiv:2406.19657](https://arxiv.org/abs/2406.19657) | Distributed quantization toolchain splitting PTQ across nodes; integrates with vLLM / TGI. |

## Hardware co-design

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 6 | **HALO: Hardware-aware Quantization with Low Critical-Path-Delay Weights** | arXiv | 2025-02 | [arXiv:2502.19662](https://arxiv.org/abs/2502.19662) | Includes timing/energy in the quantization objective so chosen weight patterns map to faster, lower-power multipliers. **2.7× perf and 51% energy savings**. |
| 7 | **MixPE: Mixed-Precision Processing Element** | arXiv (2025 conf. ver.) | 2024-11 | [arXiv:2411.16158](https://arxiv.org/abs/2411.16158) | Fuses dequantization inside the MAC datapath, removing dequant overhead that bottlenecks W4A8/W4A16. |
| 8 | **FineQ: Software-Hardware Co-Design for Low-Bit Mixed-Precision Quantization** | arXiv | 2025-04 | [arXiv:2504.19746](https://arxiv.org/abs/2504.19746) | Efficient per-group bit-width metadata + systolic-array variant. **1.79× energy efficiency, 61% area reduction**. |
| 9 | **AccLLM: Algorithm-Hardware Co-Design for Long-Context Inference** | arXiv | 2025-05 | [arXiv:2505.03745](https://arxiv.org/abs/2505.03745) | Joint W2A8KV4 + Λ-shaped attention + reconfigurable FPGA engine. 128K context. |

## Edge / mobile / cross-platform deployment

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 10 | **Which Quantization Should I Use? Unified Eval of llama.cpp Quants** | arXiv | 2025 | [arXiv:2601.14277](https://arxiv.org/abs/2601.14277) | Benchmarks **all GGUF K-quants (Q2_K … Q8_0)** on Llama-3.1-8B-Instruct across reasoning/knowledge/IF/truthfulness + CPU prefill/decode throughput. |
| 11 | **Optimizing LLMs Using Quantization for Mobile Execution** | arXiv | 2025-12 | [arXiv:2512.06490](https://arxiv.org/abs/2512.06490) | End-to-end pipeline BitsAndBytes Q4 → GGUF Q4_K_M → llama.cpp on Android for Llama-3.2-3B. **68.66% size reduction**. |
| 12 | **Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs** | arXiv | 2025-08 | [arXiv:2508.06753](https://arxiv.org/abs/2508.06753) | System benchmark of quantized LLMs on consumer Intel hardware. |

## Training-time hardware

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 13 | **NVIDIA NVFP4 Technical Note (2025)** | NVIDIA | 2025 | NVIDIA Developer Blog | Block-wise 4-bit FP format with FP8 per-block + FP32 per-tensor scaling for Blackwell inference. The format underlying many 2025 FP4 papers. |
| 14 | **OCP Microscaling Formats (MX)** | OCP spec | 2024-2025 anchors | [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | The MXFP4 / MXFP6 / MXFP8 / MXINT8 formats adopted by NVIDIA Blackwell, AMD MI355X, Intel Gaudi3, used by Quartet / NVIDIA pretraining recipes / MicroMix / MR-GPTQ. |

⚓ = late-2024 anchor every 2025 system paper builds on.
