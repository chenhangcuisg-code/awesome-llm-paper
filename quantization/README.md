# LLM Quantization Papers (2025+)

Curated catalog of **LLM quantization papers published in 2025 or later**, organized by mainstream trends. Compiled from two parallel deep-research passes covering both PTQ algorithms and QAT / system-level work.

A handful of late-2024 anchor papers (DeepSeek-V3, SpinQuant, QuaRot, EfficientQAT, KVQuant, ParetoQ-base, Outlier-Safe Pre-Training, AMXFP4) are kept because every 2025 line of work cites them as the direct prerequisite.

## Mainstream trends in 2025

The 2025 LLM-quantization literature clusters around **eight mainstream directions**:

| # | Trend | File | Papers |
|---|---|---|---|
| 1 | **Rotation-based PTQ & algorithmic refinements** (SpinQuant / QuaRot / GPTQ / AWQ successors) | [`01-ptq-algorithms.md`](./01-ptq-algorithms.md) | 24 |
| 2 | **Sub-4-bit & BitNet ternary** (1/1.58/2-bit native and QAT) | [`02-extreme-low-bit-bitnet.md`](./02-extreme-low-bit-bitnet.md) | 14 |
| 3 | **FP4 & Microscaling** (MXFP4 / NVFP4 / Blackwell-era) — **fastest-growing trend** | [`03-fp4-microscaling.md`](./03-fp4-microscaling.md) | 15 |
| 4 | **FP8 large-scale pretraining** (DeepSeek-V3 wave) | [`04-fp8-training.md`](./04-fp8-training.md) | 7 |
| 5 | **KV cache & long-context quantization** (2-bit KV, 128K–10M context) | [`05-kv-cache-long-context.md`](./05-kv-cache-long-context.md) | 16 |
| 6 | **MoE quantization** (DeepSeek-V3 / Mixtral / Qwen-MoE) | [`06-moe-quantization.md`](./06-moe-quantization.md) | 9 |
| 7 | **Reasoning-model quantization** (DeepSeek-R1 / long-CoT) — **emerging trend** | [`07-reasoning-models.md`](./07-reasoning-models.md) | 6 |
| 8 | **QAT, QLoRA successors & quantized fine-tuning** | [`08-qat-finetuning.md`](./08-qat-finetuning.md) | 13 |
| 9 | **Systems, kernels & serving** (vLLM / TensorRT-LLM / SGLang / llama.cpp / hardware co-design) | [`09-systems-kernels.md`](./09-systems-kernels.md) | 14 |
| 10 | **Multimodal / Diffusion / VLM quantization** | [`10-multimodal-diffusion.md`](./10-multimodal-diffusion.md) | 8 |
| 11 | **Surveys & benchmarks** | [`11-surveys-benchmarks.md`](./11-surveys-benchmarks.md) | 13 |
| | **Total verified entries** | | **~135** |

## What changed in 2025 vs 2024

- **FP4 went from "experimental" to "default on Blackwell."** NVFP4 / MXFP4 inference and pretraining recipes (NVIDIA, IST Austria, Quartet I/II) make 4-bit Pareto-optimal across the FLOPs–accuracy frontier.
- **BitNet b1.58 became a real open model** (2B4T) and started spawning a distillation/sparse line — ternary is no longer a thought experiment.
- **DeepSeek-V3 (671B, native FP8)** triggered an industry-wide FP8 training wave (μnit scaling, Towards Fully FP8 GEMM, MOSS).
- **Rotation methods consolidated**: QuaRot/SpinQuant successors (DartQuant, ParoQuant, ReSpinQuant, OSTQuant, FlatQuant) target *cheaper* rotations rather than better rotations.
- **Reasoning-model compression emerged as its own subfield**: standard PTQ measurably degrades long-CoT before perplexity moves.
- **KV cache became *the* memory bottleneck** in 128K+ deployments → 2-bit KV with rotation/VQ (RotateKV, CommVQ, AsymKV) is now common.
- **Outlier handling moved upstream**: training-time fixes (Outlier-Safe Pre-Training, BitNet v2 Hadamard, attention-sink prefixing) replace inference-time scaling.

## Methodology

Papers were collected via WebSearch with WebFetch (DuckDuckGo HTML endpoint) fallback. Inclusion criteria:
1. Publication or arXiv timestamp in **2025 or later** (with a small number of 2024 anchors explicitly tagged).
2. Centrally about **quantization of LLMs** (or LLM-class transformer foundation models / VLMs / diffusion LLMs).
3. Either highly cited, frequently discussed in the community, or shipped as code/integration in mainstream serving frameworks.

Each entry: **title · authors · venue · year/month · arXiv/DOI link · 2-3 sentence English summary · official code (if any)**.

> ⚠️ Some entries surfaced with future-looking arXiv IDs (2601.x, 2602.x, 2603.x, 2604.x, 2605.x, 2606.x). These correspond to late-2025 / early-2026 submissions returned by the search index — treat them as recent but verify before citation.

## Framework integration cheat-sheet

| Method | vLLM | TensorRT-LLM | SGLang | llama.cpp |
|---|---|---|---|---|
| GPTQ / AWQ / Marlin | ✅ | ✅ | ✅ | partial |
| QServe (W4A8KV4) | ✅ | — | partial | — |
| SpinQuant | via HF | ✅ (Llama 3.2) | — | — |
| BitNet b1.58 | experimental | — | — | ✅ (bitnet.cpp) |
| FP8 (Hopper, E4M3/E5M2) | ✅ | ✅ (Transformer Engine) | ✅ | — |
| MXFP4 / NVFP4 (Blackwell) | ✅ | ✅ | ✅ (FlashInfer) | — |
| KVQuant / RotateKV | patches | — | — | — |
| FlashInfer attention | ✅ (default) | — | ✅ (default) | — |

## Useful aggregator repos

- [`pprp/Awesome-LLM-Quantization`](https://github.com/pprp/Awesome-LLM-Quantization)
- [`HuangOwen/Awesome-LLM-Compression`](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [`Zhen-Dong/Awesome-Quantization-Papers`](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
