# LLM Papers in Top-Tier Journals

This folder catalogs **peer-reviewed LLM (large language model) papers** that have appeared in top-tier scientific journals — Nature, Science, Cell, NEJM, and their major sub-journals.

## Files

| File | Coverage | Papers |
|---|---|---|
| [`nature.md`](./nature.md) | **Nature** family — Nature, Nature Machine Intelligence, Nature Medicine, Nature Computational Science, Nature Methods, Nature Biotechnology, Nature Communications, Nature Human Behaviour, Nature Biomedical Engineering, Nature Genetics, Nature Reviews, Nature Health, npj Digital Medicine, npj Computational Materials, Humanities & Social Sciences Communications, etc. | **148** |
| [`science.md`](./science.md) | **Science** family — Science, Science Advances, Science Robotics | **39** |
| [`cell-nejm.md`](./cell-nejm.md) | **Cell** family + **NEJM / NEJM AI** + **Lancet** family + **JAMA** family | **60** |
| **Total verified entries** | | **247** |

> Coverage updated across three deep-research passes:
> - **Pass 1** (initial): 139 LLM-explicit entries across Nature/Science/Cell/NEJM/Lancet/JAMA.
> - **Pass 2** (+61): DeepMind flagships (AlphaTensor / AlphaCode / AlphaGeometry / AlphaProof / AlphaFold-3 / GraphCast / GenCast / Co-Scientist), Earth foundation models (Pangu-Weather, Aurora), reasoning models (DeepSeek-R1 in Nature, Humanity's Last Exam), AI-safety perspectives (Bengio/Hinton/Russell "Managing extreme AI risks" in Science 2024), evaluation-crisis papers (Densing law, MaCBench, evaluation illusion, CREOLA, CSEDB), persuasion / cognition / society LLM papers (Centaur, GPT-4 persuasion, conspiracy debate, scientific writing impact), recent clinical RCTs (Therabot in NEJM AI, Goh 2025 in Nature Medicine, Pakistan RCT, PreA primary→specialist RCT).
> - **Pass 3** (+47): **transformer foundation models** not explicitly labelled "LLM" (AlphaFold-2, RoseTTAFold, AlphaMissense, AlphaStar, Enformer, Borzoi, scBERT, RETFound, CHIEF, Prov-GigaPath, Virchow, TITAN, PanDerm, MedSAM, BiomedGPT, OpenFold, Evo 2, MoLFormer-XL, NequIP/Allegro/MACE/DPA-2, ECGFounder) and **LLM-agent / agentic-learning / autonomous-system** papers (**CICERO in Science 2022** — the canonical LLM-agent paper; Robin, SPARK, BioMedAgent, GeneAgent, multimodal AMIE, MAC, LLM-RDF, autonomous-chemistry robots, CRESt, AILA/AFMBench, RoboVLMs, narrow-task → broad-misalignment safety paper).
>
> **Intentional gaps**: (1) Anthropic mechanistic-interpretability work is still arXiv / Transformer Circuits Thread only — the only Nature touch is the *News & Views* on Sleeper Agents. (2) Toolformer, ReAct, SayCan, PaLM-E, RT-2, OpenVLA, π0, RLHF / DPO / GRPO formalisms, WebArena / GAIA / SWE-bench, generative agents (Park) — all remain conference / arXiv venues, not Nature/Science. (3) AlphaProteo and AlphaFold-Multimer remain blog / bioRxiv. (4) AlphaGo / AlphaZero / MuZero / AlphaDev are in Nature/Science but their architectures are CNN/ResNet+MCTS, not transformers — excluded from the transformer-foundation list (though they could merit a separate "RL milestones" file if you want).

## Methodology

Papers are included if they:
1. Centrally feature an LLM (transformer-based language model, foundation model, instruction-tuned chat model, or LLM agent) as either the **subject of study** or the **primary tool**;
2. Are published or accepted in a Nature / Science / Cell / NEJM / Lancet / JAMA family journal;
3. Cover the LLM era (2020 onwards), with priority on 2023–2026.

Each entry provides: **journal · year · title · link (DOI/URL) · authors · 2-3 sentence English summary**.

## Status

This curation is updated periodically. Contributions welcome via pull request.
