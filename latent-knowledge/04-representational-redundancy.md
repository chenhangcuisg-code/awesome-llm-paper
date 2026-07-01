# 04 — Representational Redundancy

How much of a model's capacity is **genuinely load-bearing**? These works probe redundancy by dropping components and measuring what actually degrades — and argue redundancy must be judged by downstream task success after recovery, not by parameter count or single-step loss.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models?** | arXiv | 2026 | [arXiv:2606.27755](https://arxiv.org/abs/2606.27755) | Investigates architectural redundancy in Vision-Language-Action (VLA) models, which inherit **oversized language backbones** from pretrained VLMs whose capacity far exceeds what short robotic instructions require. Argues redundancy can't be read off parameter count or single-step prediction loss — it must be measured by **closed-loop task success after recovery**. Finds current VLA benchmarks exert limited pressure on deep language grounding, and that future architectures should allocate capacity more deliberately across language, vision, and action. |
