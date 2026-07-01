# Latent Knowledge

How knowledge is **encoded, organized, and steered** inside large models — the study of the *internal* representations rather than the input/output behavior.

This track collects work on **mechanistic interpretability**, **functional organization of neurons/features**, **latent behavioral directions** (refusal, safety), and **representational redundancy**. The unifying question: *what does a model actually store in its weights and activations, how is it structured, and how much of it matters?*

## Sub-categories

| File | Theme |
|---|---|
| [`01-interpretable-features.md`](./01-interpretable-features.md) | Extracting human-interpretable features / concepts from the residual stream (SAEs, monosemanticity). |
| [`02-functional-modularity.md`](./02-functional-modularity.md) | Emergent domain-specific neuron populations and brain-like functional organization; LLM↔neuroscience bridges. |
| [`03-latent-directions-and-safety.md`](./03-latent-directions-and-safety.md) | Behavioral directions in latent space — refusal-direction ablation, and where/when safety decisions are actually made. |
| [`04-representational-redundancy.md`](./04-representational-redundancy.md) | How much of a model's capacity is actually used — redundancy, pruning, and drop-then-recover analysis. |

## Why this track

Benchmarks tell you *what* a model does. Latent-knowledge research tells you *why* and *where* — which internal features fire, whether concepts are localized or distributed, which directions govern behavior, and how much capacity is genuinely load-bearing. This is the layer where safety, steering, editing, and compression all ultimately operate.
