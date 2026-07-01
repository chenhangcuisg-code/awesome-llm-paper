# 01 — Interpretable Features

Extracting **human-interpretable features** from a model's internal activations. The dominant tool is the **sparse autoencoder (SAE)**, which decomposes the polysemantic residual stream into a large dictionary of sparse, often monosemantic, concept directions.

| # | Title | Venue | Year | Link | Summary |
|---|---|---|---|---|---|
| 1 | **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet** | Anthropic (Transformer Circuits) | 2024-05 | [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | Scales sparse autoencoders to a production-grade model (Claude 3 Sonnet) for the first time, extracting millions of interpretable features spanning concrete entities, abstract concepts, code constructs, and safety-relevant behaviors (deception, sycophancy, bias, dangerous content). Shows features are multilingual/multimodal, causally steer generation when clamped, and open a path to monitoring and controlling model behavior via its latent features. Landmark result that SAE-based interpretability generalizes from toy models to frontier LLMs. |
