# Awesome LLM Paper

A curated collection of LLM papers — split into two complementary tracks:

| Folder | Scope |
|---|---|
| [`must-read/`](./must-read/README.md) | The classics every LLM researcher should read. **Inherits and extends [AlphaLab-USTC/Must-Read-LLM-Papers](https://github.com/AlphaLab-USTC/Must-Read-LLM-Papers)** — pretraining, scaling laws, alignment, RL, agents, reasoning, multimodal, efficient inference, etc. |
| [`top-journals/`](./top-journals/README.md) | LLM works that have appeared in top-tier journals — **Nature**, **Science**, **Cell**, **NEJM** (and their sub-journals). Includes title, journal, year, DOI/URL, and an English summary. |

> The `must-read/` folder mirrors the upstream curation; the `top-journals/` folder is original research compiled here, with a focus on LLM-for-science and LLM-for-medicine peer-reviewed at the top venues.

## Why split this way

Most "awesome" lists for LLMs are arXiv-heavy. But when LLMs touch science and medicine, the strongest validation often comes from journal peer review — Nature / Science / Cell / NEJM and their family imprints. This repo keeps both kinds of literature side by side, so you can answer two different questions:

- *"What should I read to learn how LLMs work?"* → `must-read/`
- *"Where has LLM work been peer-reviewed at top scientific venues?"* → `top-journals/`

## Layout

```
awesome-llm-paper/
├── must-read/                 # Classics (forked + maintained)
│   └── README.md
└── top-journals/              # Top-tier journal LLM papers
    ├── README.md              # Index & methodology
    ├── nature.md              # Nature + Nature Machine Intelligence + Nature Medicine + ...
    ├── science.md             # Science + Science Advances + Science Robotics + ...
    └── cell-nejm.md           # Cell + NEJM + Lancet + JAMA families
```

## Contributing

PRs welcome for:
- Missing journal LLM papers (esp. 2024–2026 publications)
- Better English summaries
- Corrections to DOIs / metadata
- Adding emerging journal venues that publish LLM work

## License

Inherits the upstream [AlphaLab-USTC/Must-Read-LLM-Papers](https://github.com/AlphaLab-USTC/Must-Read-LLM-Papers) license for the `must-read/` portion. The `top-journals/` curation is provided as-is for academic reference; please cite the original papers.
