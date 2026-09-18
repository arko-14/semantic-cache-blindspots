# Week 01: Semantic Caching Blindspots in LLM APIs

## Goal
At what cosine similarity threshold ($\tau$) does vector-based "Semantic Caching" (e.g., Redis, Chroma, GPTCache) in front of an LLM API transition from saving compute costs to serving disastrously incorrect responses on subtle logical negations, entity substitutions, and numerical changes? Can a lightweight 2-stage verification filter eliminate false hits with negligible latency overhead?

---

## Core Reading (Verified Sources)
1. **Bang et al. (2023)** — *["GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster and Cheaper Services"](https://arxiv.org/abs/2311.01723)* (arXiv:2311.01723).
2. **Timkey & van Schijndel (2021)** — *["All Bark and No Bite: Rethinking Anisotropy in LM Representations"](https://arxiv.org/abs/2109.04404)* (arXiv:2109.04404).
3. **Reimers & Gurevych (2019)** — *["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/abs/1908.10084)* (EMNLP 2019).

---

## 3-Day Plan

### Day 1: Benchmark Dataset & Baseline Cache Engine
- [x] Build `dataset/` with 300 categorized query pairs (true paraphrases, negations, entity/number swaps).
- [x] Implement `src/cache.py` with in-memory cosine similarity search and exact hash fallback.
- [x] Run initial pairwise cosine similarity distribution and verify overlap between true paraphrases and inversions.

### Day 2: Threshold Sweeps, Visualizations & 2-Stage Filter
- [x] Sweep threshold $\tau$ from 0.75 to 0.99 across all embedding models (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`).
- [x] Generate the 3 core publication plots:
  - [x] Plot 1: Cosine Similarity Overlap Histogram (True Paraphrase vs Negation).
  - [x] Plot 2: Semantic Cache ROC Curve (True Hit Rate vs Fatal False Hit Rate).
  - [x] Plot 3: Financial Break-Even / Cost-Benefit Matrix.
- [x] Implement the Stage-2 Micro-Verification Filter (NLI / Entity check) and measure latency vs accuracy recovery.

### Day 3: Polished Repo, README & Medium Blog Draft
- [x] Write the GitHub `README.md` with system architecture, embedded charts, and quickstart commands.
- [x] Clean code, ensure 100% reproducible execution (`python run_benchmark.py`).
- [x] Draft the Medium article for publication (*Level Up Coding* / *CodeToDeploy*).
