# Week 01: Notes & Experiment Log

## Topic: Semantic Caching Blindspots in LLM APIs
- **Start Date:** 2026-09-17
- **Target Completion:** 2026-09-20

---

## Paper Summaries & Key Insights

### 1. GPTCache (Bang et al., 2023)
- Modern semantic caches use vector similarity (often cosine or Euclidean distance) to match incoming user queries to stored previous queries.
- Most implementations set a default similarity threshold around $0.80$ to $0.85$.
- *Insight:* High thresholds reject too many valid queries; low thresholds create severe false positive hazards.

### 2. Anisotropy in LM Representations (Timkey & van Schijndel, 2021)
- Deep transformer embeddings occupy a narrow cone in high-dimensional space rather than an isotropic sphere.
- Because of this "cone effect," random sentences often share baseline cosine similarity of $0.60 - 0.75$, and topical sentences with opposing meanings easily reach $0.88 - 0.94$.

---

## Hypotheses to Test
1. **Hypothesis 1:** The cosine similarity distributions of true paraphrases and logical negations overlap significantly between $0.88$ and $0.93$.
2. **Hypothesis 2:** Setting a threshold high enough to prevent 100% of negation false hits will degrade legitimate cache hit rates to under $10\%$.
3. **Hypothesis 3:** A lightweight 2-stage verification filter (token-level negation check or micro-NLI) can achieve $<5\text{ms}$ latency while restoring $>90\%$ true hit rate with $0\%$ false hits.

---

## Experiment Log

### Day 1: Benchmark Dataset & Baseline Cache Engine (2026-09-17)
- **Dataset Constructed**: 300 categorized query pairs (100 True Paraphrases, 75 Logical Negations, 75 Entity Substitutions, 50 Numerical/Temporal Inversions).
- **Models Evaluated**: `all-MiniLM-L6-v2` (384-dim) and `BAAI/bge-small-en-v1.5` (384-dim).
- **Empirical Similarity Distribution Results**:
  - **`all-MiniLM-L6-v2`**:
    - True Paraphrases: Mean = 0.887, Median = 0.908, Range = [0.638, 0.999]
    - Logical Negations: Mean = 0.865, Median = 0.875, Range = [0.686, 0.985]
    - Entity Substitutions: Mean = 0.812, Median = 0.822, Range = [0.551, 0.983]
    - Numerical / Temporal: Mean = 0.851, Median = 0.857, Range = [0.640, 0.978]
  - **`BAAI/bge-small-en-v1.5`**:
    - True Paraphrases: Mean = 0.893, Median = 0.901, Range = [0.724, 1.000]
    - Logical Negations: Mean = 0.872, Median = 0.879, Range = [0.710, 0.992]
    - Entity Substitutions: Mean = 0.829, Median = 0.835, Range = [0.612, 0.999]
    - Numerical / Temporal: Mean = 0.870, Median = 0.876, Range = [0.697, 0.996]

- **Threshold Performance at Default $\tau = 0.85$ (`bge-small-en-v1.5`)**:
  - True Hit Rate (Recall on Safe Queries): **74.0%**
  - Fatal False Hit Rate (Collisions on Unsafe Queries): **62.0%** (124 out of 200 dangerous queries falsely cached!)
  - Critical Severity Collisions (Medical/Financial Inversions): **31 lethal collisions** at $\tau = 0.85$.
  - Setting threshold to $\tau = 0.98$ drops True Hit Rate to **3.0%** while STILL permitting **5.5%** fatal false hits!

---

## Surprises & Takeaways

1. **Hypothesis 1 Confirmed (Extreme Overlap)**:
   - Deep transformer embeddings suffer from severe semantic blindspots on negations, entity reversals, and metric changes.
   - Reversing roles ("landlord to tenant" vs "tenant to landlord") achieves a **0.9986 cosine similarity** in BGE-Small!
   - Adding a legal negation ("without consent" vs "with consent") yields **0.9921 cosine similarity**!
2. **Hypothesis 2 Confirmed (The Similarity Threshold Paradox)**:
   - There is NO single cosine threshold $\tau$ that separates true paraphrases from logical inversions.
   - Setting $\tau = 0.98$ annihilates cache utility (TPR = 3%) without achieving zero false hits (FPR = 5.5%).
3. **Implication for Day 2**:
   - Single-stage vector caching in production LLM gateways is inherently hazardous without Stage-2 micro-verification (NLI / polarity check).

