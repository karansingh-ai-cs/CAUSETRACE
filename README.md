# CAUSETRACE
### Evaluating Causal Reasoning in Large Language Models

A lightweight system that extracts, evaluates, and 
stress-tests causal reasoning in LLM-generated responses.

---

## Motivation

Large Language Models generate causal explanations 
constantly — but how reliable is this reasoning?

This project investigates two core questions:
1. Do keyword-based methods accurately detect 
   causal relationships in LLM outputs?
2. Is LLM causal reasoning consistent across 
   multiple runs of the same question?

Inspired by research on causal explanation evaluation 
in AI systems — particularly the finding that 
LLM-based (blind) evaluation diverges significantly 
from similarity-based metrics.

---

## System Architecture
```
Input Question
      ↓
Layer 1: LLM Answer Generation (Groq API)
      ↓
Layer 2: Explicit Causal Extraction (keyword-based)
      ↓
Layer 3: Implicit Causality Detection (heuristic)
      ↓
Layer 4: Blind LLM Evaluation (GPT-Black style)
      ↓
Layer 5: Consistency Testing (3-run stability check)
      ↓
Final Report
```

---

## Key Findings

Tested on question: *"Why do antibiotics stop 
working over time?"*

| Metric | Result |
|--------|--------|
| Explicit causal pairs found | 2 |
| Implicit pairs flagged | 6 |
| Blind evaluation divergence rate | **67%** |
| Consistency score (3 runs) | **0.33** |
| Reasoning verdict | **UNSTABLE** |

**Finding 1:** Keyword-based detection flagged 3 
implicit pairs as causal. Blind LLM evaluation 
rejected 2 of them (67% divergence).

**Finding 2:** Same question asked 3 times produced 
different causal keywords in each run — 
consistency score of only 0.33 out of 1.0.

---

## Known Limitations

- Implicit causality detection is heuristic-based 
  and unreliable (high false positive rate)
- Keyword matching misses implicit causal 
  relationships entirely
- Consistency testing uses keyword overlap — 
  not semantic similarity
- Tested on limited question set

These limitations represent the open research 
problems this project aims to highlight.

---

## Setup
```bash
pip install groq python-dotenv
```

Create `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
python causetrace.py
```

---

## Built By

Independent researcher — pre-university level.
No institutional affiliation.
Built on personal laptop using API-based LLM.

*This project is an independent exploration of 
causal reasoning evaluation in LLMs, inspired by 
published research in the field.*
