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

## Experimental Results

Tested across 5 diverse questions spanning biology, 
economics, psychology, and astronomy.

| Question | Divergence | Consistency | Verdict |
|----------|-----------|-------------|---------|
| Antibiotics & resistance | 67% | 0.33 | UNSTABLE |
| Exercise & mental health | 100% | 0.00 | UNSTABLE |
| Economic recession | 100% | 0.25 | UNSTABLE |
| Sleep & memory | 100% | 0.00 | UNSTABLE |
| Stars at night | 100% | 0.00 | UNSTABLE |

**Average divergence rate: 93.4%**
**Average consistency score: 0.116 / 1.0**

### Key Findings

**Finding 1 — Keyword detection is unreliable:**
Across 5 questions, blind LLM evaluation rejected 
93.4% of pairs flagged by keyword-based implicit 
detection. Keyword overlap is not a valid proxy 
for causal relationship.

**Finding 2 — LLM causal reasoning is unstable:**
4 out of 5 questions produced a consistency score 
of 0.00 — meaning the same question asked 3 times 
produced completely different causal keywords each 
time. Only 1 question achieved partial stability 
(score: 0.33).

**Conclusion:** LLM-generated causal explanations 
are neither reliably detectable by surface methods 
nor consistent across multiple runs. This suggests 
fundamental limitations in how current LLMs 
represent causal knowledge.
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
