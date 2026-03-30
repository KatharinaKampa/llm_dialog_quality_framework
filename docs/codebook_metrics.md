# Codebook: Communication Quality Metrics

This codebook documents all communication quality metrics used in the project
"Können LLMs gut diskutieren?" (Kampa, 2026).

Metrics are organized into two groups:
1. **LLM evaluation metrics** (8 scales, Likert 1-7), used in Study A for both
   LLM-based evaluation and human ratings
2. **Rule-based sub-metrics** (15 metrics), computed automatically during dialog
   generation; used for pipeline quality assurance and exploratory analyses (H6/H7)


## 1. LLM Evaluation Metrics (Likert 1-7)

All eight metrics are rated on a 7-point Likert scale:
**1 = strongly disagree, 4 = neither agree nor disagree, 7 = strongly agree**

The same scale and item wording are used for both the LLM communication expert
(Study A, 3 runs, median aggregation) and the human raters (Study A, 2 raters
per dialog, mean aggregation).

### Scale Items

| Metric | Column name | Item wording |
|---|---|---|
| Clarity | `clarity` | "The dialog is clear, easy to understand, and expressed in unambiguous language." |
| Relevance | `relevance` | "Each turn in the dialog responds meaningfully to the previous one and stays on topic." |
| Truthfulness | `truthfulness` | "The dialog is internally consistent, without contradictions or misleading statements." |
| Logic / Coherence | `logic_coherence` | "The dialog presents coherent reasoning, with arguments that follow a logical structure." |
| Respect / Appreciation | `respect_appreciation` | "The dialog maintains a respectful tone, acknowledging the partner's contributions appropriately." |
| Relational Appropriateness | `relational_appropriateness` | "The dialog handles the interpersonal relationship appropriately, without tension, dominance, or relational violations." |
| Feedback / Depth | `feedback_depth` | "The dialog builds meaningfully on the partner's statements and contains sufficient conversational depth." |
| Overall Quality | `overall_quality` / `overall` | "Overall, the dialog shows high communication quality." |

### Column Names by Dataset

| Metric | Human ratings (Study A) | LLM runs | LLM medians |
|---|---|---|---|
| Clarity | `clarity` | `llm_clarity` | `llm_median_clarity` |
| Relevance | `relevance` | `llm_relevance` | `llm_median_relevance` |
| Truthfulness | `truthfulness` | `llm_truthfulness` | `llm_median_truthfulness` |
| Logic / Coherence | `logic_coherence` | `llm_logic_coherence` | `llm_median_logic_coherence` |
| Respect / Appreciation | `respect_appreciation` | `llm_respect_appreciation` | `llm_median_respect_appreciation` |
| Relational Appropriateness | `relation_appropriateness` | `llm_relational_appropriateness` | `llm_median_relational_appropriateness` |
| Feedback / Depth | `feedback_depth` | `llm_feedback_depth` | `llm_median_feedback_depth` |
| Overall Quality | `overall` | `llm_overall_quality` | `llm_median_overall_quality` |

### Theoretical Grounding

| Metric | Communication theory |
|---|---|
| Clarity | Grice (maxim of manner) |
| Relevance | Grice (maxim of relation) |
| Truthfulness | Grice (maxim of quality) |
| Logic / Coherence | Watzlawick (axiom 4: digital communication) |
| Respect / Appreciation | Rogers (unconditional positive regard) |
| Relational Appropriateness | Schulz von Thun (relationship dimension) |
| Feedback / Depth | Hargie (communicative competence) |
| Overall Quality | Composite judgment across all dimensions |

### LLM Quality Index (Study B, H6)

For Study B hypothesis H6, a composite **LLM Quality Index** is computed as the
mean of 6 communicative process metrics:

```
llm_quality_index = mean(clarity, relevance, logic_coherence,
                         respect_appreciation, relational_appropriateness,
                         feedback_depth)
```

**Excluded from index:**
- `truthfulness`: measures factual agreement with DGE dietary guidelines
  (content correctness), not communicative process quality
- `overall_quality`: aggregated overall judgment that already summarizes the
  other dimensions; inclusion would cause double-weighting


## 2. Rule-Based Sub-Metrics (15 metrics)

Rule-based sub-metrics are computed automatically in the dialog generation
notebooks (`GOOD_Dialogs_MA.ipynb`, `BAD_Dialogs_MA.ipynb`) using
pre-trained NLP models. They serve as an additional quality signal during
dialog generation and as exploratory variables in H6/H7.

### Scale

All sub-metrics are normalized to the range **[0, 1]** unless noted otherwise.
Higher values always indicate better communication quality.

### Grice's Maxims

| Sub-metric | Column | Method | Description |
|---|---|---|---|
| Precision | `grice_precision` | BERTScore F1 (roberta-base) | Lexical and semantic precision of turns relative to context |
| Relevance | `grice_relevance` | Cosine similarity (all-MiniLM-L6-v2) | Semantic relevance of each turn to the preceding turn |
| Clarity | `grice_clarity` | Flesch Reading Ease (textstat) | Linguistic clarity; higher = easier to read |
| Truthfulness | `grice_truth` | NLI cross-encoder (cross-encoder/nli-deberta-v3-large) vs. DGE guidelines | Factual consistency with nutritional reference knowledge |

> **Note (practice notebooks):** In `Practice_Dialogs_GOOD.ipynb` and
> `Practice_Dialogs_BAD.ipynb`, `grice_truth` is implemented as BERTScore F1
> (instead of NLI vs. DGE) due to computational constraints.

### Schulz von Thun's Four-Sided Model

| Sub-metric | Column | Method | Description |
|---|---|---|---|
| Self-disclosure | `thun_self` | Subjectivity score (TextBlob) | Degree of personal perspective expressed |
| Appeal | `thun_appeal` | Zero-shot classification (deberta-v3-large-zeroshot) | Presence of explicit requests or calls to action |
| Relationship | `thun_relation` | Sentiment analysis (twitter-roberta-base-sentiment) | Positive/negative relational tone |
| Factual content | `thun_factual` | Objectivity score (1 − subjectivity, TextBlob) | Degree of factual, objective content |

### Watzlawick's Axioms

| Sub-metric | Column | Method | Description |
|---|---|---|---|
| Logic consistency | `watzlawick_logic` | Zero-shot classification (deberta-v3-large-zeroshot) | Logical consistency and coherence across turns |
| Relational disturbance | `watzlawick_disturbance` | Turn-to-turn cosine similarity (paraphrase-multilingual-MiniLM-L12-v2) | Low value = high disturbance (sudden topic shifts) |

> **Note (practice notebooks):** `watzlawick_disturbance` is implemented as
> upper-triangle cosine similarity (instead of turn-to-turn sequence) in the
> practice dialog notebooks.

### Rogers' Therapeutic Conditions

| Sub-metric | Column | Method | Description |
|---|---|---|---|
| Empathy | `rogers_empathy` | Cosine similarity between agent responses (all-MiniLM-L6-v2) | Degree to which agents acknowledge each other's perspective |
| Congruence | `rogers_congruence` | Sentiment consistency across turns (TextBlob) | Consistency between expressed attitude and content |
| Respect | `rogers_respect` | NLI cross-agent (cross-encoder/nli-deberta-v3-large) | Degree of respectful, non-contradictory responding |

> **Note (practice notebooks):** `rogers_respect` is implemented as full-text
> sentiment (instead of NLI cross-agent) in the practice dialog notebooks.

### Hargie's Communicative Competence

| Sub-metric | Column | Method | Description |
|---|---|---|---|
| Feedback quality | `hargie_feedback` | Combined cosine similarity (all-MiniLM-L6-v2 + paraphrase-multilingual-MiniLM-L12-v2) | Degree to which turns build on prior content |
| Clarity of expression | `hargie_clarity` | Flesch Reading Ease + sentence length penalty | Communicative clarity from a competence perspective |


## 3. Study B - Dialog Quality Items (Human Ratings)

In Study B, participants rated the dialog after reading it using five items
(Likert 1-7, post-measurement only). These are distinct from the Study A LLM/human
evaluation scales and are used as the mediator variable in H5.

| Item | Column | Wording |
|---|---|---|
| Clarity | `dq_clarity` | "The dialog was clear and easy to follow." |
| Relevance | `dq_relevance` | "The dialog focused on relevant aspects of the recipe." |
| Logic | `dq_logic` | "The dialog was logically structured and coherent." |
| Respect | `dq_respect` | "The discussants treated each other respectfully." |
| Coherence | `dq_coherence` | "The discussants responded to each other in a meaningful way." |
| Overall (index) | `dq_mean` | Mean of the 5 items above (computed variable) |

**Manipulation check item** (single item, not part of `dq_mean`):

| Item | Column | Wording |
|---|---|---|
| Overall communication quality | `manip_check` | "How good was the communication overall? (1 = very poor, 7 = excellent)" |
