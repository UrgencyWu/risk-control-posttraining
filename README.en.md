# Risk-Control Post-Training for Large Language Models

<p align="center">
  <a href="./README.md">中文</a>
  &nbsp;·&nbsp;
  <a href="./README.en.md"><strong>English</strong></a>
  &nbsp;·&nbsp;
  <a href="https://urgencywu.github.io/risk-control-posttraining/"><strong>Live Interactive Showcase</strong></a>
</p>

## Project Summary

This project asks a concrete question: **can post-training make a compact instruction-tuned LLM learn useful sample-level risk ranking on small structured credit data, and does that value hold against a strong statistical baseline?**

The study uses `Qwen3.5-4B` and German Credit as the primary setting, with Australian Credit for multi-dataset validation. It covers data governance, LoRA SFT, DPO/SimPO, cost-sensitive decision rules, leakage-safe evaluation, and mechanism-level failure analysis.

| Core finding | Frozen result | Claim boundary |
|---|---:|---|
| LoRA SFT learns useful risk ranking | ROC-AUC `0.515 → 0.747`, a **+0.232** gain | SFT materially improves the zero-shot model |
| The statistical baseline remains slightly stronger | Logistic Regression `0.757` vs SFT `0.747` | The project claims competitiveness, not stable superiority |
| Cost-sensitive operating points matter | best SFT Cost `100`, LR `113`, zero-shot `140` | Three-seed SFT mean Cost is `123 ± 16`; one favorable seed is not treated as decisive |
| Label-level preference optimization reaches a boundary | **6 / 6** principal DPO/SimPO runs fail to exceed SFT | The conclusion is limited to the tested small-data, tabular, two-label short-answer setting |

This repository is an independent extension of the open-source [CALM](https://github.com/Dai-shen/CALM) project. CALM supplies the original task context and public datasets; this work rebuilds the experimental pipeline around `Qwen3.5-4B` and does not claim authorship of the original CALM paper, model, benchmark, or dataset collection.

## Research Questions

1. Can an instruction-tuned LLM learn generalizable risk ranking from tabular credit records represented as natural-language instructions?
2. Can LoRA SFT improve ranking, probability quality, and asymmetric business cost at the same time?
3. Can DPO/SimPO further optimize risk preferences without destroying sample-level ranking?
4. When post-training fails, is the root cause data construction, optimization instability, threshold selection, or an objective-task mismatch?

## Technical Route

```text
Audit the CALM repository and raw datasets
        ↓
Define a unified RiskDataset schema and label semantics
        ↓
Create deterministic splits and Normalized / ChatML records
        ↓
Build Majority / Logistic Regression / Qwen Zero-shot baselines
        ↓
Run Qwen3.5-4B LoRA SFT across three seeds
        ↓
Construct Oracle / Hard preference pairs
        ↓
Run DPO / SimPO / Cost-sensitive / Anchored pilots
        ↓
Freeze validation thresholds + C7 evaluation + log-probability audit
```

The route separates **representation learning** from the **decision operating point**. Training produces risk scores; the business threshold is selected only on validation predictions under `Cost = 5 × FN + 1 × FP`, then frozen before test evaluation.

## Implementation Details

### 1. Data and reproducibility

- Audited 10 financial-risk datasets covering credit scoring, fraud detection, financial distress, and claim analysis;
- unified `risk_label`, `task_type`, `target_type`, and `original_label` while preserving business semantics and provenance;
- froze German Credit's 1,000 records and 20 features into a `700/100/200` split with seed `10086`;
- built a German–Australian dataset with `1182/169/339` train/validation/test records;
- generated Normalized, ChatML SFT, Preference, and Evaluation layers;
- enforced manifests, schema checks, V1–V7 validation, and repeated SHA-256 reproducibility checks.

### 2. Model and training roles

| Model or checkpoint | Experimental role | Status |
|---|---|---|
| Majority | Class-imbalance lower bound | Implemented and verified |
| Logistic Regression | Strong non-LLM reference | Implemented and verified |
| Qwen3.5-4B Zero-shot | Unadapted LLM baseline | Implemented and verified |
| Qwen3.5-4B LoRA SFT, seeds `10086/42/7` | Primary post-training experiment | Implemented and verified |
| SFT seed `7` | Downstream checkpoint frozen by the validation PR-AUC rule | Implemented and verified |
| German–Australian Multi-SFT | Multi-dataset transfer experiment | Implemented; negative transfer on German |
| DPO / SimPO | Label-level preference-optimization stress test | Run; target not achieved |
| Cost-sensitive SFT / Anchored DPO / Risk-DPO | Weighting and anchoring controls | Failed, incomplete, or terminated |

SFT uses response-only causal-LM loss with LoRA `r=16`, `alpha=32`, `dropout=0.05`, attention and MLP projection targets, five epochs, and three random seeds. The runs were executed on `2 × RTX PRO 6000 Blackwell` GPUs.

### 3. Preference data and optimization

- German oracle preference data: 800 pairs;
- German hard preference data: 361 pairs selected from SFT ranking errors and low-confidence samples;
- German–Australian hard preference data: 549 training and 81 validation records;
- six principal experiments covering oracle/hard, single-/multi-dataset, and DPO/SimPO;
- additional Cost-sensitive SFT, Anchored DPO, and Risk-DPO pilots to test whether weighting or SFT anchoring prevents collapse.

### 4. Evaluation protocol

The unified evaluator reports Accuracy, Balanced Accuracy, Macro-F1, High-/Low-risk Recall, ROC-AUC, PR-AUC, NLL, Brier, ECE, confusion matrices, and asymmetric Cost.

```text
Cost = 5 × False Negative + 1 × False Positive
```

C7 selects thresholds only from committed validation predictions and applies the frozen threshold to corresponding test predictions. Test labels never participate in threshold search. See [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md).

## Experimental Data

### ROC-AUC trend line

![German Credit ROC-AUC trend](./docs/assets/roc_auc_trend.svg)

### German Credit test set, N = 200

| Model | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | High-risk recall | Low-risk recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B Zero-shot | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B Multi-SFT, German subset | 0.720 | 0.527 | 0.568 | 0.194 | **0.059** | 142 | 0.769 | 0.504 |

### SFT stability across three seeds

| Metric | Mean ± standard deviation |
|---|---:|
| Accuracy | `0.620 ± 0.04` |
| ROC-AUC | `0.747 ± 0.01` |
| High-risk recall | `0.821 ± 0.08` |
| Test Cost under validation-selected thresholds | `123 ± 16` |

### Six principal DPO/SimPO experiments

| Experiment | ROC-AUC | Decision behavior or Cost | Outcome |
|---|---:|---|---|
| German Oracle DPO | 0.706 | Cost 325; all low risk | Below SFT |
| German Oracle SimPO | 0.500 | Cost 325; random-level ranking | Below SFT |
| German Hard DPO | 0.478 | Cost 139; ranking below random | Below SFT |
| German Hard SimPO | 0.504 | Cost 135; random-level ranking | Below SFT |
| Multi-dataset DPO | German 0.517; Australian 0.666 | 100% high-risk decisions | Below SFT |
| Multi-dataset SimPO | German 0.525; Australian 0.648 | 100% high-risk decisions | Below SFT |

[`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) is the source of truth for final metrics. The complete experiment history is documented in [`docs/Progress_Report.md`](./docs/Progress_Report.md).

## Experimental Conclusions

1. **SFT is the only post-training method that consistently creates useful risk ranking.** The frozen checkpoint raises ROC-AUC from `0.515` to `0.747` and materially improves PR-AUC, NLL, and Brier.
2. **Logistic Regression remains the strongest ranking and probability-quality baseline.** Its `0.757` ROC-AUC exceeds SFT's `0.747`. The best SFT Cost is `100`, but the three-seed mean is `123 ± 16`, so stable superiority is not claimed.
3. **Cost-sensitive threshold selection is useful but cannot replace ranking quality.** Validation-selected thresholds reduce false-negative cost, while ROC-AUC, PR-AUC, and calibration metrics expose trivial all-high-risk behavior.
4. **Label-level DPO/SimPO is misaligned with sample-level risk ranking in this setting.** Samples within a class share the same short `low risk` / `high risk` response, so preference loss mainly moves the global label prior instead of directly constraining applicant-to-applicant ordering.
5. **The mechanism audit supports this negative result.** After DPO/SimPO, label log probabilities become extreme and `p(high)` collapses into a narrow range, producing single-class decisions.
6. **The conclusion is deliberately bounded.** The project does not claim that DPO or SimPO is ineffective for all classification, reasoning, or preference-learning tasks.

## Reproduction Guide

### 1. Reproduce frozen metrics on CPU

No GPU or model weights are required because committed validation and test prediction artifacts are version controlled.

```bash
git clone https://github.com/UrgencyWu/risk-control-posttraining.git
cd risk-control-posttraining
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -r requirements-eval.txt
python -m unittest discover -s tests -v
python -m src.evaluation.c7_final --output /tmp/c7_final_metrics.json
cmp outputs/c7_final_metrics.json /tmp/c7_final_metrics.json
```

### 2. Run the GPU workflow

```bash
python -m pip install -r requirements-train.txt
export RISK_CONTROL_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python -m src.baselines.majority
python -m src.baselines.logistic_regression
python -m src.baselines.qwen_zero_shot
sbatch scripts/sft_slurm.sh
sbatch scripts/dpo_train.sh
sbatch scripts/simpo_train.sh
```

Full training is not expected to reproduce bit-for-bit without the original hardware, model revision, and excluded weight files. The frozen C7 evaluation is fully artifact-reproducible.

### 3. Live showcase

- Live site: <https://urgencywu.github.io/risk-control-posttraining/>
- Showcase and deployment notes: [`docs/SHOWCASE.md`](./docs/SHOWCASE.md)
- Evaluation protocol: [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md)

---

This project is for research, education, and portfolio demonstration only. It must not be used for real credit, insurance, employment, or other high-impact decisions. See [`NOTICE`](./NOTICE) and [`LICENSE`](./LICENSE) for provenance and licensing details.
