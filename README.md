# Risk-Control Post-Training for Large Language Models

<p align="center">
  <a href="./README.md"><strong>English</strong></a>
  &nbsp;·&nbsp;
  <a href="./README.zh-CN.md">简体中文</a>
  &nbsp;·&nbsp;
  <a href="./docs/index.html">Interactive showcase / 交互式展示</a>
</p>

## Research Question

**Can post-training make a compact instruction-tuned LLM useful for cost-sensitive credit-risk classification, and where does it stop adding value relative to a strong statistical baseline?**

The study evaluates `Qwen3.5-4B` on German Credit, with Australian Credit used for transfer validation. It separates representation learning from decision-threshold selection and tests whether SFT, DPO, or SimPO improves sample-level risk ranking rather than merely shifting the global probability of the two labels.

## Four Quantitative Findings

| Finding | Frozen result | Interpretation |
|---|---:|---|
| **LoRA SFT learns useful risk ranking** | ROC-AUC `0.515 → 0.747` (**+0.232**) | SFT converts a near-random zero-shot ranking into a competitive risk model. |
| **Logistic Regression remains the honest ranking reference** | `0.757` vs SFT `0.747` | The remaining ROC-AUC gap is `0.010`; the project claims competitiveness, not stable superiority. |
| **Cost-sensitive operating-point selection matters** | best SFT Cost `100`, zero-shot `140`, Logistic Regression `113` | The best frozen SFT checkpoint lowers asymmetric test cost, but the three-seed SFT mean is `123 ± 16`, so a single favorable run is not treated as decisive. |
| **Label-level preference optimization reaches a method boundary** | **6 / 6** principal DPO/SimPO variants fail to exceed SFT | Oracle, hard-pair, single-dataset, and multi-dataset runs produce class-prior shifts or decision collapse instead of better sample-level ranking. |

## Project Contributions

1. **Reproducible post-training pipeline.** Defined a traceable risk-data schema, deterministic train/validation/test splits, ChatML conversion, preference records, manifests, and SHA-256 reproducibility checks.
2. **Leakage-safe, cost-sensitive evaluation.** Compared classical and LLM baselines with ROC-AUC, PR-AUC, calibration metrics, confusion matrices, and thresholds selected only from committed validation predictions.
3. **Verified SFT result with honest baseline comparison.** Ran three German Credit seeds and a German–Australian transfer experiment, showing that LoRA SFT is effective while preserving the stronger Logistic Regression ranking result.
4. **Mechanism-level negative result for DPO/SimPO.** Ran six principal preference-optimization variants and used log-probability audits to connect their failures to global label-prior movement rather than improved conditional risk separation.

This repository is an independent extension of the open-source [CALM project](https://github.com/Dai-shen/CALM). CALM supplies the original task context and public datasets; this project rebuilds the experimental pipeline around `Qwen3.5-4B` and does not claim authorship of the original CALM paper, model, benchmark, or dataset collection.

## Evidence and Implementation Status

| Workstream | Status | Verified outcome |
|---|---|---|
| Data schema and preprocessing | **Implemented and verified** | Deterministic normalized and ChatML datasets, frozen splits, manifests, validation checks, and SHA-256 reproducibility |
| Majority / Logistic Regression / Qwen zero-shot baselines | **Implemented and verified** | Unified cost-sensitive evaluation on the frozen German test set |
| Qwen3.5-4B LoRA SFT | **Implemented and verified** | Three German seeds plus a German–Australian multi-dataset run |
| Validation-selected decision threshold | **Implemented and verified** | Thresholds selected under `Cost = 5 × FN + 1 × FP`; no test-set operating-point tuning |
| Oracle and hard preference construction | **Implemented and verified** | Single- and multi-dataset preference records with frozen metadata |
| DPO and SimPO | **Run; target not achieved** | Six principal variants fail to exceed SFT and exhibit label-prior or decision collapse |
| Cost-sensitive SFT | **Run; target not achieved** | Training-time class weighting degrades ranking and pushes predictions toward high risk |
| Anchored DPO / Risk-DPO | **Incomplete or terminated** | Pilots expose validation-split and device-placement failures; no positive result is claimed |
| Final C7 evaluation and log-probability audit | **Implemented and verified** | CPU-only regeneration from frozen validation/test prediction artifacts, calibration metrics, confusion matrices, and mechanism evidence |

Large model and LoRA weight binaries are intentionally excluded. Source code, processed datasets, shell entry points, predictions, metrics, training logs, adapter metadata, and research reports are committed.

## Final Results

All headline metrics use [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) as the source of truth. C7 derives each operating point from committed validation predictions and then applies the frozen threshold to committed test predictions. It does not rerun model inference or search thresholds on test labels. See [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md).

### German Credit test set, N = 200

| Model | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | High-risk recall | Low-risk recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B zero-shot | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B multi-SFT, German subset | 0.720 | 0.527 | 0.568 | 0.194 | **0.059** | 142 | 0.769 | 0.504 |

### SFT stability across seeds

The three German SFT runs produce approximately:

| Metric | Mean ± standard deviation |
|---|---:|
| Accuracy | 0.620 ± 0.04 |
| ROC-AUC | 0.747 ± 0.01 |
| High-risk recall | 0.821 ± 0.08 |
| Validation-selected test Cost | 123 ± 16 |

The frozen seed-7 checkpoint obtains Cost 100, while the three-seed mean Cost is 123, above Logistic Regression's 113. Mean SFT ROC-AUC also remains below Logistic Regression's 0.757. The study therefore establishes that SFT is effective, but it does **not** claim statistically established superiority over Logistic Regression.

## Experimental Roles

| Model or checkpoint | Experimental role | Main finding |
|---|---|---|
| Majority classifier | Sanity-check lower bound | Obtains 67.5% accuracy from class imbalance but misses every high-risk case; Cost 325 |
| Logistic Regression | Strong non-LLM reference | Best final ranking and probability quality: ROC-AUC 0.757, NLL 0.552, Cost 113 |
| Qwen3.5-4B zero-shot | Unadapted LLM baseline | Near-random ranking, severe high-risk over-prediction, and poor calibration |
| Qwen3.5-4B LoRA SFT, seeds `10086`, `42`, `7` | Primary post-training experiment | Consistently creates useful risk-ranking ability; mean ROC-AUC 0.747 |
| Qwen3.5-4B SFT seed `7` | Frozen downstream SFT checkpoint | Selected by the predeclared validation PR-AUC rule, not by test performance; test ROC-AUC 0.747 and Cost 100 |
| Qwen3.5-4B multi-dataset SFT | Transfer and data-diversity experiment | Strong combined German–Australian ranking, but negative transfer on the primary German benchmark |
| Qwen3.5-4B DPO / SimPO | Preference-optimization stress test | Oracle, hard-pair, and multi-dataset variants all underperform SFT and often collapse toward one label |
| Cost-sensitive SFT / Anchored DPO / Risk-DPO pilots | Controls for asymmetric weighting and SFT anchoring | Cost weighting degrades ranking; anchored and risk-aware pilots do not yield a valid successful result |

## Implemented Pipeline

### 1. Data governance and reproducible preprocessing

```text
Raw data
  → Normalized risk schema
  → SFT ChatML records
  → Preference records
  → Evaluation records
```

Implemented components include:

- audit of 10 financial-risk datasets across credit scoring, fraud detection, financial distress, and claim analysis;
- unified `risk_label`, `task_type`, `target_type`, and `original_label` traceability;
- protected-attribute extraction where available;
- description-style and table-style prompt construction;
- deterministic train/validation/test splitting;
- manifests, schema validation, and repeated SHA-256 checks.

The frozen German Credit pipeline contains 1,000 records and 20 input features, maps original labels `1/2 → 0/1`, and uses a deterministic `700/100/200` split with seed `10086`. The German–Australian validation set contains 1,182 training, 169 validation, and 339 test records.

Primary evidence: [`convert_german.py`](./convert_german.py), [`data/processed/`](./data/processed), [`docs/RiskDataset_Schema.md`](./docs/RiskDataset_Schema.md), and [`docs/Progress_Report.md`](./docs/Progress_Report.md).

### 2. Cost-sensitive evaluation

The evaluator implements Accuracy, Balanced Accuracy, Macro-F1, high-/low-risk recall, ROC-AUC, PR-AUC, NLL, Brier score, expected calibration error, confusion matrices, and configurable false-negative/false-positive costs.

```text
Cost = 5 × false negatives + 1 × false positives
```

A false negative means a genuinely high-risk applicant is classified as low risk. The 5:1 cost ratio favors high-risk recall, while ranking and calibration metrics prevent an all-high-risk classifier from appearing useful.

Primary evidence: [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md), [`src/evaluation/metrics.py`](./src/evaluation/metrics.py), [`src/evaluation/c7_final.py`](./src/evaluation/c7_final.py), and [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json).

### 3. LoRA supervised fine-tuning

The final SFT experiments use `Qwen3.5-4B` with response-only causal-LM loss, LoRA rank 16, alpha 32, dropout 0.05, attention and MLP projection targets, five epochs, three German random seeds, and multi-GPU execution on 2 × RTX PRO 6000 Blackwell GPUs.

SFT is the only post-training method in this project that consistently creates useful risk discrimination.

Primary evidence: [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py), [`src/training/sft_multi.py`](./src/training/sft_multi.py), [`outputs/sft/`](./outputs/sft), and [`scripts/sft_slurm.sh`](./scripts/sft_slurm.sh).

### 4. Preference construction and optimization

The repository contains 800 German oracle preference pairs, 361 German hard pairs selected from SFT margin errors and low-confidence cases, and 549 training plus 81 validation hard pairs for the German–Australian experiment.

The six principal DPO/SimPO comparisons are:

1. German oracle DPO;
2. German oracle SimPO;
3. German hard-pair DPO;
4. German hard-pair SimPO;
5. multi-dataset hard-pair DPO;
6. multi-dataset hard-pair SimPO.

None exceeds the frozen SFT checkpoint.

Primary evidence: [`src/training/build_preference.py`](./src/training/build_preference.py), [`src/training/dpo_train.py`](./src/training/dpo_train.py), [`data/processed/german/preference/`](./data/processed/german/preference), and [`outputs/dpo/`](./outputs/dpo).

## Results That Did Not Reach the Target

### DPO and SimPO

| Experiment | ROC-AUC | Cost or decision behavior | Outcome |
|---|---:|---|---|
| German oracle DPO | 0.706 | Cost 325; all low risk | Failed |
| German oracle SimPO | 0.500 | Cost 325; random-level ranking | Failed |
| German hard DPO | 0.478 | Cost 139; ranking below random | Failed |
| German hard SimPO | 0.504 | Cost 135; random-level ranking | Failed |
| Multi-dataset DPO | German 0.517; Australian 0.666 | 100% high-risk decisions | Failed |
| Multi-dataset SimPO | German 0.525; Australian 0.648 | 100% high-risk decisions | Failed |

### Multi-dataset SFT

German–Australian SFT reaches overall ROC-AUC 0.830 and Australian ROC-AUC 0.938, but the frozen German prediction artifact reaches 0.720, below single-dataset SFT's 0.747. More data improves coverage without improving the primary German benchmark.

### Cost-sensitive SFT, Anchored DPO, and Risk-DPO

- cost-sensitive SFT with 5:1 training weights reduces German ROC-AUC to 0.597 and collapses decisions toward high risk;
- anchored DPO does not produce a valid positive result because the pilot exposes an empty validation split;
- the risk-weighted anchored pilot encounters training and device-placement failures;
- the Risk-DPO route is terminated after the underlying label-level preference objective repeatedly fails to preserve ranking.

These runs are retained as engineering and methodological evidence, not presented as successful algorithms.

## Why Preference Optimization Failed in This Setting

Each response is only one of two short strings:

```text
low risk
high risk
```

Within a class, preference pairs therefore reuse the same chosen and rejected texts. DPO/SimPO can reduce their objective by shifting the global probability of the two labels without learning that one applicant should rank above another in risk.

The multi-dataset log-probability audit supports this mechanism:

| Model | Mean log P(low) | Mean log P(high) | High-risk score range |
|---|---:|---:|---|
| SFT | -0.796 | -1.058 | 0.13–0.92 |
| DPO | -22.5 | -19.9 | 0.90–0.95 |
| SimPO | -23.6 | -19.0 | 0.98–0.996 |

After preference optimization, both label log probabilities become extreme and the high-risk score collapses into a narrow interval. The model changes its global label prior but loses useful sample-level separation.

> **Bounded conclusion:** in the tested `Qwen3.5-4B`, small tabular-credit, two-label short-answer setting, label-level DPO/SimPO does not preserve or improve risk ranking. This is not a claim that DPO or SimPO is ineffective for all classification, reasoning, or preference-learning tasks.

Primary evidence: [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) and [`docs/Progress_Report.md`](./docs/Progress_Report.md).

## Repository Guide

```text
risk-control-posttraining/
├── .github/workflows/quality.yml      # CPU-only data, metric, and leakage-regression CI
├── configs/                           # Experiment configuration artifacts
├── convert_german.py                  # Deterministic German converter and validation
├── data/processed/                    # Normalized, SFT, and preference datasets
├── docs/                              # Data contracts, audit, protocols, showcase, and reports
├── outputs/baselines/                 # Baseline predictions
├── outputs/sft/                       # SFT logs, predictions, and adapter metadata
├── outputs/dpo/                       # DPO/SimPO and pilot outputs
├── outputs/c7_final_metrics.json      # Final consolidated evaluation
├── requirements-eval.txt              # Lightweight reproduction dependencies
├── requirements-train.txt             # GPU training/inference dependencies
├── tests/                              # Dataset, metric, showcase, and leakage-regression tests
├── scripts/                           # Slurm and evaluation entry points
├── src/baselines/                     # Majority, Logistic Regression, zero-shot Qwen
├── src/evaluation/                    # Metrics, comparisons, audits, final evaluation
└── src/training/                      # SFT, preference construction, DPO/SimPO, pilots
```

Recommended starting points:

1. [`docs/index.html`](./docs/index.html) — bilingual outcome-oriented showcase;
2. [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) — frozen final metrics;
3. [`docs/Progress_Report.md`](./docs/Progress_Report.md) — complete experiment history and bounded conclusions;
4. [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py) — response-only LoRA SFT;
5. [`src/training/dpo_train.py`](./src/training/dpo_train.py) — DPO/SimPO implementation;
6. [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) — label-prior-collapse audit.

## Quick Start

The published metrics can be reproduced without a GPU or model weights because the exact validation and test prediction artifacts are version controlled.

```bash
git clone <repository-url>
cd risk-control-posttraining
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -r requirements-eval.txt
python -m unittest discover -s tests -v
python -m src.evaluation.c7_final --output /tmp/c7_final_metrics.json
cmp outputs/c7_final_metrics.json /tmp/c7_final_metrics.json
```

For the full GPU workflow, install `requirements-train.txt`, obtain the excluded base model and LoRA adapter weights under their original terms, and provide the local model path explicitly:

```bash
export RISK_CONTROL_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python -m src.baselines.majority
python -m src.baselines.logistic_regression
python -m src.baselines.qwen_zero_shot
sbatch scripts/sft_slurm.sh
sbatch scripts/dpo_train.sh
sbatch scripts/simpo_train.sh
```

`scripts/sft_multi.sh` runs the German–Australian transfer experiment. Training and inference are not expected to reproduce bit-for-bit without the original hardware, model revision, and excluded weight files; the published C7 evaluation is fully artifact-reproducible.

## Interactive Showcase and Deployment

[`docs/index.html`](./docs/index.html) is a static bilingual showcase of the frozen findings. It requires no backend or model weights and exposes the three headline results before the detailed metric explorer. The embedded data are checked against `outputs/c7_final_metrics.json` by `tests/test_showcase_data.py`.

Local validation and GitHub Pages publication instructions are intentionally kept in [`docs/SHOWCASE.md`](./docs/SHOWCASE.md), after the research findings and evidence path rather than at the project entrance.

## Skills Demonstrated

- LLM post-training: response-only SFT, LoRA, DPO, SimPO, and risk-aware objective prototyping;
- data engineering: schema governance, deterministic conversion, ChatML construction, and reproducibility validation;
- evaluation: strong classical baselines, threshold selection, calibration, asymmetric cost, and leakage control;
- experimentation: multi-seed runs, multi-dataset transfer, ablation-style controls, and termination criteria;
- systems engineering: PyTorch, Transformers, PEFT, ModelScope-compatible local models, Slurm, and multi-GPU training;
- research practice: negative-result retention, root-cause analysis, scope-limited conclusions, and evidence-linked reporting.

## Resume-Ready Summary

> Built a reproducible `Qwen3.5-4B` financial-risk post-training pipeline spanning data normalization, ChatML construction, LoRA SFT, DPO/SimPO, and asymmetric-cost evaluation. LoRA SFT improved German Credit ROC-AUC by `+0.232` (`0.515 → 0.747`), approaching Logistic Regression's `0.757`. Ran six principal DPO/SimPO variants across oracle, hard-pair, and multi-dataset settings; none exceeded SFT, and log-probability audits traced the failure to global label-prior shifts rather than improved sample-level ranking.

## Attribution

This project is derived from and informed by the original CALM research:

- Paper: *Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models*;
- Original repository: [Dai-shen/CALM](https://github.com/Dai-shen/CALM);
- original datasets, CALM-7B, and upstream assets remain subject to their respective licenses and terms.

This repository does not claim authorship of the original CALM paper, model, benchmark, or dataset collection. See [`NOTICE`](./NOTICE) for provenance details.

## Disclaimer

This project is for research, education, and portfolio demonstration only. It must not be used to make real credit, insurance, employment, or other high-impact decisions. The reported experiments are not production validation, regulatory approval, or evidence of fairness.
