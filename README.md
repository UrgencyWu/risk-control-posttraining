# Risk-Control Post-Training for Large Language Models

An end-to-end applied research project on **cost-sensitive credit-risk classification with large language models**, covering deterministic data construction, classical and LLM baselines, LoRA supervised fine-tuning, preference optimization, decision-threshold selection, and failure-mechanism analysis.

This repository is an independent extension of the open-source [CALM project](https://github.com/Dai-shen/CALM). CALM supplies the original financial-risk task context and public datasets; this project rebuilds the experimental pipeline around a newer Qwen model and evaluates whether post-training adds value over a strong conventional baseline.

> **Final experimental model:** `Qwen3.5-4B`  
> **Primary task:** binary credit-risk classification  
> **Primary benchmark:** German Credit; Australian Credit is used for multi-dataset validation  
> **Core result:** LoRA SFT substantially improves the zero-shot LLM and approaches Logistic Regression, while label-level DPO/SimPO does not improve risk ranking and repeatedly causes class-prior collapse.

## Project Status

| Workstream | Status | Verified outcome |
|---|---|---|
| Data schema and preprocessing | **Implemented and verified** | Deterministic normalized and ChatML datasets, frozen splits, manifests, validation checks, and SHA-256 reproducibility |
| Majority / Logistic Regression / Qwen zero-shot baselines | **Implemented and verified** | Unified cost-sensitive evaluation on the frozen German test set |
| Qwen3.5-4B LoRA SFT | **Implemented and verified** | Three German seeds plus a German-Australian multi-dataset run |
| Validation-selected decision threshold | **Implemented and verified** | Thresholds selected on validation data under `Cost = 5 × FN + 1 × FP`; no test-set threshold tuning |
| Oracle and hard preference construction | **Implemented and verified** | Single- and multi-dataset preference datasets with frozen metadata |
| DPO and SimPO | **Run; target not achieved** | Six principal variants failed to exceed SFT and showed label-prior or decision collapse |
| Cost-sensitive SFT | **Run; target not achieved** | Cost weighting degraded ranking and collapsed predictions toward high risk |
| Anchored DPO / Risk-DPO | **Incomplete or terminated** | Pilot exposed validation-split and device-placement failures; no positive performance claim is made |
| Final C7 evaluation and log-probability audit | **Implemented and verified** | Consolidated metrics, calibration measures, confusion matrices, and mechanism evidence are public |

The repository contains source code, processed datasets, shell entry points, predictions, metrics, training logs, adapter metadata, and research reports. Large model and LoRA weight binaries are intentionally excluded.

## Research Questions

1. Can an instruction-tuned LLM learn risk discrimination from tabular credit records represented as natural-language instructions?
2. Can SFT or preference optimization improve high-risk recall and asymmetric business cost without destroying sample-level ranking?
3. When a post-training method fails, is the failure caused by data construction, optimization instability, calibration, or a mismatch between the learning objective and the task?

## Model Roles

The project uses different models and checkpoints for distinct experimental purposes. They should not be treated as interchangeable results.

| Model or checkpoint | Experimental role | Main finding |
|---|---|---|
| Majority classifier | Sanity-check lower bound | Obtains 67.5% accuracy from class imbalance but misses every high-risk case; cost 325 |
| Logistic Regression | Strong non-LLM reference | Best final risk ranking and probability quality: ROC-AUC 0.757, NLL 0.552, cost 113 |
| Qwen3.5-4B zero-shot | Unadapted LLM baseline | Near-random ranking, severe high-risk over-prediction, and poor calibration |
| Qwen3.5-4B LoRA SFT, seeds `10086`, `42`, `7` | Primary post-training experiment | Consistently creates useful risk-ranking ability; mean ROC-AUC 0.747, but does not establish stable superiority over Logistic Regression |
| Qwen3.5-4B SFT seed `7` | Frozen downstream SFT checkpoint | Selected by the predeclared **validation PR-AUC** rule, not by test performance; final test ROC-AUC 0.747 and cost 100 |
| Qwen3.5-4B multi-dataset SFT | Transfer and data-diversity experiment | Strong combined German-Australian ranking, but negative transfer on German relative to single-dataset SFT |
| Qwen3.5-4B DPO / SimPO | Preference-optimization stress test | Oracle, hard-pair, and multi-dataset variants all underperform SFT and often collapse toward one label |
| Cost-sensitive SFT / Anchored DPO / Risk-DPO pilots | Controls for asymmetric weighting and SFT anchoring | Cost weighting degraded ranking; anchored and risk-aware pilots did not yield a valid successful result |

## Implemented and Verified Work

### 1. Data governance and reproducible preprocessing

The project defines a multi-layer data protocol:

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
- dataset manifests, schema validation, and repeated SHA-256 checks.

For German Credit, the frozen pipeline contains:

- 1,000 records and 20 input features: 13 categorical and 7 numerical;
- original label mapping `1/2 → 0/1`;
- deterministic `700/100/200` train/validation/test split with seed `10086`;
- normalized and ChatML SFT outputs;
- V1–V7 validation checks;
- identical output hashes across repeated conversion runs.

For multi-dataset validation, German and Australian Credit are unified into:

- 1,182 training records;
- 169 validation records;
- 339 test records.

Primary evidence: [`convert_german.py`](./convert_german.py), [`data/processed/`](./data/processed), [`docs/RiskDataset_Schema.md`](./docs/RiskDataset_Schema.md), and [`docs/Progress_Report.md`](./docs/Progress_Report.md).

### 2. Baselines and cost-sensitive evaluation

The unified evaluator implements:

- Accuracy, Balanced Accuracy, and Macro-F1;
- high-risk and low-risk recall;
- ROC-AUC and PR-AUC;
- NLL, Brier score, and expected calibration error;
- confusion matrices;
- configurable false-negative and false-positive cost;
- validation-only threshold selection.

The business objective is:

```text
Cost = 5 × false negatives + 1 × false positives
```

A false negative means a genuinely high-risk applicant is classified as low risk. The 5:1 cost ratio intentionally favors high-risk recall, but ranking and calibration metrics remain necessary to detect trivial all-high-risk behavior.

Primary evidence: [`src/evaluation/metrics.py`](./src/evaluation/metrics.py), [`src/evaluation/c7_final.py`](./src/evaluation/c7_final.py), and [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json).

### 3. LoRA supervised fine-tuning

The final SFT experiments use `Qwen3.5-4B` with:

- response-only causal-LM loss;
- LoRA rank 16, alpha 32, dropout 0.05;
- attention and MLP projection targets;
- five training epochs;
- three random seeds on German Credit;
- multi-GPU execution on 2 × RTX PRO 6000 Blackwell GPUs;
- validation-based checkpoint and threshold decisions.

SFT is the only post-training method in this project that consistently creates useful risk discrimination.

Primary evidence: [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py), [`src/training/sft_multi.py`](./src/training/sft_multi.py), [`outputs/sft/`](./outputs/sft), and [`scripts/sft_slurm.sh`](./scripts/sft_slurm.sh).

### 4. Preference construction and optimization experiments

The repository includes:

- 800 oracle preference pairs for German Credit;
- 361 hard preference pairs selected from SFT margin errors and low-confidence cases;
- 549 training and 81 validation hard pairs for the German-Australian experiment;
- DPO, SimPO, anchored DPO, risk-weighted DPO, and cost-sensitive SFT implementations or pilots;
- prediction outputs and training logs for completed runs.

The principal DPO/SimPO comparison comprises six completed variants:

1. German oracle DPO;
2. German oracle SimPO;
3. German hard-pair DPO;
4. German hard-pair SimPO;
5. multi-dataset hard-pair DPO;
6. multi-dataset hard-pair SimPO.

None exceeds the frozen SFT checkpoint.

Primary evidence: [`src/training/build_preference.py`](./src/training/build_preference.py), [`src/training/dpo_train.py`](./src/training/dpo_train.py), [`data/processed/german/preference/`](./data/processed/german/preference), and [`outputs/dpo/`](./outputs/dpo).

## Final Results

All headline metrics below use the consolidated C7 artifact, [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json), as the source of truth. Earlier stage reports may contain slightly different zero-shot costs because they were produced before final evaluator consolidation.

### German Credit test set, N = 200

| Model | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | High-risk recall | Low-risk recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B zero-shot | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 135 | 1.000 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B multi-SFT, German subset | 0.721 | 0.533 | 0.567 | 0.194 | **0.060** | 104 | 0.954 | 0.341 |

### SFT stability across seeds

The three German SFT runs produce approximately:

| Metric | Mean ± standard deviation |
|---|---:|
| Accuracy | 0.620 ± 0.04 |
| ROC-AUC | 0.747 ± 0.01 |
| High-risk recall | 0.821 ± 0.08 |
| Validation-selected test cost | 123 ± 16 |

This distinction matters:

- the best frozen SFT checkpoint obtains cost 100;
- the three-seed mean cost is 123, above Logistic Regression's 113;
- mean SFT ROC-AUC remains slightly below Logistic Regression's 0.757.

The project therefore demonstrates that SFT is effective, but it does **not** claim robust or statistically established superiority over Logistic Regression.

## What Succeeded

### LoRA SFT

Relative to Qwen3.5-4B zero-shot, the frozen SFT checkpoint improves:

- ROC-AUC from 0.515 to 0.747;
- PR-AUC from 0.348 to 0.555;
- NLL from 1.720 to 0.556;
- final asymmetric cost from 135 to 100 under a validation-selected threshold.

This verifies that a compact LLM can learn meaningful task discrimination from structured credit records after supervised adaptation.

### Cost-sensitive decision optimization

Validation-selected thresholds materially reduce asymmetric cost. For SFT, the selected thresholds lie near the theoretical Bayes threshold for a 5:1 false-negative/false-positive cost ratio:

```text
t* = 1 / (1 + 5) ≈ 0.167
```

This separates representation learning from downstream operating-point selection and avoids using the test set to tune business cost.

### Reproducible experimental governance

The project preserves data provenance, deterministic splits, multi-seed reporting, strong non-LLM baselines, frozen evaluation rules, and negative results. These controls are central to the project outcome, not auxiliary documentation.

## What Ran but Did Not Reach the Target

### Stable superiority over Logistic Regression

The original goal was to reliably exceed Logistic Regression on both risk ranking and asymmetric cost. That target was not achieved:

- Logistic Regression retains the best ROC-AUC and NLL;
- the best SFT seed obtains lower cost, but SFT's three-seed mean cost is higher;
- the available test set is too small to claim statistical superiority from a single favorable seed.

### DPO and SimPO

Across oracle, hard-pair, single-dataset, and multi-dataset settings, DPO and SimPO fail to improve the frozen SFT checkpoint.

Representative outcomes include:

| Experiment | ROC-AUC | Cost or decision behavior | Outcome |
|---|---:|---|---|
| German oracle DPO | 0.706 | Cost 325; all low risk | Failed |
| German oracle SimPO | 0.500 | Cost 325; random-level ranking | Failed |
| German hard DPO | 0.478 | Cost 139; ranking below random | Failed |
| German hard SimPO | 0.504 | Cost 135; random-level ranking | Failed |
| Multi-dataset DPO | German 0.517; Australian 0.666 | 100% high-risk decisions | Failed |
| Multi-dataset SimPO | German 0.525; Australian 0.648 | 100% high-risk decisions | Failed |

### Multi-dataset SFT

German-Australian SFT reaches overall ROC-AUC 0.830 and Australian ROC-AUC 0.938, showing that the shared pipeline can learn across datasets. However, German ROC-AUC falls from 0.747 to 0.721, so the additional data does not improve the primary German benchmark.

### Cost-sensitive SFT, Anchored DPO, and Risk-DPO

- cost-sensitive SFT with 5:1 weighting reduces German ROC-AUC to 0.597 and collapses decisions toward high risk;
- anchored DPO does not produce a valid positive result because the pilot exposed an empty validation split;
- the risk-weighted anchored pilot encounters training and device-placement failures;
- the Risk-DPO route is terminated after the underlying label-level preference objective repeatedly fails to preserve risk ranking.

These runs are retained as engineering and methodological evidence, but they are not presented as successful algorithms.

## Why Preference Optimization Failed in This Setting

The central hypothesis is an objective-task mismatch.

Each sample's response is only one of two short strings:

```text
low risk
high risk
```

Within a class, preference pairs therefore reuse the same chosen and rejected texts. DPO/SimPO can reduce their objective primarily by shifting the global probability of the two labels, without learning that one applicant should rank above another applicant in risk.

The multi-dataset log-probability audit supports this mechanism:

| Model | Mean log P(low) | Mean log P(high) | High-risk score range |
|---|---:|---:|---|
| SFT | -0.796 | -1.058 | 0.13–0.92 |
| DPO | -22.5 | -19.9 | 0.90–0.95 |
| SimPO | -23.6 | -19.0 | 0.98–0.996 |

After preference optimization, both label log probabilities become extreme and the high-risk score collapses into a narrow interval. The model changes its global label prior but loses useful sample-level separation.

This conclusion is deliberately bounded:

> In the tested `Qwen3.5-4B`, small tabular-credit, two-label short-answer setting, label-level DPO/SimPO does not preserve or improve risk ranking. The project does not claim that DPO or SimPO is ineffective for all classification, reasoning, or preference-learning tasks.

Primary evidence: [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) and [`docs/Progress_Report.md`](./docs/Progress_Report.md).

## Repository Guide

```text
risk-control-posttraining/
├── convert_german.py                  # Deterministic German converter and validation
├── configs/                           # Experiment configuration artifacts
├── data/processed/                    # Normalized, SFT, and preference datasets
├── docs/                              # Data contracts, audit, plans, and progress report
├── outputs/baselines/                 # Baseline predictions
├── outputs/sft/                       # SFT logs, predictions, and adapter metadata
├── outputs/dpo/                       # DPO/SimPO and pilot outputs
├── outputs/c7_final_metrics.json      # Final consolidated evaluation
├── reports/baseline_report.md         # Stage-C2 baseline report
├── scripts/                           # Slurm and evaluation entry points
├── src/baselines/                     # Majority, Logistic Regression, zero-shot Qwen
├── src/evaluation/                    # Metrics, comparisons, audits, final evaluation
└── src/training/                      # SFT, preference construction, DPO/SimPO, pilots
```

Recommended starting points:

1. [`docs/Progress_Report.md`](./docs/Progress_Report.md) — full experimental history and frozen conclusions;
2. [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) — final metrics;
3. [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py) — response-only LoRA SFT;
4. [`src/training/dpo_train.py`](./src/training/dpo_train.py) — DPO/SimPO implementation;
5. [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) — label-prior collapse audit.

## Reproduction Notes

- The final reported model is `Qwen3.5-4B`.
- Training scripts currently reference an environment-specific local model path such as `/data/share/model/Qwen3.5-4B`; adapt this path before rerunning.
- The repository excludes large `adapter_model.safetensors` and tokenizer binaries, but retains training logs, adapter configurations, predictions, processed data, and evaluation outputs.
- Some early planning artifacts still mention `Qwen2.5-1.5B`; they are historical remnants and are **not** the source of the final reported metrics.
- Headline results should be read from `outputs/c7_final_metrics.json` and the final sections of `docs/Progress_Report.md`.

## Skills Demonstrated

- LLM post-training: response-only SFT, LoRA, DPO, SimPO, and risk-aware objective prototyping;
- data engineering: schema governance, deterministic conversion, ChatML construction, and reproducibility validation;
- evaluation: strong classical baselines, threshold selection, calibration, asymmetric cost, and leakage control;
- experimentation: multi-seed runs, multi-dataset transfer, ablation-style controls, and termination criteria;
- systems engineering: PyTorch, Transformers, PEFT, ModelScope-compatible local models, Slurm, and multi-GPU training;
- research practice: negative-result retention, root-cause analysis, scope-limited conclusions, and evidence-linked reporting.

## Resume-Ready Summary

> Built a reproducible `Qwen3.5-4B` financial-risk post-training pipeline spanning data normalization, ChatML construction, LoRA SFT, DPO/SimPO, and asymmetric-cost evaluation. LoRA SFT improved German Credit ROC-AUC from 0.515 to 0.747 and achieved test cost 100 using a validation-selected threshold, approaching Logistic Regression's 0.757 ROC-AUC. Ran six DPO/SimPO variants across oracle, hard-pair, and multi-dataset settings; none exceeded SFT, and log-probability audits traced the failure to global label-prior shifts rather than improved sample-level ranking.

## Attribution

This project is derived from and informed by the original CALM research:

- Paper: *Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models*;
- Original repository: [Dai-shen/CALM](https://github.com/Dai-shen/CALM);
- original datasets, CALM-7B, and upstream assets remain subject to their respective licenses and terms.

This repository does not claim authorship of the original CALM paper, model, benchmark, or dataset collection.

## Disclaimer

This project is for research, education, and portfolio demonstration only. It must not be used to make real credit, insurance, employment, or other high-impact decisions. The reported experiments are not production validation, regulatory approval, or evidence of fairness.