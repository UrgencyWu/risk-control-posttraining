# Risk-Control Post-Training for Large Language Models

A portfolio-oriented research and engineering project that studies whether small instruction-tuned language models can perform **cost-sensitive credit-risk classification** after supervised and preference-based post-training.

This repository is an independent extension of the open-source [CALM project](https://github.com/Dai-shen/CALM). The original CALM work provides the financial-risk task context and public datasets. This project focuses on rebuilding the data pipeline, establishing reproducible baselines, and evaluating post-training methods under asymmetric business costs.

> **Current publication status:** the experiment records summarized below were completed in the local development environment. The public `main` branch still contains primarily upstream CALM assets; training scripts, normalized datasets, configurations, and full run artifacts have not yet all been synchronized to GitHub. Results should therefore be treated as documented experimental records rather than independently reproducible public benchmarks at this stage.

## Project Motivation

Credit-risk decisions are asymmetric: predicting a high-risk applicant as low risk is usually more costly than rejecting a low-risk applicant. Standard accuracy alone therefore gives an incomplete view of model utility.

This project asks three practical questions:

1. Can a compact instruction-tuned LLM learn structured credit-risk classification from tabular data expressed as natural-language instructions?
2. Can preference optimization improve high-risk recall or business-weighted cost beyond supervised fine-tuning?
3. When LLM post-training fails to beat a conventional statistical baseline, what parts of the data, objective, and evaluation pipeline explain the gap?

## Scope

The implemented research scope is deliberately narrow:

- Base model: `Qwen2.5-1.5B-Instruct`
- Primary task: binary credit-risk classification
- Primary dataset: German Credit
- Training path: SFT → DPO/SimPO experiments → risk-aware preference objective exploration
- Evaluation: accuracy, high-risk recall, ROC-AUC, confusion matrix, and asymmetric misclassification cost
- Infrastructure: PyTorch, Transformers, ModelScope, vLLM, Ray, Slurm, multi-GPU training

PPO, GRPO, RLVR, and online reinforcement learning are outside the finalized project scope.

## Completed Work

### 1. Data governance and reproducible preprocessing

A normalized schema was designed for heterogeneous financial-risk datasets, covering:

- unified binary labels and task metadata;
- original-label traceability;
- protected attributes such as gender, age group, and foreign-worker status;
- deterministic train/validation/test splitting;
- ChatML-style SFT conversion;
- dataset manifests and reproducibility checks.

For the German Credit dataset, the completed pipeline contains:

- 1,000 records;
- 20 input features: 13 categorical and 7 numerical;
- label mapping from the original `1/2` convention to `0/1`;
- a deterministic `700/100/200` split with seed `10086`;
- normalized and instruction-formatted outputs;
- repeated conversion with identical SHA-256 hashes.

### 2. Baseline system

Three baseline families were evaluated on the frozen German Credit test set.

| Model | Accuracy | High-risk Recall | ROC-AUC | Cost |
|---|---:|---:|---:|---:|
| Majority classifier | 0.6750 | 0.0000 | 0.5000 | 325 |
| Logistic Regression | 0.6750 | 0.8154 | 0.7566 | 113 |
| Qwen zero-shot | 0.3200 | 0.9846 | 0.5152 | 140 |

The logistic-regression baseline is the strongest verified result. The zero-shot LLM detects most high-risk cases but substantially over-predicts risk, producing poor accuracy, near-random ranking quality, and a higher business cost than logistic regression.

### 3. Cost-sensitive evaluation

The evaluation layer was designed around the business asymmetry of credit-risk prediction rather than accuracy alone. It includes:

- high-risk-class recall;
- ROC-AUC;
- false-negative and false-positive accounting;
- configurable asymmetric cost;
- confusion-matrix analysis;
- threshold-sensitive comparison against statistical baselines.

This evaluation design prevents a model from appearing strong merely by predicting the minority risk class aggressively.

### 4. Post-training experimentation

The project established the intended training and analysis path for:

- response-only supervised fine-tuning;
- completion-mask validation;
- preference-pair construction;
- DPO and SimPO hyperparameter exploration;
- risk-aware preference construction based on high-risk errors;
- VaR/CVaR-inspired objective analysis.

The post-training experiments did **not** achieve the original target of reliably outperforming logistic regression on ROC-AUC and asymmetric cost. This negative result is retained as part of the project rather than hidden.

## Main Finding

The central result is not that an LLM solved tabular credit scoring better than conventional machine learning. It did not.

Instead, the project demonstrates a complete applied post-training workflow and identifies a meaningful failure boundary:

- instruction tuning can turn structured tabular records into a trainable language-model task;
- high-risk recall can be increased by aggressive risk prediction;
- improved recall alone does not imply better ranking or lower business cost;
- on a small tabular dataset, logistic regression remains a stronger calibrated baseline;
- preference optimization requires carefully constructed pairs, sufficient data diversity, and explicit calibration objectives to avoid merely shifting the prediction bias.

This is an important practical conclusion for applied LLM engineering: model complexity should be justified against strong non-LLM baselines, especially for low-dimensional structured data.

## Engineering and Research Skills Demonstrated

- End-to-end LLM post-training workflow design
- Deterministic data preprocessing and schema governance
- ChatML instruction construction
- Response-only SFT masking checks
- DPO/SimPO and risk-aware preference objective analysis
- Cost-sensitive machine-learning evaluation
- Classical ML versus LLM baseline comparison
- Multi-GPU experiment orchestration with Slurm, Ray, and vLLM
- Failure analysis and scope control

## Repository Roadmap

The next repository-hardening steps are:

1. synchronize the local preprocessing, baseline, and evaluation source code;
2. publish frozen manifests and non-sensitive experiment configurations;
3. add reproducible commands for data conversion, baseline evaluation, and SFT;
4. publish run summaries and failure-analysis reports;
5. clearly tag completed, partial, and planned experiments.

Until these artifacts are synchronized, this README should be read as a transparent project dossier rather than a claim of fully reproducible open-source release.

## Resume-Ready Project Summary

> Built a cost-sensitive LLM post-training pipeline for credit-risk classification, covering deterministic data normalization, ChatML conversion, SFT and preference-optimization experiments, and asymmetric-cost evaluation. Established majority, logistic-regression, and Qwen baselines; found that logistic regression remained strongest with 0.7566 ROC-AUC and cost 113, while zero-shot Qwen achieved 0.9846 high-risk recall but suffered from over-prediction and near-random ranking. Conducted failure analysis on calibration, preference-pair quality, and the limits of LLMs on small structured datasets.

## Attribution

This project is derived from and informed by the original CALM research:

- Paper: *Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models*
- Original repository: [Dai-shen/CALM](https://github.com/Dai-shen/CALM)
- Original CALM-7B model and datasets remain subject to their respective licenses and terms.

This repository does not claim authorship of the original CALM paper, benchmark, model, or upstream dataset collection.

## Disclaimer

This project is for research, education, and portfolio demonstration only. It must not be used to make real credit, insurance, employment, or other high-impact decisions. The reported experiments are not production validation, regulatory approval, or evidence of fairness.
