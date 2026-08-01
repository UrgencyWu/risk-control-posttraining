# 大语言模型风控后训练

<p align="center">
  <a href="./README.md"><strong>中文</strong></a>
  &nbsp;·&nbsp;
  <a href="./README.en.md">English</a>
  &nbsp;·&nbsp;
  <a href="https://urgencywu.github.io/risk-control-posttraining/"><strong>在线交互展示</strong></a>
</p>

## 项目摘要

本项目研究：**在小样本、结构化信用风险任务中，大语言模型后训练能否形成有效的样本级风险排序，并在强传统模型基线面前体现实际价值。**

项目以 `Qwen3.5-4B`、German Credit 为主实验对象，使用 Australian Credit 进行多数据集验证，覆盖数据治理、LoRA SFT、DPO/SimPO、成本敏感决策、无测试泄漏评测与失败机制审计。

| 核心发现 | 冻结结果 | 结论边界 |
|---|---:|---|
| LoRA SFT 学到有效风险排序 | ROC-AUC `0.515 → 0.747`，提升 **+0.232** | 显著优于零样本模型 |
| 强传统模型仍略优 | Logistic Regression `0.757` vs SFT `0.747` | 声称“接近”，不声称稳定超越 |
| 成本敏感阈值具有实际作用 | 最佳 SFT Cost `100`，LR `113`，零样本 `140` | 三随机种子 SFT 平均 Cost 为 `123 ± 16` |
| 标签级偏好优化出现方法边界 | **6 / 6** 组主 DPO/SimPO 实验未超过 SFT | 限定于当前小样本、表格数据、短标签设置 |

本仓库是开源 [CALM](https://github.com/Dai-shen/CALM) 项目的独立扩展，不主张拥有原始 CALM 论文、模型、基准或数据集的作者身份。

## 研究问题

1. 指令模型能否从自然语言形式的表格信用记录中学习可泛化的风险排序？
2. LoRA SFT 能否同时改善 ROC-AUC、概率质量和非对称业务成本？
3. DPO/SimPO 是否能够在不破坏样本级排序的前提下进一步优化风险偏好？
4. 当后训练失败时，根因来自数据、优化、阈值，还是目标函数与任务形式的错配？

## 技术路线

```text
CALM 数据与仓库审计
        ↓
统一 RiskDataset Schema 与标签语义
        ↓
确定性划分 + Normalized / ChatML 数据
        ↓
Majority / Logistic Regression / Qwen Zero-shot 基线
        ↓
Qwen3.5-4B LoRA SFT（三随机种子）
        ↓
Oracle / Hard Preference 构造
        ↓
DPO / SimPO / 成本敏感对照
        ↓
验证集冻结阈值 + C7 最终评测 + Log-probability 审计
```

模型训练负责形成风险分数，业务阈值只在验证集上按照 `Cost = 5 × FN + 1 × FP` 选择，再固定应用到测试集。

## 实施细节

### 1. 数据与可复现性

- German Credit：1,000 条记录、20 个特征，按 seed `10086` 冻结为 `700/100/200`；
- German–Australian：`1182/169/339` 条 train/valid/test 记录；
- 输出 Normalized、ChatML SFT、Preference 与 Evaluation 多层数据；
- 使用 manifest、Schema、V1–V7 与重复 SHA-256 检查保证确定性。

### 2. 模型与训练角色

| 模型或 Checkpoint | 实验角色 | 状态 |
|---|---|---|
| Majority | 类别不平衡下界 | 已实现并验证 |
| Logistic Regression | 强非 LLM 基线 | 已实现并验证 |
| Qwen3.5-4B Zero-shot | 未适配 LLM 基线 | 已实现并验证 |
| Qwen3.5-4B LoRA SFT，seeds `10086/42/7` | 主后训练实验 | 已实现并验证 |
| German–Australian Multi-SFT | 多数据集迁移 | German 上存在负迁移 |
| DPO / SimPO | 标签级偏好优化压力测试 | 已运行，未达目标 |
| Cost-sensitive / Anchored / Risk-DPO | 成本加权与锚定对照 | 失败、未完成或终止 |

SFT 使用 response-only causal-LM loss，LoRA `r=16`、`alpha=32`、`dropout=0.05`，训练 5 个 epoch，并在 `2 × RTX PRO 6000 Blackwell` 上完成多随机种子实验。

### 3. 偏好数据与评测

- German oracle preference：800 对；German hard preference：361 对；
- German–Australian hard preference：549 条训练、81 条验证；
- 统一评测包含 Accuracy、Balanced Accuracy、Macro-F1、Recall、ROC-AUC、PR-AUC、NLL、Brier、ECE、混淆矩阵与非对称 Cost；
- 测试标签不参与阈值搜索。完整协议见 [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md)。

## 实验数据

### 多指标分组柱状图

![German Credit 多指标分组柱状图](./docs/assets/metric_grouped_bars.svg)

柱状图统一比较 `ROC-AUC`、`PR-AUC`、`Brier`、`ECE`、高风险召回和低风险召回六个 `0–1` 尺度指标；每个指标组包含五种算法。`NLL` 与 `Cost` 的精确值保留在下表中。

### German Credit 测试集，N = 200

| 模型 | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | 高风险召回 | 低风险召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B Zero-shot | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B Multi-SFT（German 子集） | 0.720 | 0.527 | 0.568 | 0.194 | **0.059** | 142 | 0.769 | 0.504 |

### SFT 三随机种子稳定性

| 指标 | 均值 ± 标准差 |
|---|---:|
| Accuracy | `0.620 ± 0.04` |
| ROC-AUC | `0.747 ± 0.01` |
| 高风险召回 | `0.821 ± 0.08` |
| 验证集阈值下测试 Cost | `123 ± 16` |

### 六组主 DPO/SimPO 实验

| 实验 | ROC-AUC | 决策行为或 Cost | 结果 |
|---|---:|---|---|
| German Oracle DPO | 0.706 | Cost 325；全部预测 low risk | 未超过 SFT |
| German Oracle SimPO | 0.500 | Cost 325；随机水平 | 未超过 SFT |
| German Hard DPO | 0.478 | Cost 139；排序低于随机 | 未超过 SFT |
| German Hard SimPO | 0.504 | Cost 135；随机水平 | 未超过 SFT |
| Multi-dataset DPO | German 0.517；Australian 0.666 | 100% high-risk 决策 | 未超过 SFT |
| Multi-dataset SimPO | German 0.525；Australian 0.648 | 100% high-risk 决策 | 未超过 SFT |

最终指标以 [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) 为权威来源。

## 实验结论

1. **SFT 是唯一稳定产生有效风险排序的后训练方法。** ROC-AUC 从 `0.515` 提升到 `0.747`。
2. **Logistic Regression 仍是排序和概率质量最强的基线。** `0.757` 高于 SFT 的 `0.747`，不宣称稳定超越。
3. **成本敏感阈值有效，但不能替代排序能力。** AUC 与校准指标用于识别“全部预测 high risk”的伪改进。
4. **当前标签级 DPO/SimPO 与样本级风险排序存在目标错配。** 偏好损失主要移动全局标签先验。
5. **结论严格限定于当前设置。** 不推广为 DPO/SimPO 对所有任务均无效。

## 复现指引

### 1. CPU 复现冻结指标

```bash
git clone https://github.com/UrgencyWu/risk-control-posttraining.git
cd risk-control-posttraining
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v
python -m src.evaluation.c7_final --output /tmp/c7_final_metrics.json
cmp outputs/c7_final_metrics.json /tmp/c7_final_metrics.json
```

### 2. GPU 训练与推理

```bash
python -m pip install -r requirements-train.txt
export RISK_CONTROL_MODEL_ID=/absolute/path/to/Qwen3.5-4B
sbatch scripts/sft_slurm.sh
sbatch scripts/dpo_train.sh
sbatch scripts/simpo_train.sh
```

- 在线交互页面：<https://urgencywu.github.io/risk-control-posttraining/>
- 展示页说明：[`docs/SHOWCASE.md`](./docs/SHOWCASE.md)

---

本项目仅用于研究、教育与作品展示，不得用于实际高影响决策。来源与许可证说明见 [`NOTICE`](./NOTICE) 和 [`LICENSE`](./LICENSE)。
