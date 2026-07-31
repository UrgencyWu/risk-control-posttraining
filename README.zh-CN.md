# 大语言模型风控后训练

<p align="center">
  <a href="./README.md">English</a>
  &nbsp;·&nbsp;
  <a href="./README.zh-CN.md"><strong>简体中文</strong></a>
  &nbsp;·&nbsp;
  <a href="./docs/index.html">交互式展示 / Interactive showcase</a>
</p>

## 研究问题

**在成本敏感的信用风险分类中，后训练能否让一个小型指令模型形成有效的样本级风险排序？相较强传统模型，它的收益边界在哪里？**

项目以 `Qwen3.5-4B` 和 German Credit 为主要实验对象，并使用 Australian Credit 验证跨数据集迁移。实验将表示学习与决策阈值选择分离，重点判断 SFT、DPO、SimPO 是否真正改善样本间风险排序，而不是仅改变 `low risk` / `high risk` 两个标签的整体概率先验。

## 四条量化结论

| 结论 | 冻结结果 | 解释 |
|---|---:|---|
| **LoRA SFT 学到了有效风险排序** | ROC-AUC `0.515 → 0.747`（**+0.232**） | SFT 将近似随机的零样本排序提升到具有竞争力的水平。 |
| **Logistic Regression 仍是诚实的排序基线** | `0.757` vs SFT `0.747` | ROC-AUC 仍相差 `0.010`；项目声称“接近”，不声称稳定超越。 |
| **成本敏感 operating point 具有实际作用** | 最佳 SFT Cost `100`，零样本 `140`，Logistic Regression `113` | 最佳冻结 SFT checkpoint 的非对称测试成本最低，但三随机种子均值为 `123 ± 16`，不能用单次有利结果替代稳定性结论。 |
| **标签级偏好优化出现明确方法边界** | **6 / 6** 组主 DPO/SimPO 实验未超过 SFT | Oracle、hard pair、单数据集和多数据集实验均出现标签先验偏移或决策坍缩，而不是更好的样本级排序。 |

## 项目贡献

1. **可复现的后训练数据链路。** 建立可追溯 RiskDataset Schema、确定性 train/valid/test 划分、ChatML 转换、偏好记录、manifest 与 SHA-256 可复现性检查。
2. **无测试泄漏的成本敏感评测。** 统一比较传统模型与 LLM，覆盖 ROC-AUC、PR-AUC、校准、混淆矩阵和仅由已提交验证预测选择的业务阈值。
3. **SFT 有效性与强基线的诚实对比。** 完成三个 German 随机种子和 German–Australian 迁移实验，证明 LoRA SFT 有效，同时保留 Logistic Regression 更高的排序结果。
4. **DPO/SimPO 的机制性负结论。** 完成六组主偏好优化实验，并通过 log-probability 审计将失败定位为全局标签先验移动，而非条件风险判别改善。

本仓库是开源 [CALM](https://github.com/Dai-shen/CALM) 项目的独立扩展。CALM 提供原始任务背景和公开数据；本项目围绕 `Qwen3.5-4B` 重建实验管线，不主张拥有原始 CALM 论文、模型、基准或数据集的作者身份。

## 证据与实现状态

| 工作流 | 状态 | 已验证结果 |
|---|---|---|
| 数据 Schema 与预处理 | **已实现并验证** | 归一化数据、ChatML 数据、冻结划分、manifest、校验与 SHA-256 可复现性 |
| Majority / Logistic Regression / Qwen 零样本基线 | **已实现并验证** | 在冻结 German 测试集上使用统一成本敏感评测 |
| Qwen3.5-4B LoRA SFT | **已实现并验证** | 三个 German 随机种子，以及 German–Australian 多数据集实验 |
| 验证集选阈值 | **已实现并验证** | 仅以验证预测最小化 `Cost = 5 × FN + 1 × FP`；禁止测试集调 operating point |
| Oracle 与 hard preference 构造 | **已实现并验证** | 单数据集和多数据集偏好记录，元数据冻结 |
| DPO / SimPO | **已运行，未达目标** | 六组主实验均未超过 SFT，并出现标签先验或决策坍缩 |
| Cost-sensitive SFT | **已运行，未达目标** | 训练期成本加权降低排序能力，并使预测向 high risk 偏移 |
| Anchored DPO / Risk-DPO | **未完成或终止** | Pilot 暴露验证集与设备放置问题；不作正向性能声明 |
| C7 最终评测与 log-prob 审计 | **已实现并验证** | 从冻结 valid/test 预测工件 CPU 重建指标、校准、混淆矩阵与机制证据 |

仓库已提交源代码、处理后数据、Shell 入口、预测结果、指标、训练日志、adapter 元数据和研究报告；大模型与 LoRA 权重二进制文件未提交。

## 最终结果

所有首要指标均以 [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) 为准。C7 只从已提交的验证预测确定阈值，再将冻结阈值应用到对应测试预测；不会重新执行模型推理，也不会使用测试标签搜索阈值。完整口径见 [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md)。

### German Credit 测试集，N = 200

| 模型 | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | 高风险召回 | 低风险召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B 零样本 | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B 多数据集 SFT（German 子集） | 0.720 | 0.527 | 0.568 | 0.194 | **0.059** | 142 | 0.769 | 0.504 |

### SFT 稳定性

三个 German SFT 实验的均值约为：

| 指标 | 均值 ± 标准差 |
|---|---:|
| Accuracy | 0.620 ± 0.04 |
| ROC-AUC | 0.747 ± 0.01 |
| 高风险召回 | 0.821 ± 0.08 |
| 验证集阈值下测试 Cost | 123 ± 16 |

冻结的 seed-7 checkpoint 获得 Cost 100，但三随机种子平均 Cost 为 123，高于 Logistic Regression 的 113；SFT 平均 ROC-AUC 也仍低于 Logistic Regression 的 0.757。因此项目证明 SFT 有效，但不声称其在统计意义上稳定优于 Logistic Regression。

## 各模型的实验角色

| 模型或 checkpoint | 实验角色 | 主要结论 |
|---|---|---|
| Majority classifier | Sanity-check 下界 | 借助类别不平衡获得 67.5% Accuracy，但漏掉全部高风险样本；Cost 325 |
| Logistic Regression | 强非 LLM 参考模型 | 最终风险排序与概率质量最佳：ROC-AUC 0.757、NLL 0.552、Cost 113 |
| Qwen3.5-4B 零样本 | 未适配 LLM 基线 | 排序接近随机、严重过预测 high risk、校准较差 |
| Qwen3.5-4B LoRA SFT，seed `10086`、`42`、`7` | 主要后训练实验 | 三次运行均形成有效排序能力，平均 ROC-AUC 0.747 |
| Qwen3.5-4B SFT seed `7` | 冻结下游 checkpoint | 按预定义验证集 PR-AUC 规则选择，而非按测试结果选择；测试 ROC-AUC 0.747、Cost 100 |
| Qwen3.5-4B 多数据集 SFT | 迁移与数据多样性实验 | German–Australian 整体排序较强，但主 German 基准出现负迁移 |
| Qwen3.5-4B DPO / SimPO | 偏好优化压力测试 | Oracle、hard pair 与多数据集变体均未超过 SFT，并经常向单一标签坍缩 |
| Cost-sensitive SFT / Anchored DPO / Risk-DPO Pilot | 非对称加权与 SFT 锚定对照 | 成本加权损害排序；anchored 和 risk-aware Pilot 未形成有效正向结果 |

## 已实现的实验链路

### 1. 数据治理与可复现预处理

```text
Raw data
  → Normalized risk schema
  → SFT ChatML records
  → Preference records
  → Evaluation records
```

已实现内容包括：

- 审计 10 个金融风险数据集，覆盖信用评分、欺诈检测、财务困境和理赔分析；
- 统一 `risk_label`、`task_type`、`target_type` 并保留 `original_label` 追溯；
- 在可用数据中提取受保护属性；
- 构建 description-style 与 table-style prompt；
- 确定性 train/valid/test 划分；
- manifest、Schema 校验和重复 SHA-256 检查。

冻结的 German Credit 管线包含 1,000 条记录和 20 个输入特征，将原标签 `1/2 → 0/1`，并以 seed `10086` 划分为 `700/100/200`。German–Australian 统一数据包含 1,182 条训练、169 条验证和 339 条测试记录。

主要证据：[`convert_german.py`](./convert_german.py)、[`data/processed/`](./data/processed)、[`docs/RiskDataset_Schema.md`](./docs/RiskDataset_Schema.md) 和 [`docs/Progress_Report.md`](./docs/Progress_Report.md)。

### 2. 成本敏感评测

统一评测器覆盖 Accuracy、Balanced Accuracy、Macro-F1、高/低风险召回、ROC-AUC、PR-AUC、NLL、Brier、ECE、混淆矩阵，以及可配置的 FN/FP 成本。

```text
Cost = 5 × false negatives + 1 × false positives
```

False negative 表示真实高风险样本被判断为低风险。5:1 成本比偏向高风险召回，而排序与校准指标用于识别“全部预测 high risk”这类无效策略。

主要证据：[`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md)、[`src/evaluation/metrics.py`](./src/evaluation/metrics.py)、[`src/evaluation/c7_final.py`](./src/evaluation/c7_final.py) 和 [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json)。

### 3. LoRA SFT

最终 SFT 使用 `Qwen3.5-4B`，采用 response-only causal-LM loss、LoRA rank 16、alpha 32、dropout 0.05、attention/MLP projection target、五个 epoch、三个 German 随机种子，并在 2 × RTX PRO 6000 Blackwell GPU 上执行。

SFT 是本项目中唯一持续形成有效风险判别能力的后训练方法。

主要证据：[`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py)、[`src/training/sft_multi.py`](./src/training/sft_multi.py)、[`outputs/sft/`](./outputs/sft) 和 [`scripts/sft_slurm.sh`](./scripts/sft_slurm.sh)。

### 4. 偏好数据与偏好优化

仓库包含 800 对 German oracle preference、361 对由 SFT margin error 与低置信样本构成的 German hard pair，以及 German–Australian 实验的 549 对训练、81 对验证 hard pair。

六组主 DPO/SimPO 对比为：

1. German oracle DPO；
2. German oracle SimPO；
3. German hard-pair DPO；
4. German hard-pair SimPO；
5. 多数据集 hard-pair DPO；
6. 多数据集 hard-pair SimPO。

六组实验均未超过冻结 SFT checkpoint。

主要证据：[`src/training/build_preference.py`](./src/training/build_preference.py)、[`src/training/dpo_train.py`](./src/training/dpo_train.py)、[`data/processed/german/preference/`](./data/processed/german/preference) 和 [`outputs/dpo/`](./outputs/dpo)。

## 已运行但未达到目标的实验

### DPO / SimPO

| 实验 | ROC-AUC | Cost 或决策行为 | 结果 |
|---|---:|---|---|
| German oracle DPO | 0.706 | Cost 325；全部 low risk | 失败 |
| German oracle SimPO | 0.500 | Cost 325；随机水平排序 | 失败 |
| German hard DPO | 0.478 | Cost 139；排序低于随机 | 失败 |
| German hard SimPO | 0.504 | Cost 135；随机水平排序 | 失败 |
| 多数据集 DPO | German 0.517；Australian 0.666 | 100% high-risk 决策 | 失败 |
| 多数据集 SimPO | German 0.525；Australian 0.648 | 100% high-risk 决策 | 失败 |

### 多数据集 SFT

German–Australian SFT 整体 ROC-AUC 为 0.830，Australian ROC-AUC 为 0.938，但冻结 German 预测工件仅为 0.720，低于单数据集 SFT 的 0.747。更多数据扩大了覆盖面，却没有提升主 German 基准。

### Cost-sensitive SFT、Anchored DPO 与 Risk-DPO

- 5:1 训练权重的 cost-sensitive SFT 将 German ROC-AUC 降至 0.597，并使决策向 high risk 坍缩；
- Anchored DPO Pilot 因验证划分为空，没有形成有效正向结果；
- Risk-weighted anchored Pilot 遇到训练和设备放置故障；
- 在基础标签级偏好目标反复破坏排序后，Risk-DPO 路线被终止。

这些实验作为工程和方法证据被保留，但不包装成成功算法。

## 偏好优化为何在当前设定中失败

每个样本的回答只有两个短字符串：

```text
low risk
high risk
```

同一类别的 preference pair 因而复用完全相同的 chosen/rejected 文本。DPO/SimPO 可以通过改变两个标签的全局概率来降低目标，而不必学习“申请人 A 的风险应高于申请人 B”。

多数据集 log-probability 审计支持该机制：

| 模型 | Mean log P(low) | Mean log P(high) | High-risk score 区间 |
|---|---:|---:|---|
| SFT | -0.796 | -1.058 | 0.13–0.92 |
| DPO | -22.5 | -19.9 | 0.90–0.95 |
| SimPO | -23.6 | -19.0 | 0.98–0.996 |

偏好优化后，两个标签的 log-probability 都变得极端，high-risk score 收缩到狭窄区间。模型改变了全局标签先验，却失去了有效样本级分离。

> **边界化结论：** 在当前 `Qwen3.5-4B`、小样本表格信用、二标签短回答设定中，标签级 DPO/SimPO 未能保持或提升风险排序。这不等价于 DPO/SimPO 在所有分类、推理或偏好学习任务中无效。

主要证据：[`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) 和 [`docs/Progress_Report.md`](./docs/Progress_Report.md)。

## 仓库导航

```text
risk-control-posttraining/
├── .github/workflows/quality.yml      # CPU 数据、指标与泄漏回归 CI
├── configs/                           # 实验配置
├── convert_german.py                  # German 确定性转换与校验
├── data/processed/                    # Normalized、SFT 与 preference 数据
├── docs/                              # 数据协议、审计、评测、展示与报告
├── outputs/baselines/                 # 基线预测
├── outputs/sft/                       # SFT 日志、预测与 adapter 元数据
├── outputs/dpo/                       # DPO/SimPO 与 Pilot 产物
├── outputs/c7_final_metrics.json      # 最终统一评测
├── requirements-eval.txt              # 轻量评测依赖
├── requirements-train.txt             # GPU 训练/推理依赖
├── tests/                              # 数据、指标、展示和泄漏回归测试
├── scripts/                           # Slurm 与评测入口
├── src/baselines/                     # Majority、Logistic Regression、零样本 Qwen
├── src/evaluation/                    # 指标、对比、审计与最终评测
└── src/training/                      # SFT、偏好构造、DPO/SimPO 与 Pilot
```

推荐阅读顺序：

1. [`docs/index.html`](./docs/index.html) — 中英双语、成果优先的交互展示；
2. [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) — 冻结最终指标；
3. [`docs/Progress_Report.md`](./docs/Progress_Report.md) — 完整实验历史与边界化结论；
4. [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py) — response-only LoRA SFT；
5. [`src/training/dpo_train.py`](./src/training/dpo_train.py) — DPO / SimPO；
6. [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) — 标签先验坍缩审计。

## 快速复现

已提交的验证/测试预测工件允许在不使用 GPU 或模型权重的情况下重建公开指标：

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

完整 GPU 训练需安装 `requirements-train.txt`，取得原始条款下的基础模型与未提交权重，并显式配置模型路径：

```bash
export RISK_CONTROL_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python -m src.baselines.majority
python -m src.baselines.logistic_regression
python -m src.baselines.qwen_zero_shot
sbatch scripts/sft_slurm.sh
sbatch scripts/dpo_train.sh
sbatch scripts/simpo_train.sh
```

`scripts/sft_multi.sh` 对应 German–Australian 迁移实验。完整训练不承诺在缺少原硬件、模型 revision 与未提交权重时逐比特一致；公开 C7 指标可由数据工件完整重建。

## 交互式展示与部署

[`docs/index.html`](./docs/index.html) 是无需后端和模型权重的中英双语静态展示页。页面先展示三项核心研究发现，再提供冻结指标的交互比较。嵌入数据由 `tests/test_showcase_data.py` 与 `outputs/c7_final_metrics.json` 自动核对。

本地验证与 GitHub Pages 发布方法集中在 [`docs/SHOWCASE.md`](./docs/SHOWCASE.md) 文档后部，不再占据项目叙事入口。

## 技能覆盖

- LLM 后训练：response-only SFT、LoRA、DPO、SimPO 与风险感知目标原型；
- 数据工程：Schema 治理、确定性转换、ChatML 构建与可复现性校验；
- 评测：强传统基线、阈值选择、校准、非对称成本与泄漏控制；
- 实验：多随机种子、多数据集迁移、对照实验与终止准则；
- 系统：PyTorch、Transformers、PEFT、ModelScope 兼容本地模型、Slurm 与多 GPU 训练；
- 研究实践：保留负结果、根因分析、范围受限的结论与证据链接。

## 简历摘要

> 构建可复现的 `Qwen3.5-4B` 金融风控后训练链路，覆盖数据标准化、ChatML、LoRA SFT、DPO/SimPO 和非对称成本评测。LoRA SFT 将 German Credit ROC-AUC 提升 `+0.232`（`0.515 → 0.747`），接近 Logistic Regression 的 `0.757`。完成 oracle、hard pair 与多数据集条件下的六组主 DPO/SimPO 实验，均未超过 SFT；log-probability 审计将失败定位为全局标签先验偏移，而非样本级风险排序改善。

## 署名与免责声明

本项目衍生并受原始 CALM 研究启发；不主张拥有原论文、模型、基准或数据集的作者身份。详见 [`NOTICE`](./NOTICE) 与英文 README 的 Attribution 部分。

本项目仅用于研究、教育与作品展示，不得用于实际信用、保险、就业或其他高影响决策；这些实验不构成生产验证、监管批准或公平性证据。
