# 大语言模型风控后训练

<p align="center">
  <a href="./README.md">English</a>
  &nbsp;·&nbsp;
  <a href="./README.zh-CN.md"><strong>简体中文</strong></a>
  &nbsp;·&nbsp;
  <a href="./docs/index.html">交互式展示 / Interactive showcase</a>
</p>

> 这是一个端到端的应用型研究项目：使用大语言模型进行成本敏感的信用风险二分类，覆盖数据治理、传统与 LLM 基线、LoRA SFT、偏好优化、决策阈值选择，以及失败机制分析。

本仓库是开源 [CALM](https://github.com/Dai-shen/CALM) 项目的独立扩展。CALM 提供原始金融风控任务背景和公开数据；本项目围绕更新的 Qwen 模型重建实验管线，并检验后训练能否超越强传统基线。

| 项目定位 | 已验证结论 |
|---|---|
| 最终实验模型 | `Qwen3.5-4B` |
| 主任务 | 二元信用风险分类 |
| 主基准 | German Credit；Australian Credit 用于多数据集验证 |
| 核心结果 | LoRA SFT 显著优于零样本 LLM，并接近 Logistic Regression；当前二标签短回答设定下，DPO/SimPO 未提升风险排序，且反复出现类别先验坍缩。 |

## 交互式展示

[`docs/index.html`](./docs/index.html) 是一个无需后端的中英双语展示页，可在 GitHub Pages 中直接运行。它支持：

- 中英文一键切换；
- 在 ROC-AUC、PR-AUC 与业务 Cost 之间切换柱状图；
- 选择任一模型查看其冻结阈值、评估指标和实验角色；
- 以数据工件为证据展示成功路径与负结果边界。

部署方法见 [`docs/SHOWCASE.md`](./docs/SHOWCASE.md)。展示数据由 `outputs/c7_final_metrics.json` 提供，并由自动化测试校验，避免展示文案与已提交指标漂移。

## 已完成的工作

| 工作流 | 状态 | 已验证结果 |
|---|---|---|
| 数据 Schema 与预处理 | **已实现并验证** | 归一化数据、ChatML 数据、冻结划分、manifest、校验检查与 SHA-256 可复现性 |
| Majority / Logistic Regression / Qwen 零样本基线 | **已实现并验证** | 在冻结 German 测试集上使用统一成本敏感评测 |
| Qwen3.5-4B LoRA SFT | **已实现并验证** | 三个 German 随机种子，以及 German–Australian 多数据集实验 |
| 验证集选阈值 | **已实现并验证** | 仅以验证集最小化 `Cost = 5 × FN + 1 × FP`；禁止测试集调阈值 |
| Oracle 与 hard preference 构造 | **已实现并验证** | 单数据集和多数据集偏好数据，且元数据冻结 |
| DPO / SimPO | **已运行，未达目标** | 六组主实验均未超过 SFT，并出现标签先验或决策坍缩 |
| Cost-sensitive SFT | **已运行，未达目标** | 加权训练降低排序能力，并使预测向 high risk 坍缩 |
| Anchored DPO / Risk-DPO | **未完成或终止** | Pilot 暴露了验证集与设备放置问题；不作正向性能声明 |
| C7 最终评测与 log-prob 审计 | **已实现并验证** | 从冻结 valid/test 预测工件 CPU 复现指标、校准与混淆矩阵 |

## 研究问题

1. 指令微调后的 LLM 能否从自然语言形式的表格信用记录中学习风险判别？
2. SFT 或偏好优化能否在不破坏样本级排序的前提下，提升高风险召回并降低非对称业务成本？
3. 当后训练方法失败时，原因是数据构造、优化不稳定、校准问题，还是目标函数与任务的错配？

## 最终结果

所有下列指标均以 [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) 为准。C7 仅在验证预测上选择阈值，再将冻结阈值应用于对应测试预测；它不会重新推理，也不会用测试标签搜索 operating point。完整口径见 [`docs/EVALUATION_PROTOCOL.md`](./docs/EVALUATION_PROTOCOL.md)。

### German Credit 测试集，N = 200

| 模型 | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | 高风险召回 | 低风险召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.500 | 0.325 | 8.980 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen3.5-4B 零样本 | 0.515 | 0.348 | 1.720 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | 0.076 | 113 | 0.815 | 0.607 |
| Qwen3.5-4B SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | **100** | 0.939 | 0.407 |
| Qwen3.5-4B 多数据集 SFT（German 子集） | 0.720 | 0.527 | 0.568 | 0.194 | **0.059** | 142 | 0.769 | 0.504 |

### SFT 稳定性

三个 German SFT 实验的均值约为：ROC-AUC `0.747 ± 0.01`、高风险召回 `0.821 ± 0.08`、验证集阈值下的测试 Cost `123 ± 16`。因此本项目证明 SFT 有效，但不声称其稳定或显著优于 Logistic Regression。

### 得到的正向结论

相对 Qwen3.5-4B 零样本，冻结的 SFT checkpoint 将：

- ROC-AUC 从 `0.515` 提升至 `0.747`；
- PR-AUC 从 `0.348` 提升至 `0.555`；
- NLL 从 `1.720` 降至 `0.556`；
- 验证集选阈值下的最终非对称 Cost 从 `140` 降至 `100`。

### 被保留的负结果

六组 DPO/SimPO 变体（oracle、hard pair、单数据集、多数据集）均未超过冻结 SFT。审计显示，DPO/SimPO 可以通过整体移动 `low` / `high` 标签先验来降低偏好目标，而不是学习样本间的风险排序：DPO 与 SimPO 的标签 log-probabilities 变得极端，`p(high)` 区间随之坍缩。

结论受到严格边界限制：在当前 `Qwen3.5-4B`、小样本表格信用、二标签短回答设定中，标签级 DPO/SimPO 未能保持或提升风险排序；这并不等价于 DPO/SimPO 在所有分类、推理或偏好任务中无效。

## 可复现性

已提交的 valid/test 预测工件允许不依赖 GPU 或模型权重重现公开指标：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -r requirements-eval.txt
python -m unittest discover -s tests -v
python -m src.evaluation.c7_final --output /tmp/c7_final_metrics.json
cmp outputs/c7_final_metrics.json /tmp/c7_final_metrics.json
```

完整 GPU 训练需安装 `requirements-train.txt`，取得原始条款下的 Qwen 模型与未提交权重，并显式设置模型路径：

```bash
export RISK_CONTROL_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python -m src.baselines.majority
python -m src.baselines.logistic_regression
python -m src.baselines.qwen_zero_shot
sbatch scripts/sft_slurm.sh
sbatch scripts/dpo_train.sh
sbatch scripts/simpo_train.sh
```

`scripts/sft_multi.sh` 对应 German–Australian 迁移实验。C7 指标可完全由工件复现；完整训练并不承诺在缺少原始硬件、模型版本和未提交权重时实现逐比特一致。

## 推荐阅读路径

1. [`docs/index.html`](./docs/index.html) — 中英双语交互展示；
2. [`docs/Progress_Report.md`](./docs/Progress_Report.md) — 完整实验历史与冻结结论；
3. [`outputs/c7_final_metrics.json`](./outputs/c7_final_metrics.json) — 最终统一指标；
4. [`src/training/sft_lora_manual.py`](./src/training/sft_lora_manual.py) — response-only LoRA SFT；
5. [`src/training/dpo_train.py`](./src/training/dpo_train.py) — DPO / SimPO；
6. [`src/evaluation/c5v3_audit.py`](./src/evaluation/c5v3_audit.py) — 标签先验坍缩审计。

## 技能覆盖

- LLM 后训练：response-only SFT、LoRA、DPO、SimPO 与风险感知目标原型；
- 数据工程：Schema 治理、确定性转换、ChatML 构建与可复现性校验；
- 评测：强传统基线、阈值选择、校准、非对称成本与泄漏控制；
- 实验：多随机种子、多数据集迁移、对照实验和终止准则；
- 系统：PyTorch、Transformers、PEFT、Slurm 和多 GPU 训练；
- 研究实践：保留负结果、根因分析、范围受限的结论与证据链接。

## 署名与免责声明

本项目衍生并受原始 CALM 研究启发；不主张拥有原论文、模型、基准或数据集的作者身份。详见 [`NOTICE`](./NOTICE) 与英文 README 的 Attribution 部分。

本项目仅用于研究、教育与作品展示，不得用于实际信用、保险、就业或其他高影响决策；这些实验不构成生产验证、监管批准或公平性证据。
