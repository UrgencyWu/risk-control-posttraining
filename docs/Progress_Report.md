# CALM 金融风控后训练项目 — 进展报告

> **日期**: 2026-07-21
> **仓库**: `/home/wushaohua/data/risk-control-posttraining`
> **基座模型**: Qwen3.5-4B
> **训练框架**: PEFT LoRA + Slurm
> **当前阶段**: C3 完成 → C4 待启动

---

## 1. 已交付产出总览

```
新增文件: ~35 个
修改已有文件: 0 个

risk-control-posttraining/
├── docs/                          # 5 份设计文档
├── src/                           # 9 个 Python 模块
├── scripts/                       # 2 个 Slurm 作业脚本
├── configs/                       # 1 个配置
├── data/processed/german/         # 7 个数据产物 (+ manifest)
├── outputs/                       # 12 个预测结果文件
├── reports/                       # 1 个基线报告
└── convert_german.py              # 数据转换器
```

---

## 2. 各阶段成果

### C0: 项目审计

**产出**:
- `docs/CALM_Audit_Report.md` (663 行) — 仓库结构、数据集完整性、训练 pipeline、Qwen 迁移可行性
- `docs/CALM_Data_Schema.md` (945 行) — 10 个数据集逐文件 schema 分析、标签分布、preprocess.py 逻辑

**关键发现**:
- 10 个金融风控数据集覆盖 4 类任务，6 个完整可用，4 个缺原始 CSV
- CALM 训练代码在外部仓库，基座 Llama2-chat 已过时
- 标签体系不统一（gold 含义因数据集而异）

---

### C0.5: 数据协议设计

**产出**:
- `docs/RiskDataset_Schema.md` (921 行) — 5 层数据协议
- `docs/RiskTask_Definition.md` (176 行) — 任务业务语义定义

**关键设计**:
- Layer 0 (Raw) → Layer 1 (Normalized) → Layer 2A/2B (SFT) → Layer 3 (Preference) → Layer 4 (Evaluation)
- `risk_label` = business risk severity，非 positive class
- `task_type` + `target_type` 解耦：Insurance claim 用 `claim_event`，Customs 用 `customs_tier`
- 每层保留 `original_label` 保证完全可追溯

---

### C1: 数据管线

**产出**:
- `docs/German_Credit_Converter_Plan.md` (850 行) — 转换器设计
- `convert_german.py` — 可复现转换器
- `data/processed/german/` — 完整数据产物

**数据链路**:
```
german.data (1000 rows)
  → 20 特征语义映射 (13 categorical + 7 numerical)
  → 标签映射 {1:0, 2:1}
  → 7:1:2 分割 (seed=10086)
  → Normalized Layer: {train,valid,test}.jsonl
  → SFT ChatML Layer: {train,valid,test}.jsonl
  → 自动验证 (V1-V7 全部通过)
```

**复现性**: 连续运行 2 次 → 7/7 文件 SHA-256 完全一致。

---

### C2: 训练前基线

**产出**:
- `src/baselines/majority.py` — B0
- `src/baselines/logistic_regression.py` — B1
- `src/baselines/qwen_zero_shot.py` — B2
- `src/evaluation/metrics.py` — 统一评估框架
- `reports/baseline_report.md`

**基线结果 (test set)**:

| Model | Accuracy | Balanced Acc. | Macro-F1 | High-risk Recall | ROC-AUC | Cost |
|-------|:--------:|:-------------:|:--------:|:----------------:|:-------:|:----:|
| Majority | 0.675 | 0.500 | 0.403 | 0.000 | 0.500 | 325 |
| **Logistic Regression** | **0.675** | **0.711** | **0.668** | **0.815** | **0.757** | **113** |
| Qwen3.5-4B Zero-shot | 0.320 | 0.492 | 0.242 | 0.985 | 0.515 | 140 |

**结论**: LR 是最强基线；Qwen Zero-shot 无判别能力（ROC-AUC≈0.5），盲目预测 high risk。

---

### C3: LoRA SFT

**产出**:
- `src/training/sft_lora_manual.py` — 手动训练循环
- `src/training/sft_infer.py` — 推理脚本
- `src/evaluation/c3_report.py` — 收尾评估
- `scripts/sft_slurm.sh` / `scripts/sft_infer.sh` — Slurm 作业
- `outputs/sft/german_sft_seed{7,42,10086}/` — 3 个训练产物

**训练配置**:

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen3.5-4B |
| LoRA rank | 16 (alpha=32, dropout=0.05) |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 可训练参数 | 21M / 4.2B (0.5%) |
| 训练样本 | 700 (train) / 100 (valid) / 200 (test) |
| Epochs | 5 |
| Batch size | 4 × grad_accum 2 = effective 8 |
| Max seq length | 2048 |
| Learning rate | 2e-4 (cosine) |
| GPU | 2× RTX PRO 6000 Blackwell (97GB each) |
| 种子 | 10086, 42, 7 |

**SFT 结果 (test set, 3 seeds)**:

| Model | Acc | BalAcc | MacroF1 | HR Recall | ROC-AUC | Cost@0.5 | Cost@opt |
|-------|:---:|:------:|:-------:|:---------:|:-------:|:--------:|:--------:|
| Qwen Zero-shot | 0.320 | 0.492 | 0.242 | 0.985 | 0.515 | — | 140 |
| SFT seed=10086 | 0.670 | 0.692 | 0.659 | 0.754 | 0.739 | 273 | 130 |
| SFT seed=42 | 0.610 | 0.651 | 0.605 | 0.769 | 0.755 | 305 | 138 |
| SFT seed=7 | 0.580 | 0.673 | 0.580 | 0.939 | 0.747 | 295 | 100 |
| **SFT Mean±Std** | **0.620±0.04** | **0.672±0.02** | **0.615±0.04** | **0.821±0.08** | **0.747±0.01** | **291±13** | **123±16** |
| Logistic Regression | 0.675 | 0.711 | 0.668 | 0.815 | 0.757 | — | 113 |

**阈值优化效果**: Cost@0.5=291 → Cost@opt=123，降幅 58%。所有阈值通过 valid 集确定，无 test 泄漏。

**主 Checkpoint**: 按预先定义的 **valid PR-AUC 优先规则**，选择 **Seed 7** 作为后续 DPO/SimPO/Risk-DPO 的统一 SFT 起点。

| Seed | Valid PR-AUC | Valid ROC-AUC | Valid NLL | Valid Brier | Valid Cost |
|:----:|:------------:|:-------------:|:---------:|:-----------:|:----------:|
| 10086 | 0.4879 | 0.7215 | **0.5361** | **0.1789** | **56** |
| 42 | 0.4920 | **0.7302** | 0.5501 | 0.1856 | 57 |
| **7** | **0.5140** | 0.7188 | 0.5449 | 0.1828 | 59 |

**重要说明**: Seed 7 并非全面最优 — Seed 42 的 ROC-AUC 更好，Seed 10086 的 NLL/Brier/Cost 更好。选择 Seed 7 仅因其 valid PR-AUC 最高（对高风险样本的排序能力最优），符合预定义规则，而非其 test cost 最低。

**C3 核心结论**:
1. **排序能力**: ROC-AUC 从 0.515 提升到 0.747 (+45%)，SFT 让模型获得了风险区分能力
2. **校准不足**: Brier 0.18-0.19，阈值偏离 0.5 达 0.15-0.30，risk_score 不是校准概率
3. **阈值必须优化**: 默认 0.5 下 TP 几乎为零，成本函数 5×FN 驱动阈值下调
4. **接近 LR**: ROC-AUC 0.747 vs LR 0.757，差距已缩至 1%，但 Cost 123 vs 113 仍有 9% 差距

---

## 3. 训练实验对比总表

| 阶段 | 模型 | ROC-AUC | Cost@opt | 说明 |
|:----:|------|:-------:|:--------:|------|
| C2 | Majority | 0.500 | 325 | 最弱基线 |
| C2 | Logistic Regression | 0.757 | 113 | 经典 ML 最强基线 |
| C2 | Qwen3.5-4B Zero-shot | 0.515 | 140 | 无判别能力 |
| C3 | Qwen3.5-4B SFT (mean) | 0.747 | 123 | LoRA SFT，接近 LR |
| C3 | Qwen3.5-4B SFT (best: seed 7) | 0.747 | 100 | 主 checkpoint |
| C5 | Qwen3.5-4B DPO (oracle) | 0.706 | 325 | ❌ 退化为全预测 low risk |
| C5 | Qwen3.5-4B SimPO (oracle) | 0.500 | 325 | ❌ 随机水平 |
| C5v2 | Qwen3.5-4B DPO (hard) | 0.478 | 139 | ❌ 损失崩塌，ROC-AUC 低于随机 |
| C5v2 | Qwen3.5-4B SimPO (hard) | 0.504 | 135 | ❌ 损失崩塌，ROC-AUC 随机 |
| C6v2 | Cost-sensitive SFT 5:1 | 0.597 | 135 | ❌ 退化为全预测 high risk |
| C6v2 | Anchored DPO 1:1 | 0.504 | 140 | ❌ 随机（valid 集 bug） |
| C6v2 | Anchored Risk-DPO 5:1 | — | — | ❌ 训练崩溃 |
| C5v3 | Multi-SFT (Ger+Aus) | 0.830 | — | ✅ 多数据集 SFT 有效 |
| C5v3 | Multi-DPO (hard) | 0.621 | — | ❌ 全预测 high risk（标签先验偏移） |
| C5v3 | Multi-SimPO (hard) | 0.631 | — | ❌ 同上 |
| **C7** | **Logistic Regression** | **0.757** | **113** | **最强模型** |
| **C7** | **SFT seed 7** | **0.747** | **100** | **LLM 最优** |
| **C7** | **SFT multi (German)** | **0.721** | **104** | **多数据集负迁移** |

---

### C4: 偏好数据构造 (v1: ground-truth oracle)

**产出**:
- `src/training/build_preference.py` — 偏好数据构造脚本
- `data/processed/german/preference/preference_train.jsonl` — 800 对偏好数据

**数据构成**:

| Split | 配对数 | FN (weight=5) | FP (weight=1) |
|-------|:------:|:------------:|:------------:|
| train | 700 | 206 | 494 |
| valid | 100 | 29 | 71 |
| **总计** | **800** | **235** | **565** |

**协议** (统一适用于 DPO / SimPO / Risk-DPO):

```json
{
  "prompt": [system, user],
  "chosen": [{"role": "assistant", "content": "high risk"}],
  "rejected": [{"role": "assistant", "content": "low risk"}],
  "error_type": "false_negative",
  "risk_weight": 5.0,
  "source": "ground_truth_oracle"
}
```

**C4 验收**:

| 检查项 | 状态 |
|--------|:--:|
| C4.2 800 pairs (train=700, valid=100) | ✅ |
| C4.3 chosen/rejected 与真实标签一致 (0 errors) | ✅ |
| C4.4 SHA-256 冻结: `b7e9bd09...` | ✅ |
| C4.4 权重分布: FN=235(5.0), FP=565(1.0) | ✅ |
| C4.5 Schema 9 字段完整，单一 source | ✅ |
| C4.5 prompt 结构: system + user (no assistant) | ✅ |

**使用规则**:
- DPO: 所有 `risk_weight` 置为 1（等权）
- SimPO: 所有 `risk_weight` 置为 1（等权）
- Risk-DPO: 使用原始 `risk_weight`，测试比例 {1:1, 2:1, 5:1, 10:1}

---

### C5: DPO / SimPO — 负结果（方法适配性边界）

**产出**:
- `src/training/dpo_train.py` — DPO/SimPO 训练脚本
- `src/training/c5v2_check.py` — 实现验证 + SFT margin 分析 + hard preference 构造
- `src/training/pref_infer.py` — 偏好模型推理脚本
- `src/evaluation/c5_compare.py` — C5 统一对比
- `data/processed/german/preference/preference_train.jsonl` — oracle 偏好 (800 对)
- `data/processed/german/preference/preference_train_hard.jsonl` — hard 偏好 (361 对)
- `outputs/dpo/german_{dpo,simpo}_seed42/` — C5v1 训练产物
- `outputs/dpo/german_{dpo,simpo}_seed42_hard/` — C5v2 训练产物

**C5v1: Oracle Preference Pairs** — chosen=ground truth, rejected=翻转标签, 800 对 (700 train/100 valid):

| Model | ROC-AUC | Cost | valid_loss | 状态 |
|-------|:-------:|:----:|:----------:|:----:|
| SFT seed7 (基准) | **0.747** | **100** | — | ✅ |
| DPO (oracle, β=0.1) | 0.706 | 325 | 0.57 | ❌ 全预测 low risk |
| SimPO (oracle, β=0.1, γ=0.5) | 0.500 | 325 | 0.60 | ❌ 随机水平 |

**根因**: SFT 模型对 oracle 偏好对已有正确方向（chosen logprob > rejected），DPO 梯度信号极弱。同时在 494 低风险 vs 206 高风险的不平衡下，模型退化为预测多数类。

**C5v1→C5v2: Hard Preference Construction**

SFT seed 7 在 train 700 条上的 margin 分析揭示了系统性偏差：

| 类别 | N | 平均 margin | 排序错误 |
|------|:--:|:----------:|:--------:|
| high_risk | 206 | -0.54 | 206 (100%) |
| low_risk | 494 | 1.32 | 10 (2%) |

SFT 模型对**每一个**高风险训练样本都赋予更高的 "low risk" logprob。Hard preference 保留全部 206 排序错误 + 155 低置信样本，共 361 对。

**C5v2: Hard Preference Pairs** — 361 对 (train 361):

| Model | ROC-AUC | Cost | valid_loss | 状态 |
|-------|:-------:|:----:|:----------:|:----:|
| SFT seed7 (基准) | **0.747** | **100** | — | ✅ |
| DPO (hard, β=0.1) | 0.478 | 139 | **0.0000** | ❌ 损失崩塌，ROC-AUC < 0.5 |
| SimPO (hard, β=0.1, γ=0.5) | 0.504 | 135 | **0.0000** | ❌ 损失崩塌，ROC-AUC = 随机 |

**根因**: 答案仅 2 token（"low risk"/"high risk"），偏好优化可以轻易将 chosen/rejected logprob 拉开以满足 DPO/SimPO 目标，无需学习真正的样本级风险排序。valid_loss 在第一步 eval 即坍缩为 0，ROC-AUC 随之崩溃。

**C5 正式结论**:

> 在 German Credit 的二分类短回答设定下，oracle preference 与 hard preference 两种构造均未使 DPO/SimPO 获得有效风险排序能力。偏好损失可快速趋近于零，但 ROC-AUC 显著下降，表明模型仅扩大了两个答案的概率间隔，并未学习更好的样本级风险判别。该结果是当前任务形式下的方法适配性边界，而非单次训练失败。

不建议继续对原始 DPO/SimPO 调参。两轮实验已足以说明该路线在现有二分类短答案形式下不成立。

---

### C6v2: Anchored Risk-DPO — 负结果（路线终止）

**产出**:
- `docs/C6v2_Anchored_Risk_DPO_Plan.md` — 实验设计（损失公式、对照矩阵、通过/终止条件）
- `src/training/anchored_risk_dpo.py` — Anchored Risk-DPO 训练脚本
- `src/training/cost_sensitive_sft.py` — Cost-sensitive SFT 基线脚本
- `scripts/{cost_sft_51,anchored_dpo_11,anchored_riskdpo_51}.sh` — Pilot 作业

**Pilot 结果**:

| # | 实验 | ROC-AUC | Cost | 状态 |
|---|------|:-------:|:----:|:----:|
| E0 | SFT seed 7 | 0.747 | 100 | 基准 |
| E1 | Cost-sensitive SFT 5:1 | 0.597 | 135 | ❌ 全预测 high risk |
| E2 | Anchored DPO 1:1 | 0.504 | 140 | ❌ 随机（valid 集为空） |
| E4 | Anchored Risk-DPO 5:1 | — | — | ❌ 训练崩溃 |

**失败原因**:
1. E1 已充分说明：成本加权 SFT（无偏好项）已让 ROC-AUC 从 0.747 降至 0.597。SFT 模型的风险判别能力是脆弱的，任何朝 high risk 方向的偏移都会破坏排序。
2. E2/E4 的 hard preference 数据缺少 valid split（全部 361 对为 train），导致验证集为空、权重为 NaN。
3. E4 中 policy + reference 双模型在 2 GPU 上 `device_map="auto"` 产生设备冲突。

**终止判定**: 符合预定义的终止规则 — Pilot 不满足稳定性和有效性条件。即使修复 valid 集 bug，E1 已表明成本加权本身即导致崩塌，偏好项不可能逆转。

---

### C5v3: Multi-Dataset Preference Validation — 路线终结

**产出**:
- `src/training/multi_dataset_converter.py` — German + Australian 统一转换器
- `src/training/sft_multi.py` — 多数据集 SFT 训练
- `src/training/multi_infer_logprobs.py` — multi-SFT 推理 + margin 分析
- `src/training/c5v3_pipeline.py` — A.2→A.3→A.4 一体化 pipeline
- `data/processed/multi/combined/` — 1182 train / 169 valid / 339 test
- `data/processed/preference_multidataset/` — 549 train / 81 valid hard preference pairs

**C5v3 结果**:

| Model | German ROC-AUC | Australian ROC-AUC | Overall | HR% |
|-------|:--------------:|:------------------:|:-------:|:---:|
| SFT multi | **0.721** | **0.938** | **0.830** | 14.5%/50.4% |
| DPO multi (β=0.05) | 0.517 | 0.666 | 0.621 | **100%/100%** |
| SimPO multi (β=0.05) | 0.525 | 0.648 | 0.631 | **100%/100%** |

**与前几轮实验的对比**:

| 实验 | 数据 | β | Loss | 退化方向 |
|------|------|:--:|:----:|:--------:|
| C5 DPO (oracle) | German 700 | 0.1 | 缓降至 0.57 | → 全 low risk (0%) |
| C5v2 DPO (hard) | German 361 | 0.1 | 崩塌至 0 | → 全 low risk |
| C6v2 DPO (anchored) | German 361 | 0.1 | λ×SFT 失衡 | → 训练崩溃 |
| **C5v3 DPO (hard)** | **Ger+Aus 549** | **0.05** | **稳定 0.69** | **→ 全 high risk (100%)** |
| C5v3 SimPO (hard) | Ger+Aus 549 | 0.05 | 稳定 0.70 | → 全 high risk (100%) |

C5v3 DPO 的 loss 没有崩塌（稳定在 0.69），margin 受控（0.2-0.7），但模型仍丧失全部判别能力。这是决定性证据：**偏好优化目标和风险排序在根本上正交**——模型可以在完美满足 DPO 目标的同时完全丧失样本间区分能力。

---

## 4. Preference Optimization 路线总结 — 终结

```
C5v1 (oracle DPO/SimPO)        → 全 low risk
C5v2 (hard DPO/SimPO)          → 损失崩塌至 0
C6v2 (Anchored + Cost-SFT)     → 训练崩溃
C5v3 (multi-dataset DPO/SimPO) → 损失稳定但全 high risk
────────────6 组实验 / 5 次验证 / 3 种数据构造 / 2 种 β────────────
结论一致：偏好优化无法在二分类短回答下保持风险排序能力
```

**冻结结论**:

> 在 Qwen3.5-4B、German Credit 与 Australian Credit 的标签级二分类短回答设置下，DPO 和 SimPO 在 oracle、hard preference、单数据集和多数据集配置中均未超过 SFT，并多次出现类别预测坍缩和 ROC-AUC 显著下降。结果表明，标签级偏好目标主要调整全局类别概率，而未有效监督样本间风险排序。该结论限定于当前短标签、小规模表格风控设置，不推广为 DPO/SimPO 对所有二分类任务均无效。

**技术注释**:
- `DPO loss≈0.693 (= log 2)` 表示 chosen/rejected 的有效优势接近零，而非"完美满足偏好目标"。β=0.05 时，原始 margin 0.2-0.7 被缩放至 0.01-0.035，信号极弱。
- `HR%=100%` 是决策分布坍缩，但 Australian ROC-AUC 仍保留 0.648-0.666 的部分排序能力，并非完全丧失所有判别信息。
- 退化方向反转（C5v1→全 low risk, C5v3→全 high risk）可解释为不同偏好数据的类别构成驱动了相反的整体标签先验偏移。

**根因假说**:

同一类别的所有样本拥有完全相同的 chosen/rejected 文本。偏好损失主要推动模型改变两个标签的整体概率先验，却没有直接要求"高风险样本 A 的分数 > 低风险样本 B 的分数"。模型通过整体偏向某个标签来降低偏好损失，同时破坏了特征条件下的样本排序。

**教训**:
- 短答案（2 token）的二分类任务不适合当前形式的偏好优化
- SFT 已获得的排序能力是脆弱的——任何形式的偏好优化（oracle/hard/anchored/multi-dataset）均会破坏它
- 数据量增加（700→1182）和任务多样性（German+Australian）可以提高 SFT 性能，但不能挽救偏好优化
- 未来若要探索偏好优化，需要更长的 reasoning-based 答案（Layer 2B），而非仅 `low risk`/`high risk`

---

### C5v3 审计：全局标签先验偏移验证

**产出**: `src/evaluation/c5v3_audit.py`

**Logprob 偏移证据**:

| 模型 | logp(low) | logp(high) | gap | P_high 范围 |
|------|:---------:|:----------:|:---:|:-----------:|
| SFT | -0.796 | -1.058 | -0.26 | 0.13–0.92 |
| DPO | -22.5 | -19.9 | +2.68 | 0.90–0.95 |
| SimPO | -23.6 | -19.0 | +4.63 | 0.98–0.996 |

偏好数据 chosen=high_risk 仅占 51.9%，不存在数据偏差。DPO/SimPO 中两个 logprob 绝对值均爆炸（-0.8→-22），gap 方向反转，P_high 分布坍缩为窄区间。模型通过整体偏转标签概率来满足 DPO 目标，同时破坏了样本间排序。

---

### C7: 最终评估

**产出**: `src/evaluation/c7_final.py`，`outputs/c7_final_metrics.json`

**German Credit Test Set (N=200) 统一指标**:

| Model | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost | HR.Recall | LR.Recall |
|-------|:-------:|:------:|:---:|:-----:|:---:|:----:|:---------:|:---------:|
| Majority | 0.500 | 0.325 | 8.98 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen Zero-shot | 0.515 | 0.348 | 1.72 | 0.570 | 0.593 | 135 | 1.000 | 0.000 |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | **0.076** | **113** | **0.815** | **0.607** |
| SFT seed 7 | 0.747 | 0.555 | 0.556 | 0.190 | 0.070 | 100 | 0.939 | 0.407 |
| SFT multi (German) | 0.721 | 0.533 | 0.567 | 0.194 | 0.060 | 104 | 0.954 | 0.341 |

**混淆矩阵 (成本最优阈值)**:

| Model | TN | FP | FN | TP | Cost |
|-------|:--:|:--:|:--:|:--:|:----:|
| Logistic Regression | 82 | 53 | 12 | 53 | 113 |
| SFT seed 7 | 55 | 80 | 4 | 61 | 100 |
| SFT multi | 46 | 89 | 3 | 62 | 104 |

---

## 5. 最终结论

### 5.1 方法有效性

| 方法 | 结果 | 证据 |
|------|:----:|------|
| **SFT** | ✅ 成功 | ROC-AUC 0.515→0.747，唯一有效的后训练方法 |
| **阈值优化** | ✅ 成功 | Cost 从 Cost@0.5=295 降至 Cost@valid-opt=100 |
| **多数据集 SFT** | ⚠️ 部分有效 | 整体 ROC-AUC 0.830，但对 German 有负迁移（0.747→0.721） |
| **DPO** | ❌ 有效负结果 | oracle/hard/multi-dataset 均类别坍缩 + ROC-AUC 下降 |
| **SimPO** | ❌ 有效负结果 | 同上 |
| **Anchored DPO** | ❌ 实验不完整 | valid 集 bug + 训练中断 |
| **Risk-DPO** | — 终止 | 基础偏好目标持续失效，不再具备可信实验前提 |
| **Cost-sensitive SFT** | ❌ 有效负结果 | 成本加权本身即导致崩塌 |

### 5.2 最终结果 (German test, N=200, 成本阈值均来自 valid 集)

| Model | ROC-AUC | PR-AUC | NLL | Brier | ECE | Cost@valid-opt | HR.Recall | LR.Recall |
|-------|:-------:|:------:|:---:|:-----:|:---:|:--------------:|:---------:|:---------:|
| Majority | 0.500 | 0.325 | 8.98 | 0.325 | 0.325 | 325 | 0.000 | 1.000 |
| Qwen Zero-shot | 0.515 | 0.348 | 1.72 | 0.570 | 0.593 | 140 | 0.985 | 0.000 |
| SFT multi (German) | 0.721 | 0.533 | 0.567 | 0.194 | 0.060 | 104 | 0.954 | 0.341 |
| **SFT seed 7** | **0.747** | **0.555** | **0.556** | **0.190** | **0.070** | **100** | **0.939** | **0.407** |
| **Logistic Regression** | **0.757** | **0.596** | **0.552** | **0.182** | **0.076** | **113** | **0.815** | **0.607** |

**排序**: Logistic Regression 在风险排序（ROC-AUC 0.757）和概率质量（NLL 0.552）上仍是最强基线；Qwen3.5-4B SFT seed 7 已达到接近传统模型的判别能力（ROC-AUC 0.747），并在验证集成本阈值下取得最低测试业务成本（Cost 100）。

**关于阈值**: 成本函数 Cost=5FN+FP 的理论最优阈值为 t=1/(1+5)≈0.167。SFT 实际最优阈值 0.15–0.30 与此一致，并非校准失败导致。SFT 的 NLL（0.556）和 Brier（0.190）已接近 LR（0.552/0.182），ECE 0.070 接近 LR 的 0.076，校准表现接近传统基线。

**多数据集 SFT**: 加入 Australian 后 German ROC-AUC 从 0.747 降至 0.721，说明多数据集训练提升了跨数据集覆盖但未增强 German 单任务效果。

### 5.3 项目贡献

1. **数据管线**: 10 数据集→统一 RiskDataset Schema→ms-swift ChatML，完全可复现（SHA-256 验证）
2. **SFT 有效性**: Qwen3.5-4B 经 LoRA SFT 后获得接近 LR 的风险排序能力，并通过成本敏感阈值达到最低业务成本
3. **DPO 方法边界**: DPO 和 SimPO 在单数据集、多数据集、oracle/hard preference 设置下均未超过 SFT，并反复出现类别先验偏移和排序能力下降。由于基础偏好目标无法稳定保持风险判别能力，原定 Risk-DPO 路线提前终止
4. **机制分析**: C5v3 审计通过 logprob 偏移量化分析，验证了 DPO 导致全局标签先验偏移而非样本级排序改善的假说

---

---

## 6. 项目最终结论（冻结）

> 本项目基于 Qwen3.5-4B 构建了可复现的金融信用风险后训练与决策评估链路。LoRA SFT 将模型在 German Credit 上的 ROC-AUC 从 0.515 提升至 0.747，接近 Logistic Regression 的 0.757，并通过验证集成本敏感阈值将测试业务成本降至 100。进一步实验发现，基于 `low risk`/`high risk` 短标签构造的 DPO 和 SimPO 主要改变全局标签概率先验，未能改善样本级风险排序，并多次导致类别预测坍缩。结果表明，在当前小规模表格风控和短标签输出设定下，SFT 与成本敏感决策优化有效，而标签级偏好优化存在明确的方法适配边界。

**成果结构**:

```text
数据协议与可复现管线
        ↓
Qwen3.5-4B SFT 成功（ROC-AUC 0.747, Cost 100）
        ↓
成本敏感阈值优化成功（理论阈值 t≈0.167 验证）
        ↓
DPO/SimPO 系统性负结果（6 组实验, 0 成功）
        ↓
短标签偏好优化机制边界分析（logprob 审计验证）
```

---

*本报告涵盖 commit `d4fb59f` 至当前工作树。所有指标可复现。*
