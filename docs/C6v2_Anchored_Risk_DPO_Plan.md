# C6v2: Anchored Risk-DPO 设计文档

> **定位**: 最后一次受控验证，不是默认一定有效
> **前提**: C5 两轮 DPO/SimPO 均已失败（oracle + hard preference 均导致损失崩塌/排序退化）
> **日期**: 2026-07-22

---

## 1. 动机

C5 的 DPO/SimPO 在二分类短回答（2 token）场景下，偏好损失可快速趋近零，但 ROC-AUC 显著下降。模型仅扩大两个答案的概率间隔，未学习更好的样本级风险判别。

C6v2 验证一个假设：

> SFT 锚定项能否阻止 DPO 损失坍塌，同时用非对称成本权重降低高风险漏判。

如果假设不成立，则终止 Preference Optimization 路线，直接进入 C7 最终评估。

---

## 2. 损失函数

### 2.1 核心公式

$$
\mathcal{L} = \frac{1}{N} \sum_i \tilde{w}_i \cdot \mathcal{L}_{\text{DPO}, i} + \lambda \cdot \mathcal{L}_{\text{SFT}}
$$

| 项 | 含义 | 作用 |
|----|------|------|
| $\mathcal{L}_{\text{DPO}, i}$ | 标准 DPO loss（含参考模型） | 拉开 chosen/rejected 间偏好 |
| $\mathcal{L}_{\text{SFT}}$ | SFT cross-entropy（同 C3 SFT） | 锚定模型不偏离有效策略 |
| $\tilde{w}_i$ | 归一化成本权重 | 高风险样本获得更高权重 |
| $\lambda$ | SFT 锚定强度 | 控制保留 vs 优化 trade-off |

### 2.2 DPO 项

```text
L_DPO = -log σ( β × (log_π(chosen|x) / log_πref(chosen|x) - log_π(rejected|x) / log_πref(rejected|x)) )
```

β=0.1，同 C5。若 pilot 中仍损失崩塌，优先降低 β 至 0.05 或 0.01。

### 2.3 SFT 锚定项

```text
L_SFT = CrossEntropy(π(x), y_true)
```

即对 chosen 序列的标准 language modeling loss。y_true 来自 preference 数据中 `chosen` 的 token 序列。

### 2.4 权重归一化

原始权重：
- false_negative（gt=1 预测为 low risk）：$w_{\text{raw}} = R$
- false_positive（gt=0 预测为 high risk）：$w_{\text{raw}} = 1$

归一化：

$$
\tilde{w}_i = \frac{w_{\text{raw}, i}}{\bar{w}_{\text{raw}}}
\quad\text{其中}\quad
\bar{w}_{\text{raw}} = \frac{1}{N} \sum_i w_{\text{raw}, i}
$$

这确保 $\sum \tilde{w}_i = N$，整体梯度尺度不因权重配置变化而改变。损失项之间的比较仅通过 λ 控制。

### 2.5 λ 校准

pilot 阶段固定 λ=1.0。若 SFT 锚定过强（DPO 无效果），降为 0.5 或 0.1。若 SFT 锚定过弱（仍崩塌），升为 2.0。

---

## 3. 实验矩阵

| # | 实验 | 权重配置 | λ | 作用 |
|---|------|:-------:|:--:|------|
| E0 | SFT seed 7 | — | — | 原始基准 |
| E1 | Cost-sensitive SFT 5:1 | FN=5, FP=1 | — | 判断收益是否仅来自加权分类 |
| E2 | Anchored DPO 1:1 | FN=1, FP=1 | 1.0 | SFT 锚定本身是否有效 |
| E3 | Anchored Risk-DPO 2:1 | FN=2, FP=1 | 1.0 | 低强度成本权重 |
| E4 | Anchored Risk-DPO 5:1 | FN=5, FP=1 | 1.0 | German Credit 原始成本比例 |
| E5 | Anchored Risk-DPO 10:1 | FN=10, FP=1 | 1.0 | 检查过度风险偏置 |

**必须对照组**:
- E1（Cost-sensitive SFT）：若 E3-E5 优于 E2 但不优于 E1，说明收益仅来自加权分类，偏好项无额外价值
- E2（Anchored DPO 1:1）：若 E2-E5 均不优于 E0，说明 SFT 锚定本身无效

---

## 4. 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3.5-4B | 同 C3 |
| 起点 | SFT seed 7 best_adapter | 固定 |
| 偏好数据 | hard preference (361 train) | C5v2 产出，SHA-256: `ce9d3049...` |
| LoRA | r=16, alpha=32, target=q/k/v/o/gate/up/down | 同 C3 |
| Batch | 1 × grad_accum 8 = effective 8 | 降低单步 batch 以观察 loss 变化 |
| Epochs | 2 (pilot) | 减少训练量，优先观察稳定性 |
| LR | 5e-6 | 低于 C5 的 1e-5 |
| β (DPO) | 0.1 | 同 C5；若崩塌降至 0.05 或 0.01 |
| λ (SFT anchor) | 1.0 | pilot 默认值 |
| 参考模型 | SFT seed 7 (frozen) | 同 C5 DPO |
| Max seq len | 2048 | 同 C3 |
| 随机种子 | 42 | pilot 单种子 |
| 硬件 | 2× RTX PRO 6000 Blackwell | Slurm |

---

## 5. 监控指标（Pilot 阶段，每 10 steps 记录）

| 指标 | 含义 | 正常范围 | 危险信号 |
|------|------|---------|---------|
| DPO loss | 偏好损失 | ~0.4-0.7 | →0 或 →1 |
| SFT loss | 锚定损失 | ~0.2-0.5 | 持续上升 |
| Chosen logprob | π(chosen|x) | -0.5 ~ 0 | >0（过拟合） |
| Rejected logprob | π(rejected|x) | -5 ~ -2 | 骤降（崩坏） |
| Chosen/Rejected margin | logp_c - logp_r | 0.5-3.0 | >10（极端分离） |
| Valid ROC-AUC | 风险排序 | ≥ 0.72 | < 0.70（退化） |
| Valid PR-AUC | 高风险排序 | ≥ 0.48 | < 0.45 |
| Valid NLL | 校准 | < 0.65 | > 1.0 |
| Valid Brier | 校准 | < 0.22 | > 0.30 |
| Valid Cost (opt) | 业务成本 | 50-70 | > 80 |
| HR prediction rate | 预测为 high risk 比例 | 20-80% | < 10% 或 > 95% |

---

## 6. Pilot 通过条件

### 6.1 稳定性通过（必须全部满足）

- [ ] valid loss（DPO + λ×SFT）不立即坍缩至 0
- [ ] valid ROC-AUC ≥ SFT - 0.02（即 ≥ 0.72）
- [ ] HR prediction rate 在 20%-80% 之间
- [ ] valid Cost 不劣于 SFT（≤ SFT cost + 10）

### 6.2 有效性通过（至少满足一项）

- [ ] Anchored Risk-DPO Cost < Anchored DPO 1:1 Cost（证明风险权重有效）
- [ ] Anchored Risk-DPO Cost < Cost-sensitive SFT Cost（证明偏好项有额外价值）

### 6.3 终止规则

若 pilot（E2 + E4）出现以下任一情况，停止继续调参，直接进入 C7：

- DPO loss 在 5 步内降至 < 0.01
- Valid ROC-AUC 比 SFT 下降超过 0.05
- HR prediction rate 不足 5% 或超过 95%
- 全部不满足 6.1 或 6.2

---

## 7. 执行顺序

```text
Step 1: 实现 Anchored Risk-DPO 训练脚本
        (L = w̃ × L_DPO + λ × L_SFT)

Step 2: 实现 Cost-sensitive SFT 脚本
        (L = w_ce × CrossEntropy, FN:FP 权重)

Step 3: Pilot — E1 + E2 + E4（3 个 Slurm 作业）
        - Cost-sensitive SFT 5:1
        - Anchored DPO 1:1
        - Anchored Risk-DPO 5:1

Step 4: 评估 Pilot
        - 检查 6.1 稳定性条件
        - 检查 6.2 有效性条件

Step 5a (通过): 补跑 E3 + E5（2:1, 10:1）
Step 5b (未通过): 终止，记录负结果，进入 C7
```

---

## 8. 判定矩阵

| 场景 | Anchored Risk-DPO vs SFT | vs Cost-SFT | vs Anchored DPO 1:1 | 结论 |
|------|:--:|:--:|:--:|------|
| A | ✅ | ✅ | ✅ | 偏好优化 + 成本权重双重有效 |
| B | ✅ | ✅ | ❌ | 成本权重是主因，偏好项有一定价值 |
| C | ✅ | ❌ | ❌ | 只有 SFT 锚定在起作用，成本权重和偏好项均无效 |
| D | ❌ | ❌ | ❌ | C6v2 失败，偏好优化路线终止 |

---

## 9. 文件清单

训练完成后输出：

```text
src/training/
├── anchored_risk_dpo.py       # Anchored Risk-DPO 训练脚本
└── cost_sensitive_sft.py      # Cost-sensitive SFT 脚本

scripts/
├── anchored_dpo_11.sh         # E2: Anchored DPO 1:1
├── anchored_riskdpo_51.sh     # E4: Anchored Risk-DPO 5:1
└── cost_sensitive_sft_51.sh   # E1: Cost-sensitive SFT 5:1

outputs/dpo/
├── german_anchored_dpo_11/
├── german_anchored_riskdpo_51/
└── german_cost_sft_51/
```

---

*本设计基于 C5 两轮负结果。所有超参数、对照实验和终止条件已预先冻结。若 pilot 不通过，不得继续调参。*
