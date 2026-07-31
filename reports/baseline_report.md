# German Credit: Pre-Training Baseline Report

> **阶段**: C2 — 训练前 Baseline
> **日期**: 2026-07-21
> **数据集**: German Credit (train=700, valid=100, test=200)
> **成本函数**: C = 5×FN + 1×FP

---

## Baseline Metrics (test set)

| Model | Accuracy | Balanced Acc. | Macro-F1 | High-risk Recall | ROC-AUC | PR-AUC | Cost | Valid Rate |
|-------|:--------:|:-------------:|:--------:|:----------------:|:-------:|:------:|:----:|:----------:|
| Majority | 0.6750 | 0.5000 | 0.4030 | 0.0000 | 0.5000 | 0.3250 | 325 | 1.0000 |
| Logistic Regression | 0.6750 | 0.7114 | 0.6680 | 0.8154 | 0.7566 | 0.5957 | 113 | 1.0000 |
| Qwen3.5-4B Zero-shot | 0.3200 | 0.4923 | 0.2424 | 0.9846 | 0.5152 | 0.3481 | 140 | 1.0000 |

## Per-model Analysis

### B0: Majority Class

始终预测 "low risk"。

- Accuracy 0.675 与标签分布 (67.5% low risk) 一致
- High-risk recall = 0：漏判所有高风险样本，Cost = 325 (65×5)
- 这是业务上不可接受的基线 — 全部高风险客户被放行

### B1: Logistic Regression

使用 20 个原始表格特征 (13 categorical → OneHot, 7 numerical → StandardScaler)。

- Balanced accuracy 0.7114，在所有模型中最高
- High-risk recall 0.8154：成功识别 81.5% 的高风险样本
- ROC-AUC 0.7566：具有一定的排序能力
- Cost = 113 (12×5 FN + 53×1 FP)
- 阈值在 valid 集优化到 0.20 (偏向召回率以降低 FN 成本)

**Logistic Regression 是当前最强基线。**

### B2: Qwen3.5-4B Zero-shot

不训练，直接使用 SFT prompt 让基础模型判断 "low risk" / "high risk"。
通过 token-level logprob 计算 p_high = exp(s_high) / (exp(s_low) + exp(s_high))。

- Accuracy 0.3200：远低于多数类基线
- High-risk recall 0.9846：几乎捕获所有高风险样本
- ROC-AUC 0.5152：接近随机 (0.5)，模型无判别能力
- Cost = 140：优于 Majority (325) 但弱于 Logistic Regression (113)
- 阈值在 valid 集优化到 0.80 (极高，反映了模型持续输出高 logprob 给 "high risk")

**Qwen3.5-4B 在 zero-shot 下严重偏向预测 "high risk"，缺乏对 German Credit 任务的领域理解。**

## 关键结论

1. **Logistic Regression 是最强基线** — 在小样本表格数据上，经典 ML 方法仍然优于未微调的 LLM。这不代表项目失败，而是正常的实验结果。

2. **Qwen Zero-shot 无判别能力** — ROC-AUC 接近 0.5。虽然 recall 极高 (0.98)，但这是以大量 false positive 为代价 (accuracy 仅 0.32)。模型不理解 German Credit 的任务语义。

3. **SFT 的目标** — 将 Qwen 从 "盲目预测 high risk" 提升到至少匹配 Logistic Regression 的水平。这是 C3 阶段的核心验证点:
   - 如果 Qwen SFT > LR → LLM 微调在该任务上有价值
   - 如果 Qwen SFT < LR → 需要更大的模型或不同的微调策略

4. **成本函数导致高召回偏好** — FN (漏判高风险) 成本是 FP 的 5 倍。Logistic Regression 的阈值自动下调至 0.20 以捕获更多高风险样本。

## 模型对比

```
                    Accuracy  High-risk Recall  Cost
Majority            ████████░  ░░░░░░░░░░       325
LogisticRegression  ████████░  ████████░░       113  ← 最强
Qwen Zero-shot      ███░░░░░░  ██████████       140
```

## 下一步: C3 LoRA SFT

SFT 训练目标是证明:

1. Qwen SFT 的 ROC-AUC 显著优于 Qwen Zero-shot (0.52 → target > 0.70)
2. Qwen SFT 的 Cost 接近或优于 Logistic Regression (140 → target < 113)
3. Qwen SFT 不过拟合 (valid/test gap 小)

---

*报告生成: outputs/baselines/*.jsonl + src/evaluation/metrics.py*
