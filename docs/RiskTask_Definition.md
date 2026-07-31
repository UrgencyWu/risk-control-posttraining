# RiskTask Definition: 金融风控任务语义定义

> **用途**: 在 `RiskDataset_Schema.md` 的基础上，集中定义 `task_type` + `target_type` 组合的业务语义。
> **何时查阅**: 多任务混合训练、评估指标选择、prompt 设计、下游任务扩展时。

---

## 任务总览

| task_type | target_type | 业务语义 | 数据集 | 风险方向 |
|-----------|-------------|---------|--------|---------|
| `credit_scoring` | `binary_risk` | **违约风险**: 借款人无法按期还款的可能性 | German, Australian, LendingClub | 高 risk_label = 违约 |
| `fraud_detection` | `binary_risk` | **欺诈风险**: 交易/行为存在欺骗意图的可能性 | CreditCardFraud, ccFraud | 高 risk_label = 欺诈 |
| `bankruptcy_prediction` | `binary_risk` | **企业破产风险**: 企业在预测期内资不抵债的可能性 | Polish, Taiwan | 高 risk_label = 破产 |
| `claim_analysis` | `claim_event` | **理赔事件**: 保单在有效期内是否发生赔付 | TravelInsurance, PortoSeguro | 高 risk_label = 发生理赔 |
| `customs_declaration` | `customs_tier` | **海关违规等级**: 申报行为违反海关法规的严重程度 | Customs | 高 risk_label = 更严重违规 |

---

## 逐任务定义

### 1. credit_scoring / binary_risk

```
业务语义: 违约风险 (Default Risk)
─────────────────────────────────
risk_label = 0 (low risk):   借款人预期按时还款，信用良好
risk_label = 1 (high risk):  借款人预期发生违约，信用不良

评估关注:
  - 优先降低 false_negative (漏判违约 → 坏账损失)
  - 可接受一定 false_positive (误判违约 → 拒贷损失)

典型 prompt 语境:
  "Evaluate the creditworthiness of a customer..."
  "Assess the client's loan status..."
```

### 2. fraud_detection / binary_risk

```
业务语义: 欺诈风险 (Fraud Risk)
─────────────────────────────────
risk_label = 0 (low risk):   交易/行为正常，无欺诈迹象
risk_label = 1 (high risk):  交易/行为存在欺诈嫌疑

评估关注:
  - 严重不平衡数据 (欺诈样本通常 < 5%)
  - 高召回率优先: 宁可误报 (false_positive) 不可漏报 (false_negative)
  - AUPRC 比 Accuracy 更有意义

典型 prompt 语境:
  "Detect the credit card fraud..."
  "Identify whether it constitutes customs fraud..."
```

### 3. bankruptcy_prediction / binary_risk

```
业务语义: 企业破产风险 (Bankruptcy Risk)
─────────────────────────────────────────
risk_label = 0 (low risk):   企业财务健康，短期内无破产风险
risk_label = 1 (high risk):  企业财务恶化，面临破产风险

评估关注:
  - 特征数量多 (Polish 64, Taiwan 95)
  - 高维财务比率 → LLM 需要处理长上下文
  - 不平衡: 破产企业占比 3-7%

典型 prompt 语境:
  "Predict whether the company will face bankruptcy..."
```

### 4. claim_analysis / claim_event

```
业务语义: 理赔事件 (Claim Event)
─────────────────────────────────
risk_label = 0 (no event):   保单未发生理赔
risk_label = 1 (event):      保单已发生理赔

与 binary_risk 的区别:
  - binary_risk: 评估的是"未来风险" (预测)
  - claim_event:  记录的是"已发生事件" (分类)

  claim=1 不必然等于 "这个客户是坏客户"。
  它可能是一个合理的理赔（客户购买了保险，确实发生了意外）。
  但在保险风险建模中，发生过理赔的客户在未来更可能再次理赔。

  因此 claim_analysis 在 SFT 阶段仍可输出 "low risk" / "high risk"，
  但业务上应理解为 "claim likelihood" 而非 "customer risk"。

评估关注:
  - 极度不平衡 (Travel Insurance claim rate ~1.5%)
  - 特征包含 categorical (Agency, Product Name) + numerical (Net Sales, Age)

典型 prompt 语境:
  "Identify the claim status of insurance companies..."
  "Identify whether or not to files a claim..."
```

### 5. customs_declaration / customs_tier

```
业务语义: 海关违规等级 (Customs Violation Tier)
───────────────────────────────────────────────
risk_label = 0:  正常申报 (compliant)
risk_label = 1:  一般违规 (fraud — 意图减少关税)
risk_label = 2:  严重违规 (critical fraud — 威胁公共安全)

与 binary_risk 的区别:
  - customs_tier 是三分类问题
  - risk_label=2 (critical fraud) 的后果远严重于 risk_label=1 (tax evasion)
  - 在 Binary 训练时可合并 1+2 → "high risk"
  - 在 Multiclass 训练时保留三层独立

评估关注:
  - 二层 (binary) vs 三层 (multiclass) 的选择需业务确认
  - critical fraud 样本极少 (~0.9%)
  - 漏判 critical fraud 的代价极高 (公共安全威胁)

典型 prompt 语境:
  "Identify the provided customs import declaration information to determine
   whether it constitutes customs fraud..."
```

---

## 多任务训练时的任务区分

当混合多个 task_type 训练时，通过 system prompt 区分任务:

```json
// credit_scoring
{"messages": [{"role": "system", "content": "You are a financial risk assessment expert..."}]}

// fraud_detection
{"messages": [{"role": "system", "content": "You are a financial fraud detection expert..."}]}

// bankruptcy_prediction
{"messages": [{"role": "system", "content": "You are a corporate financial analyst..."}]}

// claim_analysis
{"messages": [{"role": "system", "content": "You are an insurance risk analyst..."}]}

// customs_declaration
{"messages": [{"role": "system", "content": "You are a customs compliance analyst..."}]}
```

`metadata.task_type` 字段可用于训练后按任务拆分评估。

---

## 扩展指南

### 新增任务时

1. 确定 `task_type` (新值或复用已有)
2. 确定 `target_type` (binary_risk / claim_event / customs_tier / 新增类型)
3. 明确 `risk_label` 的语义: label=1 在业务上代表什么
4. 编写对应的 system prompt
5. 更新本文件

### 新增 target_type 时

| target_type | 何时使用 | risk_label 域 | 示例 |
|-------------|---------|:------------:|------|
| `binary_risk` | 标准二元风险评估 | {0, 1} | 违约、欺诈、破产 |
| `claim_event` | 事件是否发生 | {0, 1} | 理赔 |
| `customs_tier` | 有序多级风险 | {0, 1, 2} | 海关违规等级 |
| `credit_rating` (未来) | 有序多级信用评级 | {0..N} | AAA / AA / A / BBB / ... |
| `fraud_amount` (未来) | 回归: 欺诈金额估计 | R+ | 预测欺诈交易金额 |

---

*本文件与 `docs/RiskDataset_Schema.md` 配套使用。字段定义见 Schema 文档，业务语义见本文件。*
