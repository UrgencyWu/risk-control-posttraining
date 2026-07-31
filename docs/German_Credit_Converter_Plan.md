# German Credit Converter Plan

> **阶段**: C1.1 — Converter Design
> **设计日期**: 2026-07-21
> **依赖文档**:
> - `docs/CALM_Data_Schema.md` — CALM 数据资产审计
> - `docs/RiskDataset_Schema.md` — 5 层数据协议定义
> - `docs/RiskTask_Definition.md` — 任务语义定义
>
> **原则**: 仅设计，不实现代码。不修改已有文件。不生成训练数据。

---

## 目录

1. [第一部分: 输入数据分析](#第一部分-输入数据分析)
2. [第二部分: 原始字段映射设计](#第二部分-原始字段映射设计)
3. [第三部分: Normalized Layer 设计](#第三部分-normalized-layer-设计)
4. [第四部分: 标签映射设计](#第四部分-标签映射设计)
5. [第五部分: Prompt 构造设计](#第五部分-prompt-构造设计)
6. [第六部分: MVP SFT 输出设计](#第六部分-mvp-sft-输出设计)
7. [第七部分: 数据划分设计](#第七部分-数据划分设计)
8. [第八部分: 数据验证设计](#第八部分-数据验证设计)
9. [第九部分: 输出文件设计](#第九部分-输出文件设计)
10. [第十部分: 未来扩展接口](#第十部分-未来扩展接口)

---

## 第一部分: 输入数据分析

### German Raw Dataset Specification

**文件依据**: `data/credit_scoring/German/german.data`

| 属性 | 值 |
|------|-----|
| **文件** | `german.data` |
| **路径** | `data/credit_scoring/German/german.data` |
| **格式** | 空格分隔，无表头 |
| **编码** | 分类变量使用字符串编码 (如 `A11`, `A34`)，数值变量使用数字 |
| **行数** | 1,000 |
| **列数** | 21 (20 features + 1 label) |
| **标签列位置** | 第 21 列 (索引 20，最后一列) |
| **原始标签** | `1` = good (信用良好), `2` = bad (信用不良) |
| **标签分布** | good: 700 (70%), bad: 300 (30%) |

### 原始数据示例 (前 3 行)

```
A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1
```

### 特征分类

| 类别 | 数量 | 索引 |
|------|:---:|------|
| 数值特征 | 7 | 1, 4, 7, 10, 12, 15, 17 |
| 分类特征 (含映射表) | 13 | 0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19 |

### 现有预处理代码

**文件**: `data/credit_scoring/German/prepocess.py`

已有功能:
- `dict`: 13 个分类特征的 raw_code → 人类可读描述映射表 (`prepocess.py:24-78`)
- `mean_list`: 20 个特征的英文名称 (`prepocess.py:16-22`)
- `process()`: 生成 CALM Raw 格式 `{id, query, answer, choices, gold, text}` (`prepocess.py:85-99`)
- `json_save()`: 输出 Parquet 和 JSONL (`prepocess.py:102-116`)
- 分割逻辑: `train:dev:test = 7:1:2`, seed=10086 (`prepocess.py:135-150`)

---

## 第二部分: 原始字段映射设计

### German Feature Mapping Table

**文件依据**: `data/credit_scoring/German/prepocess.py:16-78`

| Index | Raw Code | Semantic Field | Description | Type |
|:-----:|----------|----------------|-------------|------|
| 0 | `A11` | `checking_account_status` | Status of existing checking account | categorical |
| — | `A12` | (同上) | 0 ≤ balance < 200 DM | categorical |
| — | `A13` | (同上) | balance ≥ 200 DM or salary assignment | categorical |
| — | `A14` | (同上) | No checking account | categorical |
| 1 | *numeric* | `duration_month` | Duration in month | numerical |
| 2 | `A30` | `credit_history` | Credit history | categorical |
| — | `A31` | (同上) | All credits at this bank paid back duly | categorical |
| — | `A32` | (同上) | Existing credits paid back duly till now | categorical |
| — | `A33` | (同上) | Delay in paying off in the past | categorical |
| — | `A34` | (同上) | Critical account / other credits existing | categorical |
| 3 | `A40` | `purpose` | Purpose of loan | categorical |
| — | `A41` | (同上) | Car (used) | categorical |
| — | `A42` | (同上) | Furniture or equipment | categorical |
| — | `A43` | (同上) | Radio or television | categorical |
| — | `A44` | (同上) | Domestic appliances | categorical |
| — | `A45` | (同上) | Repairs | categorical |
| — | `A46` | (同上) | Education | categorical |
| — | `A47` | (同上) | Vacation | categorical |
| — | `A48` | (同上) | Retraining | categorical |
| — | `A49` | (同上) | Business | categorical |
| — | `A410` | (同上) | Others | categorical |
| 4 | *numeric* | `credit_amount` | Credit amount (in DM) | numerical |
| 5 | `A61` | `savings_account` | Savings account or bonds | categorical |
| — | `A62` | (同上) | 100 ≤ balance < 500 DM | categorical |
| — | `A63` | (同上) | 500 ≤ balance < 1000 DM | categorical |
| — | `A64` | (同上) | balance ≥ 1000 DM | categorical |
| — | `A65` | (同上) | Unknown or no savings account | categorical |
| 6 | `A71` | `employment_since` | Present employment since | categorical |
| — | `A72` | (同上) | Less than 1 year | categorical |
| — | `A73` | (同上) | 1 ≤ years < 4 | categorical |
| — | `A74` | (同上) | 4 ≤ years < 7 | categorical |
| — | `A75` | (同上) | 7 years or more | categorical |
| 7 | *numeric* | `installment_rate` | Installment rate (% of disposable income) | numerical |
| 8 | `A91` | `personal_status_sex` | Personal status and sex | categorical |
| — | `A92` | (同上) | Female: divorced/separated/married | categorical |
| — | `A93` | (同上) | Male and single | categorical |
| — | `A94` | (同上) | Male and married or widowed | categorical |
| — | `A95` | (同上) | Female and single | categorical |
| 9 | `A101` | `other_debtors` | Other debtors or guarantors | categorical |
| — | `A102` | (同上) | Co-applicant | categorical |
| — | `A103` | (同上) | Guarantor | categorical |
| 10 | *numeric* | `residence_since` | Present residence since (years) | numerical |
| 11 | `A121` | `property` | Property | categorical |
| — | `A122` | (同上) | Building society savings / life insurance | categorical |
| — | `A123` | (同上) | Car or other | categorical |
| — | `A124` | (同上) | Unknown or no property | categorical |
| 12 | *numeric* | `age_years` | Age in years | numerical |
| 13 | `A141` | `other_installment_plans` | Other installment plans | categorical |
| — | `A142` | (同上) | Stores | categorical |
| — | `A143` | (同上) | None | categorical |
| 14 | `A151` | `housing` | Housing | categorical |
| — | `A152` | (同上) | Own | categorical |
| — | `A153` | (同上) | For free | categorical |
| 15 | *numeric* | `existing_credits_count` | Number of existing credits at this bank | numerical |
| 16 | `A171` | `job` | Job | categorical |
| — | `A172` | (同上) | Unskilled or resident | categorical |
| — | `A173` | (同上) | Skilled employee or official | categorical |
| — | `A174` | (同上) | Management / self-employed / highly qualified | categorical |
| 17 | *numeric* | `maintenance_liability_count` | Number of people being liable to provide maintenance for | numerical |
| 18 | `A191` | `telephone` | Telephone | categorical |
| — | `A192` | (同上) | Yes, registered under the customer's name | categorical |
| 19 | `A201` | `foreign_worker` | Foreign worker | categorical |
| — | `A202` | (同上) | No (domestic) | categorical |
| — | — | **(Label)** | — | — |
| 20 | `1` / `2` | `credit_risk` (target) | Credit risk: 1=good, 2=bad | binary label |

### 字段映射摘要

- **总特征数**: 20
- **数值特征**: 7 个 (duration_month, credit_amount, installment_rate, residence_since, age_years, existing_credits_count, maintenance_liability_count)
- **分类特征**: 13 个 (含 raw_code → description 映射)
- **语义字段名**: 全部来自 CALM `mean_list`，转为 snake_case
- **描述文本**: 全部来自 CALM `dict`，可直接复用

### 敏感/保护属性识别

**文件依据**: `src/bias/bias-german.py`, `src/bias/process.py`

| Index | Semantic Field | 保护属性 | 二值化规则 |
|:-----:|----------------|---------|-----------|
| 8 | `personal_status_sex` | gender | A91/A93/A94 → male(0), A92/A95 → female(1) |
| 12 | `age_years` | age_group | ≤45 → young(0), >45 → old(1) |
| 19 | `foreign_worker` | foreign_status | A201 → yes(0), A202 → no(1) |

---

## 第三部分: Normalized Layer 设计

### 目标

将 CALM Raw 单条记录转换为符合 `docs/RiskDataset_Schema.md` Layer 1 的 JSON 格式。

### JSON Schema

```json
{
  "sample_id": "german_0",
  "dataset": "German",
  "split": "train",
  "task_type": "credit_scoring",
  "target_type": "binary_risk",
  "risk_label": 0,
  "original_label": {
    "value": 1,
    "meaning": "good",
    "raw_format": "int"
  },
  "calm_gold": 0,
  "text": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. The state of Duration in month is 10...",
  "prompt_format": "description",
  "feature_count": 20,
  "features": {
    "checking_account_status": "bigger than 0 DM but smaller than 200 DM",
    "duration_month": "10",
    "credit_history": "existing credits paid back duly till now",
    "purpose": "furniture or equipment",
    "credit_amount": "1521",
    "savings_account": "smaller than 100 DM",
    "employment_since": "bigger than 1 smaller than 4 years",
    "installment_rate": "4",
    "personal_status_sex": "male: divorced or separated",
    "other_debtors": "none",
    "residence_since": "2",
    "property": "car or other",
    "age_years": "31",
    "other_installment_plans": "none",
    "housing": "own",
    "existing_credits_count": "1",
    "job": "unskilled or resident",
    "maintenance_liability_count": "1",
    "telephone": "none",
    "foreign_worker": "yes"
  },
  "protected_attributes": {
    "gender": "male",
    "age_group": "young",
    "foreign_status": "yes"
  }
}
```

### 字段来源映射

| 字段 | 来源 | 计算方式 |
|------|------|---------|
| `sample_id` | **新增** | `f"german_{id}"` (CALM `id` 字段) |
| `dataset` | **新增** | 固定值 `"German"` |
| `split` | **新增** | 按 7:1:2 分割后赋值 `train`/`valid`/`test` |
| `task_type` | **新增** | 固定值 `"credit_scoring"` |
| `target_type` | **新增** | 固定值 `"binary_risk"` |
| `risk_label` | 计算 | `GermanAdapter.map_risk_label(original_label.value)` → `{1:0, 2:1}` |
| `original_label.value` | `german.data` 最后一列 | 直接读取 |
| `original_label.meaning` | 查表 | `1 → "good"`, `2 → "bad"` |
| `original_label.raw_format` | **新增** | 固定值 `"int"` |
| `calm_gold` | CALM `gold` 字段 | 与 `risk_label` 相同映射 (German: `{1:0, 2:1}`) |
| `text` | CALM `text` 字段 | `prepocess.py:88-95` 的拼接结果 |
| `prompt_format` | **新增** | 固定值 `"description"` |
| `feature_count` | **新增** | 固定值 `20` |
| `features.{key}` | CALM `dict` 映射 + 原始值 | 分类: dict[raw_code], 数值: 直接转字符串 |
| `protected_attributes` | **新增** (计算) | `personal_status_sex` → gender 二值化; `age_years` → age_group; `foreign_worker` → foreign_status |

### features 字段的 key 命名规则

| 规则 | 示例 |
|------|------|
| 使用 snake_case | `checking_account_status` |
| 数值特征名直接来自 `mean_list` | `duration_month`, `credit_amount` |
| 分类特征名以 `_status` / `_type` 等结尾 | `checking_account_status`, `employment_since` |
| value 使用 human-readable 描述 (非 raw_code) | `"bigger than 0 DM but smaller than 200 DM"` 而非 `"A12"` |

---

## 第四部分: 标签映射设计

### 映射定义

```
CALM 原始标签
     │
     ├── 1 = "good" (信用良好)
     │       │
     │       └── risk_label = 0 (low risk)
     │           业务含义: 低违约风险，预期正常还款
     │
     └── 2 = "bad" (信用不良)
             │
             └── risk_label = 1 (high risk)
                 业务含义: 高违约风险，预期发生违约
```

### 字段保留规则

| 字段 | 保留原因 |
|------|---------|
| `original_label.value` | 审计追踪: 可回退到原始标签 |
| `original_label.meaning` | 人类可读: 无需查表即可理解标签语义 |
| `calm_gold` | 论文复现: 与 CALM 原论文结果对比 |
| `risk_label` | 训练使用: 统一所有 binary_risk 任务的标签空间 |

### 不使用的术语

- ❌ "positive class" / "negative class"
- ❌ "正样本" / "负样本"
- ✅ "low risk" / "high risk" (业务风险严重度)

### 标签转换: Adapter 负责语义，BaseConverter 不知道含义

**设计原则**: `BaseConverter` 不包含任何标签映射逻辑。标签语义由每个 `DatasetAdapter` 的 `map_risk_label()` 方法显式定义。

**GermanAdapter 实现**:

```python
class GermanAdapter:
    def map_risk_label(self, original_label: int) -> int:
        """German: 1=good → low risk, 2=bad → high risk"""
        mapping = {
            1: 0,  # good → low risk
            2: 1,  # bad  → high risk
        }
        return mapping[original_label]
```

**验证**:
```
original_label = 1 (good) → mapping[1] → risk_label = 0 (low risk)   ✓
original_label = 2 (bad)  → mapping[2] → risk_label = 1 (high risk)  ✓
```

**为什么不用公式 `original - 1`**:

| 数据集 | 原始标签 | 需要的映射 | `orig - 1` 是否正确 |
|--------|---------|-----------|:-------------------:|
| German | 1=good, 2=bad | 1→0, 2→1 | ✅ 恰好成立 |
| Australian | 1=good, 0=bad | 1→0, 0→1 | ❌ 不成立 |
| Taiwan | 0=non-bankrupt, 1=bankrupt | 0→0, 1→1 | ❌ 恒等映射 |
| Credit Card Fraud | 0=normal, 1=fraud | 0→0, 1→1 | ❌ 恒等映射 |

因此 **`BaseConverter` 不应内置任何 `original - 1` 假设**。每个 Adapter 通过显式 mapping dict 定义自己的标签语义。

---

## 第五部分: Prompt 构造设计

### 转换流程

```
Normalized Layer                    SFT ChatML Layer
─────────────────                   ────────────────
sample_id                           messages[0] (system)
dataset         ──┐                 messages[1] (user)
task_type       ──┼─ metadata       messages[2] (assistant)
risk_label      ──┘
text             ─── user content
risk_label       ─── assistant content ("low risk" / "high risk")
```

### System Prompt

**固定内容** (来自 `docs/RiskTask_Definition.md` §1):

```
You are a financial risk assessment expert. Evaluate the creditworthiness
based on the customer's financial profile. Classify the risk level as:
- low risk: the customer is likely to repay
- high risk: the customer is likely to default
Respond with only 'low risk' or 'high risk'.
```

### User Prompt

**来源**: Layer 1 `text` 字段

```
The state of Status of existing checking account is bigger than 0 DM but
smaller than 200 DM. The state of Duration in month is 10. The state of
Credit history is existing credits paid back duly till now. The state of
Purpose is furniture or equipment. The state of Credit amount is 1521.
The state of Savings account or bonds is smaller than 100 DM. The state of
Present employment since is bigger than 1 smaller than 4 years. The state of
Installment rate in percentage of disposable income is 4. The state of
Personal status and sex is male: divorced or separated. The state of Other
debtors or guarantors is none. The state of Present residence since is 2.
The state of Property is car or other. The state of Age in years is 31.
The state of Other installment plans is none. The state of Housing is own.
The state of Number of existing credits at this bank is 1. The state of Job
is unskilled or resident. The state of Number of people being liable to
provide maintenance for is 1. The state of Telephone is none. The state of
foreign worker is yes.
```

**注意**: user prompt 直接使用 CALM `text` 字段。相比 CALM `query` 字段:
- ❌ 移除了前导的 `"Evaluate the creditworthiness...\nText: '"`
- ❌ 移除了末尾的 `"'\nAnswer:"`
- ✅ 仅保留纯特征描述 — system prompt 取代了原来的内嵌指令

### Assistant Output

| risk_label | assistant content |
|:----------:|------|
| 0 | `"low risk"` |
| 1 | `"high risk"` |

### 完整 ChatML 消息

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a financial risk assessment expert. Evaluate the creditworthiness based on the customer's financial profile. Classify the risk level as:\n- low risk: the customer is likely to repay\n- high risk: the customer is likely to default\nRespond with only 'low risk' or 'high risk'."
    },
    {
      "role": "user",
      "content": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. The state of Duration in month is 10. The state of Credit history is existing credits paid back duly till now. The state of Purpose is furniture or equipment. The state of Credit amount is 1521. The state of Savings account or bonds is smaller than 100 DM. The state of Present employment since is bigger than 1 smaller than 4 years. The state of Installment rate in percentage of disposable income is 4. The state of Personal status and sex is male: divorced or separated. The state of Other debtors or guarantors is none. The state of Present residence since is 2. The state of Property is car or other. The state of Age in years is 31. The state of Other installment plans is none. The state of Housing is own. The state of Number of existing credits at this bank is 1. The state of Job is unskilled or resident. The state of Number of people being liable to provide maintenance for is 1. The state of Telephone is none. The state of foreign worker is yes."
    },
    {
      "role": "assistant",
      "content": "low risk"
    }
  ],
  "metadata": {
    "sample_id": "german_0",
    "dataset": "German",
    "split": "train",
    "task_type": "credit_scoring",
    "target_type": "binary_risk",
    "risk_label": 0,
    "original_label": {"value": 1, "meaning": "good", "raw_format": "int"},
    "prompt_format": "description"
  }
}
```

---

## 第六部分: MVP SFT 输出设计

### 约束

MVP 阶段 (Layer 2A Classification SFT):

| 允许 | 禁止 |
|------|------|
| ✅ 二元分类输出: `"low risk"` / `"high risk"` | ❌ 结构化 JSON 输出 |
| ✅ 单一 token 响应 | ❌ 多句自然语言推理 |
| ✅ system + user + assistant 三段式 | ❌ `key_factors` / `reasoning` 字段 |

### MVP 输出示例

```
assistant: "low risk"
assistant: "high risk"
```

### 未来 Layer 2B 扩展 (不在 MVP)

```
assistant: {
  "risk_level": "low risk",
  "key_factors": [
    "Positive credit history: all credits paid back duly",
    "Stable employment: 1-4 years at current job"
  ],
  "reasoning": "The customer demonstrates responsible credit behavior..."
}
```

> Layer 2B 需要额外数据建设 (teacher model 蒸馏或人工标注)，参见 `docs/RiskDataset_Schema.md` Layer 2B 部分。

---

## 第七部分: 数据划分设计

### 划分参数

**文件依据**: `data/credit_scoring/German/prepocess.py:11,139`

| 参数 | 值 | 来源 |
|------|-----|------|
| train 比例 | 0.7 (70%) | CALM `prepocess.py` |
| dev 比例 | 0.1 (10%) | CALM `prepocess.py` |
| test 比例 | 0.2 (20%) | CALM `prepocess.py` |
| 随机种子 | 10086 | CALM `prepocess.py:137` |
| 划分方式 | 随机不放回抽样 | `random.sample()` |

### 预期样本分布

| Split | 样本数 | 预期 good (risk_label=0) | 预期 bad (risk_label=1) |
|-------|:-----:|:------------------------:|:------------------------:|
| train | 700 | ~490 | ~210 |
| valid | 100 | ~70 | ~30 |
| test | 200 | ~140 | ~60 |
| **总计** | **1,000** | **700** | **300** |

> 注: 实际分布取决于随机种子 10086。建议在转换后输出实际分布并与预期比较。

### 与 CALM 原始分割保持一致

CALM `prepocess.py` 使用 `random.sample()` 顺序抽样: 先抽 train_ind → 从剩余中抽 dev_ind → 剩余为 test。Converter 应遵循相同逻辑以确保可复现。

### Manifest 设计

```json
{
  "dataset": "German",
  "task_type": "credit_scoring",
  "target_type": "binary_risk",
  "created_at": "2026-07-21T12:00:00Z",
  "source_file": "data/credit_scoring/German/german.data",
  "source_preprocess": "data/credit_scoring/German/prepocess.py",
  "random_seed": 10086,
  "total_samples": 1000,
  "feature_count": 20,
  "label_distribution": {
    "overall": {"good": 700, "bad": 300},
    "train": {"good": 490, "bad": 210},
    "valid": {"good": 70, "bad": 30},
    "test": {"good": 140, "bad": 60}
  },
  "splits": {
    "train": {"count": 700, "sample_ids": ["german_0", "german_1", "..."]},
    "valid": {"count": 100, "sample_ids": ["..."]},
    "test": {"count": 200, "sample_ids": ["..."]}
  },
  "files_generated": [
    "normalized/german/train.jsonl",
    "normalized/german/valid.jsonl",
    "normalized/german/test.jsonl",
    "normalized/german/manifest.json",
    "sft/german/train.jsonl",
    "sft/german/valid.jsonl",
    "sft/german/test.jsonl",
    "sft/german/manifest.json"
  ]
}
```

---

## 第八部分: 数据验证设计

### 验证规则

#### V1: 样本数量检查

```yaml
rule: row_count
expected:
  total: 1000
  train: 700
  valid: 100
  test: 200
check: "训练集 + 验证集 + 测试集 行数之和 == 1000"
```

#### V2: 标签检查

```yaml
rule: risk_label_domain
expected: "risk_label ∈ {0, 1}"
check: "所有样本的 risk_label 为 0 或 1"
---
rule: label_distribution
expected:
  risk_label_0: "~700 (good → low risk)"
  risk_label_1: "~300 (bad → high risk)"
check: "每个 split 中 0/1 比例与总体分布一致 (允许抽样误差)"
```

#### V3: 字段完整性

```yaml
rule: feature_count
expected: 20
check: "每个样本 features 字段包含 20 个 key"
---
rule: required_fields
expected:
  normalized:
    - sample_id
    - dataset
    - split
    - task_type
    - target_type
    - risk_label
    - original_label
    - calm_gold
    - text
    - prompt_format
    - feature_count
    - features
  sft:
    - messages (3 elements: system, user, assistant)
    - metadata (含 sample_id, dataset, split, task_type, risk_label)
check: "所有必填字段非空"
```

#### V4: Split 无交叉

```yaml
rule: no_overlap
check: "train ∩ valid == ∅, train ∩ test == ∅, valid ∩ test == ∅"
implementation: "对 sample_id 去重，确保每个 id 只出现在一个 split 中"
```

#### V5: ChatML 格式合法

```yaml
rule: messages_structure
checks:
  - "messages 数组长度为 3"
  - "messages[0].role == 'system'"
  - "messages[1].role == 'user'"
  - "messages[2].role == 'assistant'"
  - "messages[2].content ∈ {'low risk', 'high risk'}"
  - "所有 content 非空字符串"
```

#### V6: Metadata 可追踪

```yaml
rule: metadata_traceability
checks:
  - "metadata.sample_id 与 normalized_layer 中的 sample_id 一致"
  - "metadata.original_label.value 与原始 german.data 中标签一致"
  - "metadata.risk_label == GermanAdapter.map_risk_label(metadata.original_label.value)"
```

#### V7: 特征值映射正确性

```yaml
rule: feature_mapping
checks:
  - "分类特征的值不包含原始 raw_code (如 A11) — 必须已映射为描述文本"
  - "数值特征的值在合理范围内 (如 age_years 在 18-100 之间)"
```

### 验证执行顺序

```
V1 (数量) → V2 (标签) → V3 (字段完整性) → V4 (无交叉) → V7 (特征映射) → V5 (ChatML) → V6 (可追踪)
```

---

## 第九部分: 输出文件设计

### 文件结构

```
data/
├── raw/
│   └── german/
│       ├── german.data              # 原始文件 (只读，不修改)
│       ├── german.data-numeric      # 数值编码版 (只读)
│       └── german.doc               # 数据描述 (只读)
│
├── normalized/
│   └── german/
│       ├── train.jsonl              # Layer 1: 训练集 700 条
│       ├── valid.jsonl              # Layer 1: 验证集 100 条
│       ├── test.jsonl               # Layer 1: 测试集 200 条
│       └── manifest.json            # 元信息: 标签分布、生成时间、source 追溯
│
├── sft/
│   └── german/
│       ├── train.jsonl              # Layer 2A: ChatML 格式训练集
│       ├── valid.jsonl              # Layer 2A: ChatML 格式验证集
│       ├── test.jsonl               # Layer 2A: ChatML 格式测试集
│       └── manifest.json            # 元信息
│
└── evaluation/
    └── german/
        ├── predictions.jsonl         # Layer 4: 推理结果 (未来)
        └── metrics.json              # Layer 4: 评估指标 (未来)
```

### 文件作用说明

| 文件 | 层 | 消费者 | 作用 |
|------|:--:|--------|------|
| `raw/german/german.data` | L0 | Converter | 输入: 原始数据，只读 |
| `normalized/german/train.jsonl` | L1 | Converter → SFT builder | 中间格式: 统一 schema + 原始标签保留 |
| `normalized/german/manifest.json` | L1 | 审计 / 数据版本管理 | 记录: 生成参数、标签分布、样本ID列表 |
| `sft/german/train.jsonl` | L2A | ms-swift `--dataset` | 训练输入: ChatML 格式 |
| `sft/german/manifest.json` | L2A | 训练日志 | 记录: SFT 训练数据版本 |
| `evaluation/german/predictions.jsonl` | L4 | 评估脚本 | 记录: 多 checkpoint 推理结果 |
| `evaluation/german/metrics.json` | L4 | 报告生成 | 聚合指标: accuracy, F1, fairness |

### JSONL 格式

每行一个独立 JSON 对象，不含外层数组括号。一行一样本。

```
normalized/german/train.jsonl:
{"sample_id":"german_0","dataset":"German","split":"train",...}
{"sample_id":"german_3","dataset":"German","split":"train",...}
...
```

---

## 第十部分: 未来扩展接口

### 设计原则

German Converter 不应包含任何 German-specific 硬编码逻辑。通过**抽象基类 + 数据集适配器**模式实现:

### 概念架构

```
┌─────────────────────────────────────────┐
│           BaseConverter (抽象)           │
│                                         │
│  + load_raw(file) → raw_records         │
│  + split(raw_records, seed, ratio)      │
│  + to_normalized(raw_record) → norm     │
│  + to_sft(norm_record) → sft            │
│  + validate(norm_records, sft_records)  │
│  + save(files)                          │
│                                         │
│  抽象方法 (子类实现):                     │
│  - parse_line(line) → dict              │
│  - extract_label(parsed) → original_label│
│  - build_features(parsed) → features    │
│  - get_system_prompt() → string         │
│  - get_task_type() → string             │
│  - get_target_type() → string           │
│  - map_risk_label(original_label) → int │
└────────────────┬────────────────────────┘
                 │ 继承
    ┌────────────┼────────────┬──────────────┐
    ▼            ▼            ▼              ▼
┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐
│German  │ │Australian│ │Taiwan  │ │TravelInsurance│
│Adapter │ │Adapter   │ │Adapter │ │Adapter       │
└────────┘ └──────────┘ └────────┘ └──────────────┘
```

### BaseConverter 职责

| 方法 | 职责 | 是否需要子类覆写 |
|------|------|:--------------:|
| `load_raw(path)` | 读取原始文件，按行/按分隔符解析 | ❌ 通用 |
| `split(records, seed, ratio)` | 随机分割 train/valid/test | ❌ 通用 |
| `to_normalized(record)` | 构建 Layer 1 JSON | ❌ 通用 (调用子类的抽象方法) |
| `to_sft(norm_record)` | 构建 Layer 2A ChatML JSON | ❌ 通用 (调用子类 `get_system_prompt()`) |
| `validate(norm, sft)` | 执行验证规则 V1-V7 | ❌ 通用 |
| `save(records, output_dir)` | 写 JSONL + manifest | ❌ 通用 |

### Dataset Adapter 需实现的抽象方法

| 方法 | German 实现 | Australian 实现 | Taiwan 实现 |
|------|------------|----------------|------------|
| `parse_line(line)` | `line.strip().split(' ')` | `line.strip().split(' ')` | `csv.reader` |
| `extract_label(parsed)` | `parsed[-1]` (最后一列) | `parsed[-1]` (最后一列) | `parsed[0]` (第一列!) |
| `build_features(parsed)` | dict 映射 + raw_code → desc | raw feature name → value | raw feature name → value |
| `get_system_prompt()` | credit_scoring prompt | credit_scoring prompt | bankruptcy_prediction prompt |
| `get_task_type()` | `"credit_scoring"` | `"credit_scoring"` | `"bankruptcy_prediction"` |
| `get_target_type()` | `"binary_risk"` | `"binary_risk"` | `"binary_risk"` |
| `map_risk_label(orig)` | 显式 mapping: `{1:0, 2:1}` | 显式 mapping: `{1:0, 0:1}` | 显式 mapping: `{0:0, 1:1}` |
| `get_feature_count()` | `20` | `14` | `95` |
| `get_prompt_format()` | `"description"` | `"table"` | `"table"` |

### Adapter 变体摘要

| 数据集 | 特化逻辑 |
|--------|---------|
| German | 13 个分类特征需要 raw_code → 描述映射; 标签列在最后一列 |
| Australian | 所有特征已匿名化，直接使用 A1-A14; 标签 `1=good, 0=bad` |
| Lending Club | CSV 读取，需从 ~150 列选 21 列; 标签字符串 `Fully Paid`/`Charged Off` |
| Credit Card Fraud | 大文件降采样; 删除 Time 列; 29 个 PCA 匿名化特征 |
| ccFraud | 大文件降采样; gender 特殊映射 |
| Polish | 5 个 ARFF 合并; 64 个数值特征; 标签是字符串 `'0'`/`'1'` |
| Taiwan | 标签在第一列 (col 0); 95 个特征 |
| Travel Insurance | Duration/Age 截断; 删除 Gender 列; `target_type = claim_event` |
| PortoSeguro | 大文件降采样; gold 标签反向; `target_type = claim_event` |
| Customs | 已预分割 train/valid/test; `target_type = customs_tier`; 三层风险等级 |

### 扩展新数据集的步骤

1. 创建 `{DatasetName}Adapter` 类，继承 `BaseConverter`
2. 实现上述 9 个抽象方法
3. 注册到 Converter Registry
4. 运行 `validate()` 确保通过所有验证规则
5. 运行 `save()` 生成 normalized + sft 文件

---

## 附录 A: 转换流程图

```
german.data (1000 rows, space-separated)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. parse_line()                     │
│    split by space → 21-element list │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. extract_label()                  │
│    parsed[20] → original_label      │
│    1 → {"value":1, "meaning":"good"}│
│    2 → {"value":2, "meaning":"bad"} │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. build_features()                 │
│    for i in 0..19:                  │
│      if i in dict_keys:             │
│        value = dict[i][raw_code]    │
│      else:                          │
│        value = str(raw_value)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. map_risk_label()                 │
│    GermanAdapter: {1:0, 2:1}        │
│    original=1 → risk_label=0        │
│    original=2 → risk_label=1        │
│    calm_gold = risk_label           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. to_normalized()                  │
│    assemble Layer 1 JSON record     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. split()                          │
│    seed=10086, 7:1:2               │
│    assign split = train/valid/test  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 7. to_sft()                         │
│    system = get_system_prompt()     │
│    user = norm_record.text          │
│    assistant = "low risk"/"high"    │
│    metadata = embed norm fields     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 8. validate()                       │
│    V1-V7 checks pass                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 9. save()                           │
│    normalized/german/{split}.jsonl  │
│    sft/german/{split}.jsonl         │
│    manifest.json                    │
└─────────────────────────────────────┘
```

---

## 附录 B: 数据血缘

```
german.data (1000 rows)
    │
    │ prepocess.py (CALM 原有逻辑, 不修改)
    │ 产生: CALM raw {id, query, answer, choices, gold, text}
    ▼
Layer 0: German.parquet
    │
    │ GermanAdapter.parse_line() + build_features() + map_risk_label()
    │ 产生: 统一 normalized JSON
    ▼
Layer 1: normalized/german/{train,valid,test}.jsonl  ← 新增
    │
    │ BaseConverter.to_sft()
    │ 产生: ChatML messages + metadata
    ▼
Layer 2A: sft/german/{train,valid,test}.jsonl         ← 新增
    │
    │ ms-swift swift sft --dataset sft/german/train.jsonl
    ▼
Qwen2.5-7B-Instruct (fine-tuned)                       ← 训练产物
```

---

*本设计文档基于 `docs/CALM_Data_Schema.md` 和 `docs/RiskDataset_Schema.md`。所有字段映射来自 `data/credit_scoring/German/prepocess.py` 实际代码。未进行任何代码修改或数据生成。*
