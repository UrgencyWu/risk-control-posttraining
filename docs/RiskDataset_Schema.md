# RiskDataset Schema: 金融风控大模型数据协议

> **版本**: v1.0
> **设计日期**: 2026-07-21
> **设计基础**: `docs/CALM_Data_Schema.md` (CALM 数据资产审计)
> **目标框架**: Qwen2.5 + ms-swift + DPO/SimPO/Risk-DPO
> **设计原则**:
> - 逐层递进，每层可独立使用
> - 保留原始标签，保证可追溯
> - 不强制所有任务使用统一标签 — 允许 task_type + target_type 组合
> - 为未来 preference optimization 预留扩展点

---

## 目录

1. [Layer 0: Raw Layer](#layer-0-raw-layer)
2. [Layer 1: Normalized Layer](#layer-1-normalized-layer)
3. [Layer 2A: Classification SFT (MVP)](#layer-2a-classification-sft-mvp)
4. [Layer 2B: Explanation SFT (Future)](#layer-2b-explanation-sft-future)
5. [Layer 3: Preference Optimization Layer](#layer-3-preference-optimization-layer)
6. [Layer 4: Evaluation Layer](#layer-4-evaluation-layer)
7. [附录 A: 全数据集 task_type / target_type 映射](#附录-a-全数据集-task_type--target_type-映射)
8. [附录 B: 各层字段溯源矩阵](#附录-b-各层字段溯源矩阵)
9. [附录 C: 标签语义设计理由](#附录-c-标签语义设计理由)

---

## Layer 0: Raw Layer

### 定位

原始 CALM 数据格式的**完整保留层**。不做任何变换，仅用于审计溯源和复现原论文结果。

### Schema

```json
{
  "id": 0,
  "query": "<full prompt + text + Answer:>",
  "answer": "good",
  "choices": ["good", "bad"],
  "gold": 0,
  "text": "<features described in natural language>"
}
```

### 字段定义

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `id` | int | `prepocess.py` 生成 | 数据集内序号，从 0 开始 |
| `query` | string | `prepocess.py` `process()` 拼接 | 完整 LLM 输入: prompt 模板 + text 描述 + `\nAnswer:` |
| `answer` | string | 标签映射 | 文本标签，值因数据集而异 (good/bad, yes/no, Fully Paid/Charged Off) |
| `choices` | [string, string] | 硬编码 | 两个可选标签，顺序因数据集而异 |
| `gold` | int (0/1) | `原始标签 → 0/1` 映射 | 二值化标签，但 0/1 含义在不同任务间不统一 |
| `text` | string | 特征值 → 自然语言 | 不含 prompt 模板的纯特征描述 |

### 约束

- ⚠️ `gold` 的语义不一致: German 中 `gold=0` = good (低风险)；Travel Insurance 中 `gold=0` = Yes (已理赔，高风险事件)。参见 `docs/CALM_Data_Schema.md` 第四部分。
- ⚠️ `query` 混合了 system prompt + user text + output indicator，不适合直接作为 ChatML 输入。
- ✅ 此层**不做任何修改**，是后续所有层的唯一数据来源。

### 数据来源

| 数据集 | 原始文件 | preprocess 脚本 | 生成格式 |
|--------|---------|----------------|---------|
| German | `german.data` | `data/credit_scoring/German/prepocess.py` | `{train,valid,test}.parquet` |
| Australian | `australian.dat` | `data/credit_scoring/Australian/prepocess.py` | 同上 |
| Lending Club | `accepted_2007_to_2018Q4.csv` ❌ | `data/credit_scoring/Lending Club/prepocess.py` | 同上 |
| Credit Card Fraud | `creditcard.csv` ❌ | `data/fraud detection/Credit Card Fraud/prepocess.py` | 同上 |
| ccFraud | `ccFraud.csv` ❌ | `data/fraud detection/ccFraud/prepocess.py` | 同上 |
| Polish | `{1-5}year.arff` | `data/bankruptcy prediction/Polish/prepocess.py` | 同上 |
| Taiwan | `taiwan.csv` | `data/bankruptcy prediction/Taiwan Economic Journal/prepocess.py` | 同上 |
| Travel Insurance | `travel insurance.csv` | `data/insurance claim analysis/Travel Insurance/prepocess.py` | 同上 |
| PortoSeguro | `PortoSeguro.csv` ❌ | `data/insurance claim analysis/PortoSeguro/prepocess.py` | 同上 |
| Customs | `df_syn_{train,valid,test}_eng.csv` | `data/customs/prepocess.py` | 同上 |

---

## Layer 1: Normalized Layer

### 定位

将 10 个异构数据集的标签体系、任务类型、特征描述**统一规范化**。这是后续所有训练格式的**唯一数据源**。

### Schema

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
  "text": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM...",
  "prompt_format": "description",
  "feature_count": 20,
  "features": {
    "checking_account_status": "bigger than 0 DM but smaller than 200 DM",
    "duration_month": 10,
    "credit_history": "existing credits paid back duly till now",
    "purpose": "furniture or equipment",
    "credit_amount": 1521
  },
  "protected_attributes": {
    "gender": "male",
    "age_group": "young",
    "foreign_worker": "yes"
  },
  "calm_gold": 0
}
```

### 字段定义

#### 标识字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `sample_id` | string | `"{dataset}_{id}"` | 全局唯一标识，用于跨数据集追踪 |
| `dataset` | string | 目录名 | 数据集名称: `German`, `Australian`, `LendingClub`, `CreditCardFraud`, `ccFraud`, `Polish`, `Taiwan`, `TravelInsurance`, `PortoSeguro`, `Customs` |
| `split` | enum | preprocess 分割 | `train` / `valid` / `test` |

#### 任务分类字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `task_type` | enum | **新增** | 业务任务大类 (见下方枚举) |
| `target_type` | enum | **新增** | 标签语义类型 (见下方枚举) |

**`task_type` 枚举**:

| 值 | 含义 | 适用数据集 |
|----|------|-----------|
| `credit_scoring` | 信用评分：评估个人/企业的还款能力和意愿 | German, Australian, LendingClub |
| `fraud_detection` | 欺诈检测：识别交易/申报中的欺诈行为 | CreditCardFraud, ccFraud |
| `bankruptcy_prediction` | 破产预测：预测企业是否面临破产 | Polish, Taiwan |
| `claim_analysis` | 理赔分析：评估保单是否会发生理赔 | TravelInsurance, PortoSeguro |
| `customs_declaration` | 海关申报审查：识别申报中的违规行为 | Customs |

**`target_type` 枚举**:

| 值 | 含义 | 适用场景 |
|----|------|---------|
| `binary_risk` | 标准二元风险: 0=低风险(业务期望), 1=高风险(业务规避) | 大多数 binary classification 任务 |
| `claim_event` | 理赔事件: 0=无理赔, 1=有理赔。标注为事件是否发生，而非风险高低 | TravelInsurance, PortoSeguro |
| `customs_tier` | 海关风险分层: 0=正常, 1=一般违规, 2=严重违规。多级风险 | Customs |

#### 标签字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `risk_label` | int | `normalize_risk_label()` 计算 | 业务风险严重度: 0=无风险/低风险, 1=高风险。**只对 binary_risk 和 claim_event 有效**。对 customs_tier，此字段取值为原始 fraud 标签。 |
| `original_label` | object | CALM 原始数据 | 保留原始标签值和含义，用于审计和回退 |
| `original_label.value` | int/string | 原始数据标签列 | 未经任何转换的原始标签值 |
| `original_label.meaning` | string | 人工标注 | 原始标签的语义解释 |
| `original_label.raw_format` | string | 原始列类型 | `"int"` / `"string"` / `"float"` |
| `calm_gold` | int (0/1) | CALM `gold` 字段 | CALM 的二值化标签，**含义随数据集变化**。仅用于与 CALM 原论文结果对比。 |

#### 特征字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `text` | string | CALM `text` | 特征的自然语言描述，直接复用 |
| `prompt_format` | enum | **新增** | `"description"` — 人类可读自然语言描述 / `"table"` — 键值对列表 |
| `feature_count` | int | 代码计算 | 特征数量 |
| `features` | object | **新增** | 结构化特征键值对。key 使用 snake_case 英文特征名，value 为原始值（数值保留原精度，分类保留原文） |

#### 敏感/保护属性

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `protected_attributes` | object? | **新增** | 仅当数据集包含已知的敏感字段时填充。用于偏差分析和公平性约束训练。当前覆盖: German (gender, age, foreign_worker), ccFraud (gender), TravelInsurance (age)。其他数据集为 `null`。 |

### 标签转换规则

#### binary_risk (标准二元风险)

```
原始标签 "good" → risk_label = 0  (低风险)
原始标签 "bad"  → risk_label = 1  (高风险)
```

**适用**: German, Australian, LendingClub, CreditCardFraud, ccFraud, Polish, Taiwan

#### claim_event (理赔事件)

```
原始标签 "No"  → risk_label = 0  (无理赔事件)
原始标签 "Yes" → risk_label = 1  (有理赔事件)
```

**说明**: 理赔分析的任务语义与风险评分不同。Claim=Yes 在保险业务中是"事件发生"，不等于"客户本身高风险"。`risk_label` 在此上下文中表示**事件严重度**而非客户风险评估。

**适用**: TravelInsurance, PortoSeguro

#### customs_tier (海关风险分层)

```
原始标签 0 → risk_label = 0  (正常申报)
原始标签 1 → risk_label = 1  (一般违规)
原始标签 2 → risk_label = 2  (严重违规/critical fraud)
```

**说明**: Customs 的三层风险等级。在 Binary 训练时可合并 1 和 2 为 `risk_label=1`；在 Multiclass 训练时保留 0/1/2 三层。

**适用**: Customs

### Layer 1 文件的物理组织

```
data/normalized/
├── train.jsonl       # 所有数据集的训练集合并
├── valid.jsonl       # 所有数据集的验证集合并
├── test.jsonl        # 所有数据集的测试集合并
└── manifest.json     # 元信息: 各数据集样本数、标签分布、生成时间
```

---

## Layer 2A: Classification SFT (MVP)

### 定位

面向 ms-swift SFT 训练的直接输入格式。**MVP 阶段**仅输出二元分类标签。所有下游训练 (SFT → DPO → Risk-DPO) 的第一个训练阶段。

### Schema

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a financial risk assessment expert. Evaluate the creditworthiness based on the customer's financial profile. Classify the risk level as:\n- low risk: the customer is likely to repay\n- high risk: the customer is likely to default\nRespond with only 'low risk' or 'high risk'."
    },
    {
      "role": "user",
      "content": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. The state of Duration in month is 10. The state of Credit history is existing credits paid back duly till now..."
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

### 字段定义: messages

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `messages[0].role` | `"system"` | **新增** | System prompt |
| `messages[0].content` | string | 见下方 System Prompt 设计 | 任务指令 + 输出格式约束 |
| `messages[1].role` | `"user"` | — | — |
| `messages[1].content` | string | Layer 1 `text` | 特征的自然语言描述 |
| `messages[2].role` | `"assistant"` | — | — |
| `messages[2].content` | string | `"low risk"` / `"high risk"` | 统一二元输出，基于 `risk_label` 映射 |

### 字段定义: metadata

`metadata` 块完整嵌入 Layer 1 的所有标识和标签字段，确保训练样本可以追溯到原始数据。字段含义参见 [Layer 1 字段定义](#字段定义-1)。

### System Prompt 设计

所有 system prompt 遵循统一模板:
1. 角色定位
2. 任务说明
3. 输出类别定义
4. 输出格式约束

#### credit_scoring

```
You are a financial risk assessment expert. Evaluate the creditworthiness
based on the customer's financial profile. Classify the risk level as:
- low risk: the customer is likely to repay
- high risk: the customer is likely to default
Respond with only 'low risk' or 'high risk'.
```

#### fraud_detection

```
You are a financial fraud detection expert. Analyze the transaction or
declaration attributes to identify potential fraud. Classify the risk level as:
- low risk: the transaction/declaration appears normal
- high risk: the transaction/declaration shows fraud indicators
Respond with only 'low risk' or 'high risk'.
```

#### bankruptcy_prediction

```
You are a corporate financial analyst. Evaluate the company's financial
ratios to predict bankruptcy risk. Classify the risk level as:
- low risk: the company is financially stable
- high risk: the company is at risk of bankruptcy
Respond with only 'low risk' or 'high risk'.
```

#### claim_analysis

```
You are an insurance risk analyst. Analyze the policyholder's profile
and insurance attributes to assess claim likelihood. Classify as:
- low risk: unlikely to file a claim
- high risk: likely to file a claim
Respond with only 'low risk' or 'high risk'.
```

#### customs_declaration

```
You are a customs compliance analyst. Review the import declaration
attributes to identify potential violations. Classify the risk level as:
- low risk: the declaration appears compliant
- high risk: the declaration shows violation indicators
Respond with only 'low risk' or 'high risk'.
```

### 输出格式

| `risk_label` | assistant `content` | 含义 |
|:---:|------|------|
| 0 | `"low risk"` | 低风险 / 正常 / 无违约 / 无欺诈 / 无破产 / 无理赔 |
| 1 | `"high risk"` | 高风险 / 违约 / 欺诈 / 破产 / 理赔事件 |
| 2 | `"high risk"` | (仅 Customs) 严重违规 → 合并为 high risk (二元模式) |

### ms-swift 使用

```bash
# 单任务训练
swift sft \
  --model_type qwen2.5 \
  --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset train.jsonl \
  --val_dataset valid.jsonl \
  --output_dir ./output/risk-sft-mvp

# 多任务混合训练 (利用 metadata.task_type 控制采样)
swift sft \
  --dataset train.jsonl \
  --dataset_shuffle true
```

### 物理组织

```
data/sft/
├── train.jsonl       # 训练集 (从 normalized/train.jsonl 转换)
├── valid.jsonl       # 验证集
├── test.jsonl        # 测试集
└── per_dataset/      # 按数据集拆分
    ├── german_train.jsonl
    ├── australian_train.jsonl
    └── ...
```

---

## Layer 2B: Explanation SFT (Future)

### 定位

**未来研究阶段格式，不生成于 CALM ground truth，不属于 MVP。**

面向需要模型输出**可解释风险评估**的场景。需要额外的标注数据或通过 teacher model 蒸馏生成。

### 设计约束

> ⚠️ **重要声明**:
> - Layer 2B 的训练数据**不来自 CALM 原始数据**。CALM 数据只包含二元标签，不含推理过程。
> - Layer 2B 需要额外的数据建设工作: (1) 人工标注推理链; 或 (2) 使用强模型 (GPT-4/Qwen-Max) 生成推理链再蒸馏。
> - MVP 阶段仅需 Layer 2A 完成二元分类 SFT。

### Schema (目标格式)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a financial risk assessment expert. Analyze the customer's financial profile and provide a structured risk assessment. Output in JSON format with the following fields: risk_level ('low risk' or 'high risk'), key_factors (array of 2-4 key risk indicators), and reasoning (brief explanation of your assessment)."
    },
    {
      "role": "user",
      "content": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"risk_level\": \"low risk\",\n  \"key_factors\": [\n    \"Positive credit history: all credits paid back duly\",\n    \"Stable employment: 1-4 years at current job\",\n    \"Property ownership: owns car or other assets\"\n  ],\n  \"reasoning\": \"The customer demonstrates responsible credit behavior with a clean repayment history. Their employment stability and asset ownership further support a low-risk classification, despite having a relatively modest checking account balance.\"\n}"
    }
  ],
  "metadata": {
    "sample_id": "german_0",
    "dataset": "German",
    "task_type": "credit_scoring",
    "risk_label": 0,
    "explanation_source": "gpt4_distilled",
    "generation_date": "2026-08-01"
  }
}
```

### 字段定义 (与 Layer 2A 差异)

| 字段 | 说明 |
|------|------|
| `messages[2].content` | JSON 结构化输出: `risk_level` + `key_factors` + `reasoning` |
| `metadata.explanation_source` | 推理链来源: `human_annotated` / `gpt4_distilled` / `qwen_max_distilled` |
| `metadata.generation_date` | 推理链生成日期，用于版本管理 |

### 输出 JSON Schema

```json
{
  "risk_level": "low risk | high risk",
  "key_factors": ["<factor 1>", "<factor 2>", "..."],
  "reasoning": "<natural language reasoning, 50-200 words>"
}
```

### 与 Layer 2A 的关系

```
Layer 2A (MVP)                    Layer 2B (Future)
──────────────                    ─────────────────
SFT → binary classifier           SFT → explainable classifier
Output: "low risk"                Output: {"risk_level": "low risk",
                                            "key_factors": [...],
                                            "reasoning": "..."}
CALM 数据直接可用                 需要额外数据建设
支持 Risk-DPO                    支持 Explanation Quality Reward
```

---

## Layer 3: Preference Optimization Layer

### 定位

面向 DPO / SimPO / Risk-DPO 训练的**偏好数据格式**。核心是构建 `(prompt, chosen, rejected)` 三元组，并附加足够的元数据以支持风险感知的偏好优化。

### Schema

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a financial risk assessment expert..."
    },
    {
      "role": "user",
      "content": "The state of Status of existing checking account is bigger than 0 DM..."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "low risk"
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "high risk"
    }
  ],
  "metadata": {
    "sample_id": "german_0",
    "dataset": "German",
    "task_type": "credit_scoring",
    "target_type": "binary_risk",
    "risk_label": 0,
    "original_label": {"value": 1, "meaning": "good"},
    "split": "train"
  },
  "error_type": "false_positive",
  "risk_weight": 2.0,
  "source": "model_generated",
  "generation_stage": "sft_epoch3",
  "model": "Qwen2.5-7B-Instruct",
  "generation_timestamp": "2026-08-01T12:00:00Z"
}
```

### 字段定义

#### 核心字段 (标准 DPO)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `prompt` | messages[] | Layer 2A `messages[0:2]` | system + user messages，不含 assistant |
| `chosen` | messages[] | **构建** | 正确的风险判断 (ground truth) |
| `rejected` | messages[] | **构建** | 错误的风险判断 (模型生成或人工构造) |

#### 错误分析字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `error_type` | enum | **新增** | 模型所犯错误的类型，用于 Risk-DPO 的差异化惩罚 |

**`error_type` 枚举**:

| 值 | 含义 | 业务影响 | DPO 惩罚倾向 |
|----|------|---------|-------------|
| `false_positive` | 低风险样本被误判为高风险 | 客户被拒贷/被误标记 | 中等惩罚 — 损失商业机会 |
| `false_negative` | 高风险样本被误判为低风险 | 坏账/欺诈漏过 | **高惩罚** — 造成实际损失 |
| `severity_mismatch` | (仅 multiclass) 风险等级偏差 | Customs 一般违规判为严重违规 | 中等惩罚 |
| `correct` | 预测正确 | 无 | weight=0 (不参与 DPO) |

#### 风险惩罚权重

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `risk_weight` | float | **新增** | 偏好优化的非对称权重。高风险漏判 (false_negative) 的 weight 应大于低风险误判 (false_positive) |

**推荐默认值**:

| error_type | risk_weight | 理由 |
|------------|:-----------:|------|
| `false_negative` | 3.0 | 漏判坏客户造成的损失远大于错判好客户 |
| `false_positive` | 1.0 | 基准权重 |
| `severity_mismatch` | 1.5 | 中间惩罚 |
| `correct` | 0.0 | 不参与偏好优化 |

> 权重可根据业务场景调整。例如: 小额消费贷可能降低 `false_positive` 到 0.5 (容忍更多误拒以换取更多放款)；大额企业贷可能提高 `false_negative` 到 5.0。

#### 来源追踪字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `source` | enum | **新增** | 偏好数据的产生方式 |
| `generation_stage` | string | **新增** | 产生该偏好数据的训练阶段 |
| `model` | string | **新增** | 产生 rejected 响应的模型标识 |
| `generation_timestamp` | ISO8601 | **新增** | 生成时间戳 |

**`source` 枚举**:

| 值 | 含义 | 场景 |
|----|------|------|
| `model_generated` | 模型推理产生的错误输出 | SFT 模型在 test 集上的错误预测 |
| `human_feedback` | 人工标注的偏好 | 风控专家标注的排序偏好 |
| `rule_based` | 基于业务规则构造 | 故意翻转标签或注入已知偏见模式 |
| `adversarial` | 对抗样本构造 | 在特征描述中注入误导信息后的模型输出 |

**`generation_stage` 示例**:

| 值 | 含义 |
|----|------|
| `sft_epoch3` | 第 3 个 epoch 的 SFT checkpoint |
| `dpo_epoch1` | 第 1 轮 DPO 后的 checkpoint |
| `risk_dpo_epoch2` | 第 2 轮 Risk-DPO 后的 checkpoint |
| `base_zero_shot` | Base 模型零样本推理 |

### 用途

**通过 `source` + `generation_stage` + `model` 组合分析 DPO 改进轨迹**:

```
查询: generation_stage=sft_epoch3, error_type=false_negative
→ 分析: SFT 阶段漏判了多少高风险样本

查询: generation_stage=dpo_epoch2, error_type=false_negative
→ 对比: DPO 后漏判率是否下降

查询: source=model_generated vs source=human_feedback
→ 分析: 模型错误模式 vs 人类标注偏好是否一致
```

### 与 Layer 2A 的关系

```
Layer 2A 推理 → 收集预测 → 与 ground truth 对比
    │
    ├── 预测正确 → 不进入 Preference 数据 (或作为 chosen 参考)
    │
    └── 预测错误 → 构建 (prompt, chosen=gt, rejected=pred)
                   ├── 标记 error_type
                   ├── 分配 risk_weight
                   ├── 记录 generation_stage + model
                   └── 写入 Layer 3 数据
```

### 物理组织

```
data/preference/
├── dpo_train.jsonl          # DPO 训练数据
├── risk_dpo_train.jsonl     # Risk-DPO 训练数据 (含 risk_weight)
├── per_stage/               # 按 generation_stage 拆分
│   ├── sft_epoch3_errors.jsonl
│   ├── dpo_epoch1_errors.jsonl
│   └── risk_dpo_epoch2_errors.jsonl
└── manifest.json            # 元信息: 各 stage 错误分布
```

---

## Layer 4: Evaluation Layer

### 定位

**跨训练阶段的统一评估格式**。用于对比 Base / SFT / DPO / Risk-DPO 各阶段模型在相同测试集上的表现。要求记录**概率级别的输出**而非仅离散标签，以支持细粒度的模型能力分析。

### Schema

```json
{
  "sample_id": "german_0",
  "dataset": "German",
  "task_type": "credit_scoring",
  "target_type": "binary_risk",
  "split": "test",
  "ground_truth": {
    "risk_label": 0,
    "original_label": {"value": 1, "meaning": "good"}
  },
  "predictions": [
    {
      "model": "Qwen2.5-7B-Instruct",
      "stage": "sft_epoch3",
      "checkpoint": "output/risk-sft-mvp/checkpoint-1500",
      "prediction": "low risk",
      "predicted_label": 0,
      "risk_score": 0.12,
      "confidence": 0.88,
      "token_logprob": {
        "low risk": -0.13,
        "high risk": -2.05
      },
      "is_correct": true,
      "error_type": null,
      "inference_timestamp": "2026-08-01T12:00:00Z"
    },
    {
      "model": "Qwen2.5-7B-Instruct",
      "stage": "risk_dpo_epoch2",
      "checkpoint": "output/risk-dpo/checkpoint-800",
      "prediction": "low risk",
      "predicted_label": 0,
      "risk_score": 0.08,
      "confidence": 0.94,
      "token_logprob": {
        "low risk": -0.06,
        "high risk": -2.81
      },
      "is_correct": true,
      "error_type": null,
      "inference_timestamp": "2026-08-05T15:00:00Z"
    }
  ],
  "protected_attributes": {
    "gender": "male",
    "age_group": "young"
  }
}
```

### 字段定义

#### Ground Truth

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `ground_truth.risk_label` | int | Layer 1 | 标准化风险标签 |
| `ground_truth.original_label` | object | Layer 1 | 原始标签 (审计用) |

#### Predictions (数组，每个 checkpoint 一个元素)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `model` | string | 模型标识 | 如 `Qwen2.5-7B-Instruct`, `Qwen2.5-1.5B-Instruct` |
| `stage` | string | 训练阶段 | `base_zero_shot`, `sft_epoch{N}`, `dpo_epoch{N}`, `risk_dpo_epoch{N}` |
| `checkpoint` | string | 文件路径/ID | checkpoint 位置，用于复现 |
| `prediction` | string | 模型输出 | `"low risk"` / `"high risk"` |
| `predicted_label` | int | 解析 prediction | 0 或 1 (对 customs_tier: 0/1/2) |
| `risk_score` | float [0, 1] | **新增** (logprob 计算) | 高风险的概率估计。通过 token_logprob 的 softmax 或直接解析生成概率得到。值越大风险越高。 |
| `confidence` | float [0, 1] | **新增** (max logprob) | 模型对预测标签的置信度。`exp(max(token_logprob))` 或模型显式输出。 |
| `token_logprob` | object | **新增** (需要 logprobs) | 各输出 token 的对数概率。LLM 评估的核心差异指标。 |
| `is_correct` | bool | 计算 | `predicted_label == ground_truth.risk_label` |
| `error_type` | enum? | 计算 | 预测错误时为 `false_positive` / `false_negative` / `severity_mismatch`，正确时为 `null` |
| `inference_timestamp` | ISO8601 | — | 推理时间 |

### 为什么需要 `risk_score` + `token_logprob`

传统的 accuracy/F1 评估只看离散标签，无法区分:
- 模型 A: 55% 置信度判"高风险" → 正确
- 模型 B: 95% 置信度判"高风险" → 正确

两者在 accuracy 上相同，但模型 B 的决策质量更高。`risk_score` 和 `token_logprob` 提供概率层面的区分度。

此外，`risk_score` 可以直接作为风控决策的阈值输入 (如 `risk_score > 0.7 → 拒绝放贷`)，而不只是 yes/no。

### Token Logprob 获取方式

**ms-swift 推理时**:
```bash
swift infer \
  --model_type qwen2.5 \
  --ckpt_dir output/risk-sft-mvp/checkpoint-1500 \
  --dataset test.jsonl \
  --logprobs true \
  --top_logprobs 5
```

**Qwen2.5 API 方式**:
```python
# 使用 logprobs 参数
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[...],
    logprobs=True,
    top_logprobs=5
)
```

### 评估指标聚合

基于 Layer 4 数据，计算:

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "stage": "risk_dpo_epoch2",
  "checkpoint": "output/risk-dpo/checkpoint-800",
  "metrics": {
    "overall": {
      "accuracy": 0.87,
      "precision": 0.82,
      "recall": 0.79,
      "f1": 0.80,
      "mean_confidence": 0.85,
      "calibration_error": 0.06
    },
    "per_dataset": {
      "German": {"accuracy": 0.88, "f1": 0.82},
      "Australian": {"accuracy": 0.85, "f1": 0.78}
    },
    "per_error_type": {
      "false_negative_count": 42,
      "false_positive_count": 67,
      "false_negative_rate": 0.14,
      "false_positive_rate": 0.09
    },
    "fairness": {
      "protected_attribute": "gender",
      "equal_opportunity_difference": 0.03,
      "average_odds_difference": -0.02
    }
  }
}
```

### 跨阶段对比

```
                          accuracy  recall(high_risk)  mean_risk_score  false_negative_rate
Base (zero-shot)          0.72      0.58               0.64             0.42
SFT epoch 3               0.85      0.74               0.78             0.26
DPO epoch 2               0.87      0.78               0.81             0.22
Risk-DPO epoch 2          0.86      0.84               0.85             0.16   ← 漏判率最低
```

### 物理组织

```
data/evaluation/
├── test_predictions.jsonl     # 所有 checkpoint 的预测结果 (一个样本多行，每行一个 checkpoint)
├── metrics_summary.json       # 聚合指标
└── per_checkpoint/            # 按 checkpoint 拆分
    ├── sft_epoch3_metrics.json
    ├── dpo_epoch2_metrics.json
    └── risk_dpo_epoch2_metrics.json
```

---

## 附录 A: 全数据集 task_type / target_type 映射

| # | Dataset | task_type | target_type | risk_label 含义 | 标签值域 |
|---|---------|-----------|-------------|----------------|---------|
| 1 | German | `credit_scoring` | `binary_risk` | 0=good(低风险), 1=bad(高风险) | 0/1 |
| 2 | Australian | `credit_scoring` | `binary_risk` | 0=good(低风险), 1=bad(高风险) | 0/1 |
| 3 | LendingClub | `credit_scoring` | `binary_risk` | 0=Fully Paid(低风险), 1=Charged Off(高风险) | 0/1 |
| 4 | CreditCardFraud | `fraud_detection` | `binary_risk` | 0=normal(低风险), 1=fraud(高风险) | 0/1 |
| 5 | ccFraud | `fraud_detection` | `binary_risk` | 0=normal(低风险), 1=fraud(高风险) | 0/1 |
| 6 | Polish | `bankruptcy_prediction` | `binary_risk` | 0=non-bankrupt(低风险), 1=bankrupt(高风险) | 0/1 |
| 7 | Taiwan | `bankruptcy_prediction` | `binary_risk` | 0=non-bankrupt(低风险), 1=bankrupt(高风险) | 0/1 |
| 8 | TravelInsurance | `claim_analysis` | `claim_event` | 0=No claim(无事件), 1=Yes claim(有事件) | 0/1 |
| 9 | PortoSeguro | `claim_analysis` | `claim_event` | 0=no claim(无事件), 1=claim(有事件) | 0/1 |
| 10 | Customs | `customs_declaration` | `customs_tier` | 0=normal, 1=fraud, 2=critical fraud | 0/1/2 |

---

## 附录 B: 各层字段溯源矩阵

| Layer 4 字段 | 来源层 | 来源字段 | 变换 |
|-------------|--------|---------|------|
| `sample_id` | L0 | `id` | `f"{dataset}_{id}"` |
| `dataset` | L0 | 目录名 | snake_case |
| `task_type` | L1 | **新增** | 人工映射 |
| `target_type` | L1 | **新增** | 人工映射 |
| `risk_label` | L1 | `normalize_risk_label(L0.gold)` | 计算 |
| `original_label` | L0 | 原始标签列 | 保留原样 |
| `text` | L0 | `text` | 直接复用 |
| `features` | L1 | **新增** | 从 text 或原始数据解析 |
| `messages[0].content` | L2A | **新增** | 人工编写 system prompt |
| `messages[1].content` | L0 → L2A | `text` | 直接复用 |
| `messages[2].content` | L0 → L2A | `risk_label` → `"low risk"/"high risk"` | 映射 |
| `chosen` | L3 | L2A ground truth | 构建 |
| `rejected` | L3 | 模型错误预测 | 构建 |
| `error_type` | L3 | **新增** | 比较 gt vs pred |
| `risk_weight` | L3 | **新增** | 按 error_type 查表 |
| `source` | L3 | **新增** | 记录生成方式 |
| `generation_stage` | L3 | **新增** | 记录训练阶段 |
| `model` | L3 | **新增** | 模型标识 |
| `risk_score` | L4 | **新增** | softmax(token_logprob) |
| `confidence` | L4 | **新增** | max(token_logprob) |
| `token_logprob` | L4 | **新增** | 推理时 logprobs |

---

## 附录 C: 标签语义设计理由

### 为什么不用 "positive class" 描述 risk_label

在传统 ML 中，"positive class" 通常指被检测的目标事件 (如 fraud=1 是 positive)。但在金融风控中:

- "positive" 在不同上下文中有完全不同含义
  - 信用场景: positive = 好客户 (good)
  - 欺诈场景: positive = 欺诈事件 (bad 事件)
- "positive class" 无法传达**业务风险的严重程度**

**`risk_label` = business risk severity** 明确表达了标签的业务语义: 0 → 业务期望，1 → 业务规避。

### 为什么 TravelInsurance 和 PortoSeguro 不使用 binary_risk

Claim analysis 的任务语义不同于风险评估:

- 信用评分: 评估"这个人会不会违约" → **risk**
- 欺诈检测: 评估"这笔交易是不是欺诈" → **risk**
- 破产预测: 评估"这家公司会不会破产" → **risk**
- 理赔分析: 记录"这个保单是否发生了理赔" → **event**

将理赔事件强制映射为"风险"会混淆以下两个问题:
1. 这个客户风险高吗? (risk assessment)
2. 这个保单发生了理赔吗? (event detection)

通过 `target_type` 区分二者，可以在训练时选择:
- 用 `binary_risk` 数据集训练风险评分卡模型
- 用 `claim_event` 数据集训练理赔预测模型
- 用 `customs_tier` 数据集训练多级违规检测模型

模型中后期可以通过 **multi-task learning** 或 **multi-head architecture** 同时处理多种 target_type。

---

## 附录 D: 各层文件清单

```
data/
├── raw/                              # Layer 0: CALM 原始输出
│   └── (来自 preprocess.py 生成的 .parquet)
│
├── normalized/                       # Layer 1: 统一规范化
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   └── manifest.json
│
├── sft/                              # Layer 2A: SFT 训练
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   ├── per_dataset/
│   │   ├── german_train.jsonl
│   │   └── ...
│   └── manifest.json
│
├── sft_explanation/                  # Layer 2B: Explanation SFT (Future)
│   ├── train.jsonl
│   └── manifest.json
│
├── preference/                       # Layer 3: Preference Optimization
│   ├── dpo_train.jsonl
│   ├── risk_dpo_train.jsonl
│   ├── per_stage/
│   │   ├── sft_epoch3_errors.jsonl
│   │   └── ...
│   └── manifest.json
│
└── evaluation/                       # Layer 4: Evaluation
    ├── test_predictions.jsonl
    ├── metrics_summary.json
    └── per_checkpoint/
        └── ...
```

---

*本协议设计基于 `docs/CALM_Data_Schema.md` 对 10 个 CALM 数据集的完整分析。所有字段设计均考虑了实际数据的约束和金融风控业务的特殊需求。*
