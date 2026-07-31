# CALM 项目审计报告

> **审计目标**: 判断 CALM 是否适合作为"金融风控大模型后训练项目"的基础
> **审计日期**: 2026-07-21
> **审计范围**: `/home/wushaohua/data/risk-control-posttraining` (commit `d4fb59f`)
> **审计原则**: 不修改代码，不安装依赖，不训练模型；所有结论基于实际文件

---

## 1. 仓库结构分析

### 1.1 顶层目录

```
risk-control-posttraining/
├── README.md              # 项目说明 (828 KB)
├── LICENSE                # MIT
├── data.png               # 数据集概览图 (80 KB)
├── prompt.png             # prompt 构造图 (828 KB)
├── res.png                # 实验结果图 (174 KB)
├── bias1.png / bias2.png  # 偏差分析图
├── data/                  # 10 个数据集 + 预处理脚本
├── src/                   # 评估与偏差分析代码
└── .idea/                 # PyCharm 项目配置
```

### 1.2 `data/` 目录： 10 个数据集

| 序号 | 数据集目录 | 任务类型 | 原始来源 | 指令格式 |
|------|-----------|---------|---------|---------|
| 1 | `credit_scoring/German` | 信用评分 | UCI Statlog | 描述型 (Description-based) |
| 2 | `credit_scoring/Australian` | 信用评分 | UCI Statlog | 表格型 (Table-based) |
| 3 | `credit_scoring/Lending Club` | 信用评分 | Kaggle | 描述型 |
| 4 | `fraud detection/Credit Card Fraud` | 欺诈检测 | Kaggle | 表格型 |
| 5 | `fraud detection/ccFraud` | 欺诈检测 | ACM论文 | 描述型 |
| 6 | `bankruptcy prediction/Polish` | 破产预测 | UCI | 表格型 |
| 7 | `bankruptcy prediction/Taiwan Economic Journal` | 破产预测 | Kaggle | 表格型 |
| 8 | `insurance claim analysis/Travel Insurance` | 理赔分析 | Kaggle | 描述型 |
| 9 | `insurance claim analysis/PortoSeguro` | 理赔分析 | Kaggle | 表格型 |
| 10 | `customs` | 海关欺诈 | GitHub (CTGAN生成) | 表格型 |

> **注**: 数据集 #10 (customs) 是额外数据集，不在原论文的 9 个数据集中。论文提到 70K 指令数据，但 customs 数据有 54K 条记录，占比很大。

### 1.3 `src/` 目录： 评估与偏差分析

```
src/
├── __init__.py
├── Precision/                  # 精度评估
│   ├── get_precision-2.py      # 评估脚本（统一读取 JSON 计算指标）
│   ├── bloomz/                  # Bloomz 推理结果
│   ├── chatglm/                 # ChatGLM2 推理结果
│   ├── chatgpt/                 # ChatGPT 推理结果
│   ├── gpt4/                    # GPT-4 推理结果
│   ├── llama/                   # Llama1 推理结果
│   ├── llama2-chat/             # Llama2-chat 推理结果
│   ├── our/                     # CALM 推理结果
│   └── vicuna/                  # Vicuna 推理结果
└── bias/                        # 偏差分析
    ├── __init__.py
    ├── process.py               # 偏差分析预处理工具函数
    ├── bias-german.py           # German 数据偏差分析
    ├── bias-ccfraud.py          # ccFraud 数据偏差分析
    ├── bias-travel.py           # Travel Insurance 偏差分析
    ├── bias_data/               # 预处理的 train/test CSVs
    ├── CALM/                    # CALM 推理结果 (bias评估专用)
    ├── chatgpt/                 # ChatGPT 推理结果 (bias评估专用)
    └── gpt4/                    # GPT-4 推理结果 (bias评估专用)
```

### 1.4 项目定位

**文件依据**: `README.md:29` — "Our project CALM aims to better utilize large language models (LLMs) to study issues related to credit and risk assessment."

CALM 是一个**学术研究项目**，核心产出是：
- 论文: "Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models" (arXiv:2310.00566)
- 指令数据集 (~70K)
- CALM-7B 模型 (基于 Llama2-chat 微调，发布于 HuggingFace)
- Benchmark 评估 + 偏差分析

**关键发现**: 训练代码不在本仓库中，README 指向外部仓库 `https://github.com/Dai-shen/CALM-train`。

---

## 2. 数据集完整性检查

### 2.1 原始数据文件完整性

| 数据集 | 原始数据文件 | 文件存在 | 记录数 | 说明 |
|--------|-------------|---------|--------|------|
| German | `german.data` | ✅ | 1,000 | 完整包含 |
| Australian | `australian.dat` | ✅ | 690 | 完整包含 |
| Lending Club | `accepted_2007_to_2018Q4.csv` | ❌ | 参考 ~1.3M | **缺失**，需从 Kaggle 下载 |
| Credit Card Fraud | `creditcard.csv` | ❌ | 参考 284K | **缺失**，需从 Kaggle 下载 |
| ccFraud | `ccFraud.csv` | ❌ | 参考 ~1M | **缺失**，需外部下载 |
| Polish | `1year.arff` ~ `5year.arff` | ✅ | 43,405 | 5 个 ARFF 文件完整 |
| Taiwan | `taiwan.csv` | ✅ | 6,819 | 完整包含 |
| Travel Insurance | `travel insurance.csv` | ✅ | 63,326 | 完整包含 |
| PortoSeguro | `PortoSeguro.csv` | ❌ | 参考 595K | **缺失**，需从 Kaggle 下载 |
| Customs | `df_syn_train/valid/test_eng.csv` | ✅ | 54,000 | 3 个 CSV 完整包含 |

**结论**: 10 个数据集中有 **4 个缺少原始 CSV 文件** (Lending Club, Credit Card Fraud, ccFraud, PortoSeguro)。这些是大规模数据集，preprocess.py 中的代码通过 `train_test_split(data, test_size=0.96~0.99)` 进行大幅降采样。缺失原因: 文件体积过大，不适合提交到 git。

### 2.2 预处理输出文件

**文件依据**: 所有 `prepocess.py` 中的 `json_save()` 函数输出到 `data/{train|valid|test}.parquet`。

**实际状态**: 仓库中**不存在任何 .parquet 或 .jsonl 预处理输出文件**。preprocess.py 脚本未被实际运行（或运行结果未被提交）。这意味着：
- 指令数据的实际内容需要通过运行 preprocess.py 来生成
- 每个 `prepocess.py` 依赖本地存在原始 CSV 文件

### 2.3 推理结果文件

**状态**: 推理结果 JSON 文件已提交，位于以下目录：

| 评估维度 | 目录 | 文件数 | 覆盖模型 |
|---------|------|--------|---------|
| 精度评估 | `src/Precision/{model}/` | 8 个模型目录 | bloomz, chatglm, chatgpt, gpt4, llama, llama2-chat, our/CALM, vicuna |
| 偏差分析 | `src/bias/{model}/` | 3 个模型目录 | CALM, chatgpt, gpt4 |

推理结果 JSON 格式 (`src/bias/CALM/flare_german_desc_write_out_info.json:2-8`):
```json
{
  "doc_id": 0,
  "prompt_0": "<full prompt text>",
  "logit_0": "bad",
  "truth": "good",
  "acc": "0.0",
  "missing": "0",
  "f1": "...",
  "mcc": "..."
}
```

---

## 3. German Credit 数据 Schema 解析

### 3.1 概述

**文件依据**: `data/credit_scoring/German/german.data`, `prepocess.py`

- 总记录数: **1,000**
- 特征数: **20** (7 数值 + 13 分类)
- 标签 (第 21 列): `1` = good (700 条, 70%), `2` = bad (300 条, 30%)
- 原始格式: 空格分隔，分类变量用字符串编码 (如 `A11`, `A12`)

### 3.2 完整 Schema

**文件依据**: `data/credit_scoring/German/prepocess.py:16-78`

| 索引 | 特征名 | 类型 | 取值/编码 |
|------|--------|------|----------|
| 0 | Status of existing checking account | 分类 | A11: <0 DM, A12: 0~200 DM, A13: >=200 DM / salary, A14: no checking |
| 1 | Duration in month | 数值 | 整数 (如 6, 12, 24, 48) |
| 2 | Credit history | 分类 | A30: no credits/paid, A31: paid duly, A32: paid till now, A33: delay, A34: critical |
| 3 | Purpose | 分类 | A40: car(new), A41: car(used), A42: furniture, A43: radio/TV, A44: appliances, A45: repairs, A46: education, A47: vacation, A48: retraining, A49: business, A410: others |
| 4 | Credit amount | 数值 | 整数 (如 1169, 5951) |
| 5 | Savings account or bonds | 分类 | A61: <100 DM, A62: 100~500, A63: 500~1000, A64: >1000, A65: unknown/none |
| 6 | Present employment since | 分类 | A71: unemployed, A72: <1yr, A73: 1~4yr, A74: 4~7yr, A75: >7yr |
| 7 | Installment rate (% of disposable income) | 数值 | 整数 1-4 |
| 8 | Personal status and sex | 分类 | A91: male divorced/separated, A92: female divorced/separated/married, A93: male single, A94: male married/widowed, A95: female single |
| 9 | Other debtors or guarantors | 分类 | A101: none, A102: co-applicant, A103: guarantor |
| 10 | Present residence since | 数值 | 整数 (年) |
| 11 | Property | 分类 | A121: real estate, A122: building society/life insurance, A123: car/other, A124: unknown/none |
| 12 | Age in years | 数值 | 整数 (如 22, 67) |
| 13 | Other installment plans | 分类 | A141: bank, A142: stores, A143: none |
| 14 | Housing | 分类 | A151: rent, A152: own, A153: for free |
| 15 | Number of existing credits at this bank | 数值 | 整数 |
| 16 | Job | 分类 | A171: unemployed/unskilled, A172: unskilled resident, A173: skilled, A174: management/self-employed |
| 17 | Number of people liable for maintenance | 数值 | 整数 1-2 |
| 18 | Telephone | 分类 | A191: none, A192: registered |
| 19 | foreign worker | 分类 | A201: yes, A202: no |
| 20 | **target** | 标签 | 1=good, 2=bad |

### 3.3 数据示例

**文件依据**: `german.data` 原始前 3 行

```
行1: A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
  → checking: <0 DM, duration=6mo, history=critical, purpose=furniture, amount=1169,
     savings=<100, employment=>7yr, installment=4%, status=male single, debtors=none,
     residence=4yr, property=real estate, age=67, plans=none, housing=own,
     credits=2, job=skilled, maintenance=1, phone=none, foreign=yes, label=good

行2: A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
  → label=bad (2)
```

### 3.4 指令数据转换

**文件依据**: `prepocess.py:85-99`

转换后每条记录为:
```json
{
  "id": 0,
  "query": "Evaluate the creditworthiness of a customer with the following financial profile. ...\nText: 'The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. The state of Duration in month is 10. ...'\nAnswer:",
  "answer": "good",
  "choices": ["good", "bad"],
  "gold": 0,
  "text": "The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. ..."
}
```

其中 `gold = target - 1`，即 `1→0 (good)`, `2→1 (bad)`。

---

## 4. preprocess.py 数据转换逻辑分析

### 4.1 通用处理架构

**覆盖所有 10 个数据集的 `prepocess.py`**，所有脚本遵循相同模式：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ 加载原始数据  │ -> │ 降采样(可选)  │ -> │ 随机分割7:1:2 │ -> │ 指令格式化   │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │ 输出 train/valid/test  │
                                                    │  .parquet + .jsonl     │
                                                    │ + GPT-4 子集(100条)    │
                                                    └───────────────────────┘
```

**固定参数** (`prepocess.py` 各脚本):
- 随机种子: `10086`
- 分割比例: `train:dev:test = 0.7:0.1:0.2`
- GPT-4 测试子集: 通常 100 条 (balanced: 50/50)，部分数据集为 200 条(Customs)或 500 条

### 4.2 大文件降采样策略

对于大规模原始数据集，使用 `sklearn.model_selection.train_test_split` 进行降采样：

| 数据集 | 原始规模 | 降采样参数 | 保留规模 | 代码位置 |
|--------|---------|-----------|---------|---------|
| Lending Club | ~1.3M | `test_size=0.99` | ~13K (1%) | `Lending Club/prepocess.py:98` |
| Credit Card Fraud | 284K | `test_size=0.96` | ~11K (4%) | `Credit Card Fraud/prepocess.py:116` |
| ccFraud | ~1M | `test_size=0.99` | ~10K (1%) | `ccFraud/prepocess.py:122` |
| PortoSeguro | 595K | `test_size=0.98` | ~12K (2%) | `PortoSeguro/prepocess.py:115` |
| Polish | 43K | `test_size=0.8` | ~8.7K (20%) | `Polish/prepocess.py:152` |
| Travel Insurance | 63K | `test_size=0.8` | ~12.7K (20%) | `Travel Insurance/prepocess.py:102` |

所有降采样均使用 `stratify` 参数按标签分层采样，保持类别分布。

### 4.3 两种指令格式

#### 描述型 (Description-based)

**适用数据集**: German, Lending Club, ccFraud, Travel Insurance

**特点**: 将特征转换为自然语言描述，如:

```
"The state of Status of existing checking account is bigger than 0 DM but smaller than 200 DM. The state of Duration in month is 10. ..."
```

**代码依据**: `German/prepocess.py:88-95` — 使用 `dict` 字典将分类编码映射为人类可读文本。

#### 表格型 (Table-based)

**适用数据集**: Australian, Credit Card Fraud, Polish, Taiwan, PortoSeguro, Customs

**特点**: 直接列出特征名和数值，如:

```
"The client has attributes: A1: 1, A2: 22.08, A3: 11.46, A4: 2, ..."
```

**代码依据**: `Australian/prepocess.py:24-28` — 简单拼接 "A{i}: value" 格式。

### 4.4 输出 Schema 统一性

**所有 10 个数据集共享统一的 JSON 输出格式** (`prepocess.py` 各脚本中的 `process_table()` / `process()` 函数):

| 字段 | 类型 | 含义 | 说明 |
|------|------|------|------|
| `id` | int | 样本序号 | 从 0 开始 |
| `query` | string | 完整 prompt + 特征描述 + "Answer:" | 直接输入 LLM |
| `answer` | string | 文本标签 | 如 "good"/"bad", "yes"/"no" |
| `choices` | [string] | 可选标签列表 | 如 ["good", "bad"] |
| `gold` | int | 数字标签 | 0/1, 含义因数据集而异 |
| `text` | string | 纯特征描述 | 不含 prompt 模板 |

**重要**: `gold` 的含义和 `choices` 的标签在不同数据集间不统一：

| 数据集 | positive label | choices[0] | gold=0 含义 |
|--------|---------------|------------|-------------|
| German | good | good | good |
| Australian | good | good | good (target=1) |
| Lending Club | good | good | Fully Paid |
| Credit Card Fraud | no (non-fraud) | no | 正常交易 |
| ccFraud | good | good | 正常交易 |
| Polish | no (non-bankrupt) | no | 正常公司 |
| Taiwan | no (non-bankrupt) | no | 正常公司 |
| Travel Insurance | yes (claimed) | yes | 有索赔 |
| PortoSeguro | no (no claim) | yes | 有索赔 |
| Customs | no (non-fraud) | no | 正常报关 |

### 4.5 特殊数据清洗

- **Travel Insurance** (`prepocess.py:26-37`): Duration > 731 截断为 731, Duration < 1 替换为均值, Age > 99 截断为 99, 删除 Gender 列
- **Lending Club** (`prepocess.py:94-97`): 仅保留 loan_status 为 "Fully Paid" 或 "Charged Off" 的样本
- **各数据集**: custID / Time / id 等无意义列被删除

### 4.6 偏差分析数据

**文件依据**: `German/prepocess.py:126-132`, `ccFraud/prepocess.py:95-100`, `Travel Insurance/prepocess.py:92-97`

部分 preprocess.py 同时生成用于偏差分析的 CSV 文件 (保存在 `src/bias/bias_data/`):

| 文件 | 记录数 | 用途 |
|------|--------|------|
| `german_train.csv` | 700 | German 偏差分析训练集 |
| `german_test.csv` | 200 | German 偏差分析测试集 |
| `ccfraud_train.csv` | ~7K | ccFraud 偏差分析训练集 |
| `ccfraud_test.csv` | ~2K | ccFraud 偏差分析测试集 |
| `TraIn_train.csv` | ~8.9K | Travel Insurance 偏差分析训练集 |
| `TraIn_test.csv` | ~2.5K | Travel Insurance 偏差分析测试集 |

---

## 5. CALM 训练 Pipeline 分析

### 5.1 训练代码位置

**文件依据**: `README.md:103` — "The code of fine-tuning our LLM (CALM-7B) can be found in [CALM-train](https://github.com/Dai-shen/CALM-train)."

**关键发现**: 训练代码**不在本仓库中**。本仓库仅包含：
1. 数据预处理（preprocess.py）
2. 评估与推理结果（src/Precision, src/bias）
3. 偏差分析脚本（src/bias/*.py）

### 5.2 已知 Pipeinle 信息（来自 README）

**文件依据**: `README.md:98-104`

| 参数 | 值 |
|------|-----|
| 基座模型 | **Llama2-chat** (7B) |
| 微调方法 | 指令微调 (Instruction Fine-tuning) |
| 训练数据 | 70K 指令样本 |
| 验证数据 | Lending Club, Polish, PortoSeguro (作为 held-out 任务) |
| 处理类不平衡 | 对 Credit Card Fraud, ccFraud, Taiwan, Travel Insurance 做少数类重采样至 2:1 |
| 框架 | 未知 (需 CALM-train 仓库确认) |

### 5.3 推理格式

**文件依据**: `src/bias/CALM/flare_german_desc_write_out_info.json:4`

推理使用对话模板：
```
Human: \n<prompt>\nText: '<input>'\nAnswer:\n\nAssistant: \n
```

这是典型的 Llama2-chat 对话格式 (`[INST]` 包裹变体)。

### 5.4 评估框架

**文件依据**: `src/Precision/get_precision-2.py`

评估逻辑简单：从 JSON 推理结果中提取 `truth` 和 `logit_0`，计算 precision, f1-score, accuracy:

```python
# get_precision-2.py:19-28
if 'no' in item['logit_0'].lower():
    logit.append('no')
elif 'yes' in item['logit_0'].lower():
    logit.append('yes')
```

这表示评估仅检查推理输出中是否包含 "yes"/"no" 关键词，是一种**弱解析** (weak parsing)。

### 5.5 偏差分析框架

**文件依据**: `src/bias/bias-german.py`, `src/bias/process.py`

使用 **IBM AIF360** 库进行公平性评估：
- 数据偏差指标: Disparate Impact (DI)
- 模型偏差指标: Equal Opportunity Difference (EOD), Average Odds Difference (AOD)
- 保护属性: gender (Personal status and sex), age (Age in years), foreign worker
- 年龄二值化: ≤45 → 0 (privileged), >45 → 1 (unprivileged)
- 性别二值化: male → 0, female → 1

---

## 6. 迁移到 Qwen2.5 + ms-swift 方案对比

### 6.1 可行性评估

| 维度 | 结论 | 风险等级 |
|------|------|---------|
| 数据格式兼容性 | ✅ 可行 | 低 |
| 模型架构兼容性 | ✅ 可行 | 低 |
| 指令模板兼容性 | ⚠️ 需适配 | 中 |
| 评估体系复用 | ✅ 可复用 | 低 |
| 偏差分析复用 | ⚠️ 需适配 | 中 |

### 6.2 详细对比

#### 数据层：指令数据格式

CALM 的数据 JSON 格式可以直接转换为 ms-swift 接受的格式:

**CALM 格式** → **ms-swift chat 格式**:
```json
// CALM 当前格式
{"id": 0, "query": "prompt + text + Answer:", "answer": "good", "choices": ["good", "bad"], "gold": 0}

// ms-swift 需要的格式 (sharegpt / alpaca)
{"conversations": [{"from": "human", "value": "<prompt>\nText: '<text>'"}, {"from": "assistant", "value": "good"}]}
```

**需要做的转换**: 编写一个简单的格式转换脚本，将 `query` 字段中的 prompt 和 Answer: 部分拆分，映射为 ms-swift 的 `conversations` 格式。

#### 模型层：Llama2-chat → Qwen2.5

| 属性 | Llama2-chat (原) | Qwen2.5 (目标) |
|------|-----------------|----------------|
| 参数规模 | 7B | 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B |
| 对话模板 | `[INST]...[/INST]` | `<|im_start|>...<|im_end|>` |
| 上下文长度 | 4096 | 32768+ (视版本) |
| 中文能力 | 弱 | 强 |
| 金融领域 | 未专门优化 | Qwen2.5 通用能力更强 |

**优势**: Qwen2.5 上下文更长，可以处理特征数多的数据集 (如 Taiwan 95 特征, PortoSeguro 57 特征) 而不截断。

#### 模板层：格式适配

**关键问题**: 10 个数据集的 prompt 模板不统一，需要统一管理。

ms-swift 支持 `--template` 参数自动拼接对话模板，Qwen2.5 使用 `qwen` 模板 (chatml 格式)。需要：
1. 移除 CALM prompt 中的 `\nAnswer:` 后缀 (由 ms-swift 的 loss 计算自动处理)
2. 确保 `choices` 信息传递到 prompt 中 (或自定义 system prompt)

#### 训练框架层

| 特性 | CALM-train (推测) | ms-swift |
|------|------------------|----------|
| LoRA 支持 | 未知 | ✅ 内置 |
| 多模型适配 | 固定 Llama2 | ✅ 200+ 模型 |
| 数据格式 | 自定义 JSON | ShareGPT / Alpaca / 自定义 |
| 评估集成 | 独立脚本 | 内置 eval |
| 部署 | 未知 | vLLM / SGLang 集成 |
| 类不平衡处理 | 手工重采样 | 需自定义或数据预处理 |

### 6.3 迁移差距清单

| 差距项 | 当前状态 | 需要的行动 | 优先级 |
|--------|---------|-----------|--------|
| 大文件数据缺失 | 4/10 缺原始 CSV | 下载 Lending Club, CreditCard, ccFraud, PortoSeguro | P0 |
| 格式转换脚本 | 无 | 编写 CALM JSON → ms-swift ShareGPT 转换器 | P0 |
| Chat Template 适配 | Llama2 格式 | 改为 Qwen2.5 chatml `<|im_start|>` 模板 | P0 |
| 指令模板统一 | 10 个不同 prompt | 制定统一的 system prompt + task-specific user prompt 规范 | P1 |
| 标签 mapping 统一 | 各数据集 gold 含义不同 | 统一标签空间 (如 0=negative, 1=positive) | P1 |
| 训练脚本 | 外部仓库 | 编写 ms-swift 微调脚本 | P1 |
| 类不平衡策略 | 手工 2:1 重采样 | 选项: (1) 保持原策略 (2) 使用 weighted loss (3) Focal Loss | P2 |
| 偏差分析迁移 | 基于 AIF360 | 保持 AIF360 或改用公平性约束训练 (ms-swift 不原生支持) | P2 |
| CALM 训练细节 | 未知 | 需查阅 CALM-train 仓库确认训练超参数 | P0 |
| 验证集策略 | held-out 任务 | 需确认是否保持原验证策略 (Lending Club/Polish/PortoSeguro 留作测试) | P1 |

### 6.4 推荐迁移路径

```
阶段1: 数据准备
  ├── 下载缺失原始数据 (4 个 CSV)
  ├── 运行所有 preprocess.py 生成 .parquet 指令数据
  ├── 编写转换脚本: parquet → ms-swift ShareGPT JSONL
  └── 统一标签映射和验证

阶段2: 基线复现
  ├── 查阅 CALM-train 确认原始训练配置
  ├── 用 Qwen2.5-7B + ms-swift 复现 CALM 训练
  ├── 用原评估 JSON 验证结果
  └── 对比 CALM-7B 基准分数

阶段3: 优化迭代
  ├── 试验不同 Qwen2.5 规格 (1.5B/3B/7B)
  ├── 优化 prompt 模板
  ├── LoRA vs Full fine-tuning
  └── 引入 fairness-aware training
```

---

## 7. MVP 实施计划

### 7.1 目标

基于 CALM 数据资产，用 Qwen2.5-7B + ms-swift 微调一个金融风控指令模型，并在至少 3 个数据集上复现原 CALM 的评估指标。

### 7.2 阶段划分

#### 阶段 0: 前置准备 (1-2 天)

| 任务 | 描述 | 输入 | 输出 |
|------|------|------|------|
| 0.1 | 下载 4 个缺失原始数据 | Kaggle/外部源 | Lending Club, Credit Card Fraud, ccFraud, PortoSeguro CSV |
| 0.2 | 查阅 CALM-train | `github.com/Dai-shen/CALM-train` | 训练超参数、框架选型 |
| 0.3 | 运行 preprocess.py | 10 个数据集原始文件 | 70K train/valid/test parquet 文件 |

#### 阶段 1: 数据工程 (2-3 天)

| 任务 | 描述 | 输出 |
|------|------|------|
| 1.1 | 编写 `calm_to_swift.py` 转换脚本 | CALM parquet → ms-swift ShareGPT JSONL |
| 1.2 | 统一标签空间 | 所有数据集 gold → {0: negative_class, 1: positive_class} |
| 1.3 | 构建 Qwen2.5 chatml 格式 prompt | system + user + assistant 三段式 |
| 1.4 | 数据质量检查 | 长度分布、标签均衡、截断检查 |
| 1.5 | 生成 train.jsonl / val.jsonl | ms-swift 可用的最终训练文件 |

#### 阶段 2: 基线训练 (2-3 天)

| 任务 | 描述 | 命令示例 |
|------|------|---------|
| 2.1 | 安装 ms-swift + 环境 | `pip install ms-swift` |
| 2.2 | LoRA 微调 Qwen2.5-7B | `swift sft --model qwen/Qwen2.5-7B-Instruct --dataset train.jsonl --lora_rank 64` |
| 2.3 | 监控训练 | TensorBoard 观察 loss 曲线 |
| 2.4 | 推理 + 评估 | `swift infer` + 适配 CALM 评估脚本 |

#### 阶段 3: 评估验证 (1-2 天)

| 任务 | 描述 |
|------|------|
| 3.1 | 在 test 集上运行推理 |
| 3.2 | 用 CALM 原评估脚本计算 precision/f1/accuracy |
| 3.3 | 与 CALM-7B 报告指标对比 |
| 3.4 | 偏差分析 (German/ccFraud/Travel Insurance) |

#### 阶段 4: 交付 (1 天)

| 任务 | 描述 |
|------|------|
| 4.1 | 编写训练报告 |
| 4.2 | 输出最终模型 (LoRA adapter) |
| 4.3 | 文档化完整流程 |

### 7.3 预估时间线

```
Week 1: 阶段0 (前置) + 阶段1 (数据)
Week 2: 阶段2 (训练) + 阶段3 (评估)
Week 3: 阶段4 (交付) + buffer
```

**总计: 2-3 周** (取决于 GPU 资源和 CALM-train 仓库复杂度)

### 7.4 关键风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CALM-train 仓库不可用/文档不全 | 无法获知训练细节 | 基于论文和 README 推断，参考同类 Llama2 微调设置 |
| 大规模原始数据下载失败 | 无法复现完整数据集 | 优先使用已有数据 (German, Australian, Polish, Taiwan, Travel Insurance, Customs) 的 6 个数据集 |
| Qwen2.5 对话模板与 prompt 不兼容 | 训练质量下降 | 充分做 prompt 消融实验 |
| ms-swift 框架不熟悉 | 开发效率降低 | 先阅读 ms-swift 文档，参考社区示例 |

---

## 8. 总体评估与建议

### 8.1 适合性判断

**CALM 适合作为"金融风控大模型后训练项目"的基础**，理由：

**优势**:
1. ✅ **高质量领域数据**: 10 个覆盖 4 类金融风控任务的结构化数据集，提供 70K 级别的指令微调样本
2. ✅ **成熟的指令模板**: 描述型 + 表格型两种格式，可直接复用
3. ✅ **完整的评估基准**: 已有 8 个基线模型 + 偏差分析的推理结果
4. ✅ **学术验证**: 有 arXiv 论文背书，数据构建流程有清晰论述
5. ✅ **MIT 许可证**: 无商业使用限制
6. ✅ **偏差分析基础设施**: AIF360 集成，可复用的公平性评估代码

**劣势**:
1. ❌ **训练代码缺失**: CALM-train 不在本仓库，需额外查阅
2. ❌ **4/10 原始数据缺失**: 需要外部下载
3. ❌ **框架老旧**: 基于 Llama2-chat (2023)，对话模板、模型能力均过时
4. ❌ **标签不统一**: gold 含义在各数据集间不一致，需要映射
5. ❌ **预处理输出缺失**: parquet/jsonl 文件未生成，需要运行所有 preprocess.py
6. ❌ **弱评估**: 仅基于关键词匹配 (logit_0 中包含 "yes"/"no")，无法处理模型拒答或不确定输出

### 8.2 迁移优势

迁移到 Qwen2.5 + ms-swift 可获得:
- 更强的中文金融理解能力
- 更长的上下文窗口 (处理高维特征数据集)
- 更成熟的训练框架 (LoRA, 多模型, 评估工具链)
- 更活跃的社区支持

### 8.3 建议优先级

1. **P0**: 查阅 CALM-train 仓库，下载 4 个缺失数据
2. **P0**: 编写格式转换脚本，确保数据兼容
3. **P1**: 用 Qwen2.5-7B + LoRA 做第一个基线
4. **P1**: 复现原 CALM 评估结果
5. **P2**: prompt 优化、模型规格探索
6. **P2**: 公平性约束训练

---

## 附录 A: 文件清单

### A.1 所有脚本文件

| 文件路径 | 行数 | 功能 |
|---------|------|------|
| `data/credit_scoring/German/prepocess.py` | 155 | German 数据预处理 (描述型) |
| `data/credit_scoring/Australian/prepocess.py` | 75 | Australian 数据预处理 (表格型) |
| `data/credit_scoring/Lending Club/prepocess.py` | 130 | Lending Club 数据预处理 (描述型) |
| `data/fraud detection/Credit Card Fraud/prepocess.py` | 127 | Credit Card Fraud 数据预处理 (表格型) |
| `data/fraud detection/ccFraud/prepocess.py` | 135 | ccFraud 数据预处理 (描述型) |
| `data/bankruptcy prediction/Polish/prepocess.py` | 164 | Polish 数据预处理 (表格型) |
| `data/bankruptcy prediction/Taiwan Economic Journal/prepocess.py` | 175 | Taiwan 数据预处理 (表格型) |
| `data/insurance claim analysis/Travel Insurance/prepocess.py` | 124 | Travel Insurance 数据预处理 (描述型/表格型) |
| `data/insurance claim analysis/Travel Insurance/process_desc.py` | 92 | Travel Insurance 描述型变体 |
| `data/insurance claim analysis/PortoSeguro/prepocess.py` | 127 | PortoSeguro 数据预处理 (表格型) |
| `data/customs/prepocess.py` | 112 | Customs 数据预处理 (表格型) |
| `src/Precision/get_precision-2.py` | 39 | 精度评估脚本 |
| `src/bias/bias-german.py` | 124 | German 偏差分析 |
| `src/bias/bias-ccfraud.py` | 82 | ccFraud 偏差分析 |
| `src/bias/bias-travel.py` | 80 | Travel Insurance 偏差分析 |
| `src/bias/process.py` | 79 | 偏差分析工具函数 |

### A.2 所有数据文件

| 文件路径 | 大小/行数 | 说明 |
|---------|----------|------|
| `data/credit_scoring/German/german.data` | 1,000 行 | 原始 German 数据 |
| `data/credit_scoring/German/german.data-numeric` | 1,000 行 | 数值编码版 (24 特征) |
| `data/credit_scoring/Australian/australian.dat` | 690 行 | 原始 Australian 数据 |
| `data/bankruptcy prediction/Polish/{1-5}year.arff` | 43,405 行 | Polish 5 年 ARFF 文件 |
| `data/bankruptcy prediction/Taiwan Economic Journal/taiwan.csv` | 6,819 行 | 台湾公司破产数据 |
| `data/insurance claim analysis/Travel Insurance/travel insurance.csv` | 63,326 行 | 旅行保险数据 |
| `data/customs/df_syn_{train,valid,test}_eng.csv` | 54,000 行 | 韩国海关数据 (CTGAN 合成) |
| `src/bias/bias_data/{german,ccfraud,TraIn}_{train,test}.csv` | ~1.1 MB | 偏差分析预分割数据 |

---

## 附录 B: 依赖项

从 preprocess.py 脚本中提取的 Python 依赖:

```
pandas
numpy
json (stdlib)
random (stdlib)
scikit-learn (train_test_split)
arff       (仅 Polish 数据集)
```

偏差分析额外依赖 (`src/bias/`):
```
aif360
```

> **注**: 本项目无 `requirements.txt` 或 `pyproject.toml` 文件。

---

*本报告由 Claude Code 在 `risk-control-posttraining` 仓库文件分析基础上生成，未进行任何代码修改或依赖安装。*
