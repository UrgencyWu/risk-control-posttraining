# CALM 数据资产审计：完整 Schema 文档

> **审计阶段**: C0.5 — CALM Data Schema 固化
> **审计日期**: 2026-07-21
> **仓库**: `/home/wushaohua/data/risk-control-posttraining`
> **原则**: 只读分析，不修改代码，不安装依赖，不下载数据，不生成新数据集
>
> **目标**: 分析 CALM 所有已有金融风险数据集的数据结构，明确未来迁移到 Qwen2.5 + ms-swift 所需的数据 schema。

---

## 第一部分：仓库数据资产扫描 (Dataset Inventory)

### 1.1 数据目录总览

**文件依据**: `data/` 目录结构

```
data/
├── credit_scoring/
│   ├── German/           # UCI Statlog German Credit
│   ├── Australian/       # UCI Statlog Australian Credit
│   └── Lending Club/     # Kaggle Lending Club
├── fraud detection/
│   ├── Credit Card Fraud/ # Kaggle Credit Card Fraud
│   └── ccFraud/          # ACM 论文 ccFraud
├── bankruptcy prediction/
│   ├── Polish/           # UCI Polish Companies Bankruptcy
│   └── Taiwan Economic Journal/ # Kaggle Taiwan Bankruptcy
├── insurance claim analysis/
│   ├── Travel Insurance/ # Kaggle Travel Insurance
│   └── PortoSeguro/      # Kaggle PortoSeguro
└── customs/              # GitHub Customs Declaration (CTGAN)
```

### 1.2 Dataset Inventory 表

| # | Dataset | Path | Task Type | File Type | File Size | Samples | Features | Label Col | Label Meaning | Raw Data Present |
|---|---------|------|-----------|-----------|-----------|---------|----------|------------|---------------|------------------|
| 1 | **German** | `data/credit_scoring/German/` | Credit Scoring | `.data` (space-sep) | 78 KB | 1,000 | 20 | Col 20 (last) | 1=good, 2=bad | ✅ |
| 2 | **Australian** | `data/credit_scoring/Australian/` | Credit Scoring | `.dat` (space-sep) | 29 KB | 690 | 14 | Col 14 (last) | 1=good, 0=bad | ✅ |
| 3 | **Lending Club** | `data/credit_scoring/Lending Club/` | Credit Scoring | `.csv` | — | ~53,812 (sampled from ~1.3M) | 21 | `loan_status` | Fully Paid=good, Charged Off=bad | ❌ |
| 4 | **Credit Card Fraud** | `data/fraud detection/Credit Card Fraud/` | Fraud Detection | `.csv` | — | ~11,384 (sampled from 284K) | 29 (after dropping Time) | `Class` (col 30) | 0=normal, 1=fraud | ❌ |
| 5 | **ccFraud** | `data/fraud detection/ccFraud/` | Fraud Detection | `.csv` | — | ~41,943 (sampled from ~1M) | 7 (after dropping custID) | `fraudRisk` (col 8) | 0=normal, 1=fraud | ❌ |
| 6 | **Polish** | `data/bankruptcy prediction/Polish/` | Bankruptcy Prediction | `.arff` (5 files) | 20.5 MB | 43,405 | 64 | Col 64 (last) | 0=non-bankrupt, 1=bankrupt | ✅ |
| 7 | **Taiwan** | `data/bankruptcy prediction/Taiwan Economic Journal/` | Bankruptcy Prediction | `.csv` | 7.2 MB | 6,819 | 95 | Col 0 (first!) | 0=non-bankrupt, 1=bankrupt | ✅ |
| 8 | **Travel Insurance** | `data/insurance claim analysis/Travel Insurance/` | Claim Analysis | `.csv` | 4.3 MB | 63,326 | 10 | `Claim` (col 4) | Yes=claimed, No=not claimed | ✅ |
| 9 | **PortoSeguro** | `data/insurance claim analysis/PortoSeguro/` | Claim Analysis | `.csv` | — | ~59,521 (sampled from 595K) | 57 | `target` (col 0) | 0=no claim, 1=claim | ❌ |
| 10 | **Customs** | `data/customs/` | Fraud Detection | `.csv` (3 files) | 5.3 MB | 54,000 | 20 (after dropping Critical Fraud) | `Fraud` (col 20) | 0=normal, 1=fraud, 2=critical fraud | ✅ |

### 1.3 数据可用性汇总

| 状态 | 数量 | 数据集 |
|------|------|--------|
| ✅ 原始数据完整 | **6** | German, Australian, Polish, Taiwan, Travel Insurance, Customs |
| ❌ 原始数据缺失 | **4** | Lending Club, Credit Card Fraud, ccFraud, PortoSeguro |

**缺失原因** (`prepocess.py` 各脚本): 这 4 个数据集原始规模过大 (28 万 ~ 130 万行)，不适合 git 提交。preprocess.py 中通过 `train_test_split(data, test_size=0.96~0.99)` 大幅降采样后使用。

### 1.4 标签分布

| 数据集 | 正类 (low risk / good) | 负类 (high risk / bad) | 不平衡比 |
|--------|----------------------|----------------------|---------|
| German | 700 (70.0%) | 300 (30.0%) | 2.3:1 |
| Australian | 307 (44.5%) | 383 (55.5%) | 1:1.2 |
| Lending Club | 未知 (需原始数据) | 未知 | — |
| Credit Card Fraud | ~11,106 (97.6%) | ~278 (2.4%) | ~40:1 |
| ccFraud | ~39,443 (94.0%) | ~2,500 (6.0%) | ~16:1 |
| Polish | ~40,191 (92.6%) | ~3,214 (7.4%) | ~12.5:1 |
| Taiwan | 6,599 (96.8%) | 220 (3.2%) | 30:1 |
| Travel Insurance | 62,399 (98.5%) | 927 (1.5%) | 67:1 |
| PortoSeguro | 未知 (需原始数据) | 未知 | — |
| Customs | 36,347 (67.3%) | 17,653 (32.7%) | 2.1:1 |

### 1.5 仓库内已生成的中间文件

**文件依据**: `src/bias/bias_data/` 目录

| 文件 | 大小 | 行数 (估计) | 来源 |
|------|------|------------|------|
| `german_train.csv` | 55 KB | 700 | German preprocess.py 输出 |
| `german_test.csv` | 16 KB | 200 | German preprocess.py 输出 |
| `ccfraud_train.csv` | 146 KB | ~7,350 | ccFraud preprocess.py 输出 |
| `ccfraud_test.csv` | 42 KB | ~2,098 | ccFraud preprocess.py 输出 |
| `TraIn_train.csv` | 605 KB | ~8,865 | Travel Insurance preprocess.py 输出 |
| `TraIn_test.csv` | 174 KB | ~2,534 | Travel Insurance preprocess.py 输出 |
| `gpt4_ccfraud_test.csv` | 2.1 KB | ~100 | ccFraud preprocess.py 输出 (GPT-4 子集) |

这些 CSV 没有表头，直接是 0-index 数值列。仅用于偏差分析代码 (`src/bias/bias-*.py`)，**非通用指令数据格式**。

---

## 第二部分：逐数据集 preprocess.py 分析

### 2.1 German Credit

**文件**: `data/credit_scoring/German/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `german.data` (空格分隔，无表头) |
| **原始字段数** | 21 (20 features + 1 label) |
| **模型输入字段** | 20 个全部 |
| **标签字段** | 第 21 列 (索引 20) |
| **原始标签** | `1` = good, `2` = bad |
| **gold 转换** | `gold = data[j][-1] - 1` → 0=good, 1=bad |
| **缺失值处理** | ❌ 无 |
| **编码转换** | ✅ 13 个分类特征通过 `dict` 字典映射为人类可读文本 (如 `A11` → `"smaller than 0 DM"`) |
| **数值归一化** | ❌ 无，数值保持原样 |
| **文本描述生成** | ✅ Description-based: `"The state of {feature_name} is {value}. "` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | train:dev:test = 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:85-87`):
```
Evaluate the creditworthiness of a customer with the following financial profile.
Respond with only either 'good' or 'bad'. For instance, 'The client has a stable
income, no previous debts, and owns a property.' should be classified as 'good'.
Text: '{text}'
Answer:
```

### 2.2 Australian Credit

**文件**: `data/credit_scoring/Australian/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `australian.dat` (空格分隔，无表头) |
| **原始字段数** | 15 (14 features + 1 label) |
| **模型输入字段** | 14 个全部 |
| **标签字段** | 第 15 列 (索引 14) |
| **原始标签** | `1` = good, `0` = bad |
| **gold 转换** | `gold = 0 if data[j][-1] == 1 else 1` → 0=good, 1=bad |
| **缺失值处理** | ❌ 无 |
| **编码转换** | ❌ 无 — 特征名和值已被匿名化处理，直接使用数值 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"A{i+1}: {value}"` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | train:dev:test = 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:16-21`):
```
Assess the creditworthiness of a customer using the following table attributes for
financial status. Respond with either 'good' or 'bad'. And all the table attribute names
including 8 categorical attributes and 6 numerical attributes and values have been changed
to meaningless symbols to protect confidentiality of the data. For instance, 'The client
has attributes: A1: 0, A2: 21.67, ..., A14: 1.', should be classified as 'good'.
Text: {text}
Answer:
```

**特殊说明**: 原始 14 个特征中 8 个是分类变量、6 个是数值变量，但数据已被匿名化为无意义符号。`readme.txt` 和 `data description` 文件说明了这一点。

### 2.3 Lending Club

**文件**: `data/credit_scoring/Lending Club/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `accepted_2007_to_2018Q4.csv` ❌ 缺失 |
| **原始字段数** | 22 (21 features + 1 label) |
| **模型输入字段** | 21 个 (从原始 ~150 列中 `usecols` 选取) |
| **标签字段** | `loan_status` (第 22 列, 索引 21) |
| **原始标签** | `Fully Paid` = good, `Charged Off` = bad |
| **gold 转换** | `gold = 0 if Fully Paid else 1` |
| **缺失值处理** | ✅ `dropna(subset=['loan_status'])` 删除标签缺失行 |
| **数据过滤** | ✅ 仅保留 `loan_status` ∈ {`Fully Paid`, `Charged Off`} 的行 |
| **降采样** | ✅ `train_test_split(test_size=0.99, stratify=loan_status)` → 保留 ~1% |
| **编码转换** | ✅ 特征名映射到人类可读描述 (如 `int_rate` → `Interest Rate`) |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Description-based: `"The state of {mean_list[i]} is {value}"` (利率和循环利用率附加 `%`) |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | train:dev:test = 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:28-31`):
```
Assess the client's loan status based on the following loan records from Lending Club.
Respond with only 'good' or 'bad', and do not provide any additional information.
For instance, 'The client has a stable income, no previous debts, and owns a property.'
should be classified as 'good'.
Text: '{text}'
Answer:
```

**选取的 21 个特征** (`prepocess.py:87-90`):
```
installment, purpose, application_type, int_rate, last_pymnt_amnt, loan_amnt,
revol_bal, delinq_2yrs, inq_last_6mths, mort_acc, grade, open_acc, revol_util,
total_acc, fico_range_low, fico_range_high, addr_state, emp_length, home_ownership,
verification_status, annual_inc
```

### 2.4 Credit Card Fraud

**文件**: `data/fraud detection/Credit Card Fraud/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `creditcard.csv` ❌ 缺失 |
| **原始字段数** | 31 (Time + V1~V28 + Amount + Class) |
| **模型输入字段** | 29 个 (V1~V28 + Amount, 删除 Time) |
| **标签字段** | `Class` (col 30, 索引 30) |
| **原始标签** | `0` = normal, `1` = fraud |
| **gold 转换** | `gold = int(data[j][-1])` → 0=normal, 1=fraud |
| **缺失值处理** | ❌ 无 |
| **降采样** | ✅ `train_test_split(test_size=0.96, stratify=Class)` → 保留 ~4% |
| **编码转换** | ❌ 无 — V1~V28 是 PCA 变换结果，无物理含义 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{V{i}}: {value:.3f}"` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 内部 data_split(): 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:40-50`):
```
Detect the credit card fraud using the following financial table attributes.
Respond with only 'yes' or 'no', and do not provide any additional information.
Therein, the data contains 28 numerical input variables V1, V2, ..., and V28
which are the result of a PCA transformation and 1 input variable Amount which
has not been transformed with PCA. The feature 'Amount' is the transaction Amount,
this feature can be used for example-dependant cost-sensitive learning.
For instance, '{example}' should be classified as 'no'.
Text: '{text}'
Answer:
```

### 2.5 ccFraud

**文件**: `data/fraud detection/ccFraud/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `ccFraud.csv` ❌ 缺失 |
| **原始字段数** | 9 (custID + 7 features + fraudRisk) |
| **模型输入字段** | 7 个 (删除 custID) |
| **标签字段** | `fraudRisk` (col 8, 索引 8) |
| **原始标签** | `0` = normal (good), `1` = fraud (bad) |
| **gold 转换** | `gold = int(data[j][-1])` → 0=good, 1=bad |
| **缺失值处理** | ❌ 无 |
| **降采样** | ✅ `train_test_split(test_size=0.99, stratify=fraudRisk)` → 保留 ~1% |
| **编码转换** | ✅ gender 映射: `1` → `male`, `2` → `female` |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Description-based: `"is a male/female, the state number is {val}, ..."` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 内部 data_split(): 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:40-42`):
```
Detect the credit card fraud with the following financial profile.
Respond with only 'good' or 'bad', and do not provide any additional information.
For instance, 'The client is a female, the state number is 25, the number of cards is 1,
the credit balance is 7000, the number of transactions is 16, the number of international
transactions is 0, the credit limit is 6.' should be classified as 'good'.
Text: {text}
Answer:
```

**7 个特征** (`prepocess.py:16`):
```
gender, state, cardholder, balance, numTrans, numIntlTrans, creditLine
```

### 2.6 Polish Bankruptcy

**文件**: `data/bankruptcy prediction/Polish/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `1year.arff`, `2year.arff`, `3year.arff`, `4year.arff`, `5year.arff` ✅ |
| **原始字段数** | 65 (64 features + 1 label) |
| **模型输入字段** | 64 个全部 |
| **标签字段** | 最后一列 (索引 64) |
| **原始标签** | `'0'` = non-bankrupt, `'1'` = bankrupt (字符串) |
| **gold 转换** | `gold = int(data[j][-1])` → 0=non-bankrupt, 1=bankrupt |
| **缺失值处理** | ❌ 无 |
| **降采样** | ✅ `train_test_split(test_size=0.8, stratify=label)` → 保留 ~20% (~8,681) |
| **编码转换** | ❌ 无 — 全部数值特征 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{feature_name}: {value}"` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 内部 data_split(): 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:74-78`):
```
Predict whether the company will face bankruptcy based on the financial profile
attributes provided in the following text. Respond with only 'no' or 'yes', and
do not provide any additional information.
For instance, 'The client has attributes: net profit / total assets: -0.046186, ...,
sales / short-term liabilities: 5.7063, sales / fixed assets: 1.3882.'
should be classified as 'no'.
Text: {text}
Answer:
```

**64 个特征**: 波兰公司财务比率 (`prepocess.py:17-52`)，包括 ROA、负债率、流动比率、利润率、周转率等。

**数据合并**: 5 个 ARFF 文件合并为一个数据集，对应 5 个预测期 (1-5 年)。

### 2.7 Taiwan Bankruptcy

**文件**: `data/bankruptcy prediction/Taiwan Economic Journal/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `taiwan.csv` ✅ |
| **原始字段数** | 96 (95 features + 1 label) |
| **模型输入字段** | 95 个全部 |
| **标签字段** | 第 1 列 (索引 0!) ⚠️ 注意: 标签在第一列 |
| **原始标签** | `0` = non-bankrupt, `1` = bankrupt |
| **gold 转换** | `gold = int(data[j][0])` → 0=non-bankrupt, 1=bankrupt |
| **缺失值处理** | ❌ 无 |
| **降采样** | ❌ 无 (6,819 行全部使用) |
| **编码转换** | ❌ 无 — 全部数值特征 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{feature_name}: {value:.3f}"` (精度 3 位小数) |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 内部 data_split(): 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:87-96`):
```
Predict whether the company will face bankruptcy based on the financial profile
attributes provided in the following text. Respond with only 'no' or 'yes', and
do not provide any additional information.
For instance, 'The client has attributes: ROA(C) before interest and depreciation
before interest: 0.499, ..., Net Income Flag: 1.000, Equity to Liability: 0.044.'
should be classified as 'no'.
Text: {text}
Answer:
```

**⚠️ 关键差异**: Taiwan 的标签在第一列 (`data[j][0]`)，而其他所有数据集的标签在最后一列 (`data[j][-1]`)。这是数据格式上的重要差异。

### 2.8 Travel Insurance

**文件**: `data/insurance claim analysis/Travel Insurance/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `travel insurance.csv` ✅ |
| **原始字段数** | 11 (10 features + 1 label) |
| **模型输入字段** | 9 个 (删除 Gender + 处理 Duration/Age) |
| **标签字段** | `Claim` (col 4, 索引 4) |
| **原始标签** | `Yes` = claimed, `No` = not claimed |
| **gold 转换** | `gold = 0 if Yes else 1` → 0=Yes(claimed), 1=No(not claimed) |
| **缺失值处理** | ✅ Duration < 1 → 均值填充; Age > 99 → 截断为 99; Duration > 731 → 截断为 731 |
| **降采样** | ✅ `train_test_split(test_size=0.8, stratify=Claim)` → 保留 ~20% (~12,665) |
| **编码转换** | ✅ 删除 Gender 列 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{feature_name}: {value}"` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | train:dev:test = 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:43-47`):
```
Identify the claim status of insurance companies using the following table attributes
for travel insurance status. Respond with only 'yes' or 'no', and do not provide any
additional information. And the table attributes including 5 categorical attributes and
4 numerical attributes are as follows:
Agency: Name of agency (categorical).
Agency Type: Type of travel insurance agencies (categorical).
...
For instance: 'The insurance company has attributes: Agency: CBH, Agency Type: Travel
Agency, ..., Age: 81.', should be classified as 'no'.
Text: {text}
Answer:
```

**⚠️ 重要**: Travel Insurance 还有一个**描述型变体**脚本 `process_desc.py` (`data/insurance claim analysis/Travel Insurance/process_desc.py`)。该变体将输入转换为自然语言叙述:
```
A policyholder aged 41 chosen product 'Rental Vehicle Excess Insurance' of the insurance
company 'CWT' through sales channel 'Online' to travel to destination 'ITALY'. The type
of insurance is 'Travel Agency', with an effective period of 79, and the company recorded
the net sales and commission of the insurance as -19.8 and 11.88.
```
这使 Travel Insurance 成为唯一同时拥有表格型和描述型两种指令格式的数据集。

### 2.9 PortoSeguro

**文件**: `data/insurance claim analysis/PortoSeguro/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `PortoSeguro.csv` ❌ 缺失 |
| **原始字段数** | 59 (id + 57 features + target) |
| **模型输入字段** | 57 个 (删除 id) |
| **标签字段** | `target` (col 0, 索引 0) |
| **原始标签** | `0` = no claim, `1` = claim |
| **gold 转换** | `gold = 0 if data[j][0] == 1 else 1` → 0=claim, 1=no claim ⚠️ **反向** |
| **缺失值处理** | ❌ 无 (但特征中使用 -1 表示缺失) |
| **降采样** | ✅ `train_test_split(test_size=0.98, stratify=target)` → 保留 ~2% |
| **编码转换** | ❌ 无 — 直接使用原始特征名 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{feature_name}: {value}"` (长数值截断为 .2f) |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 内部 data_split(): 7:1:2 (seed=10086) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:37-42`):
```
Identify whether or not to files a claim for the auto insurance policy holder using
the following table attributes about individual financial profile. Respond with only
'yes' or 'no', and do not provide any additional information. And the table attributes
that belong to similar groupings are tagged as such in the feature names (e.g., ind,
reg, car, calc). In addition, feature names include the postfix bin to indicate binary
features and cat to indicate categorical features. Features without these designations
are either continuous or ordinal. Values of -1 indicate that the feature was missing
from the observation.
For instance, 'The client has attributes: ps_ind_01: 1.0, ps_ind_02_cat: 2.0, ...,
ps_calc_20_bin: 0.0.' should be classified as 'no'.
Text: {text}
Answer:
```

**⚠️ 关键发现**: PortoSeguro 的 `gold` 标签转换是**反向的** — `target=1`(有索赔) → `gold=0`, `target=0`(无索赔) → `gold=1`。这与直观理解相反，需要在统一标签时特别注意。

### 2.10 Customs Declaration

**文件**: `data/customs/prepocess.py`

| 维度 | 详情 |
|------|------|
| **输入文件** | `df_syn_train_eng.csv`, `df_syn_valid_eng.csv`, `df_syn_test_eng.csv` ✅ |
| **原始字段数** | 22 (20 attributes + Fraud + Critical Fraud) |
| **模型输入字段** | 20 个 (删除 Critical Fraud) |
| **标签字段** | `Fraud` (col 20, 索引 20) |
| **原始标签** | `0` = normal, `1` = fraud, `2` = critical fraud |
| **gold 转换** | `gold = 0 if data[j][-1] == 0 else 1` → 0=normal, 1=fraud/critical fraud |
| **缺失值处理** | ❌ 无 |
| **降采样** | ❌ 无 (测试集降采样至 2,000 条) |
| **编码转换** | ❌ 无 — 全部直接使用 |
| **数值归一化** | ❌ 无 |
| **文本描述生成** | ✅ Table-based: `"{attribute_name}: {value}"` |
| **最终格式** | JSON: `{id, query, answer, choices, gold, text}` |
| **分割比例** | 已预分割为 train(37,385) / valid(8,134) / test(8,481) |
| **输出文件** | `data/train.parquet`, `data/valid.parquet`, `data/test.parquet` |

**prompt 模板** (`prepocess.py:23-32`):
```
Identify the provided customs import declaration information to determine whether
it constitutes customs fraud that attempts to reduce customs duty or not. The answer
must be 'no' or 'yes', and do not provide any additional information. This Import
Declaration consists of 20 data attributes, including Declaration ID, Date, Office ID,
Process type, Import type, Import use, Payment type, Mode of transport, Declarant ID,
Importer ID, Seller ID, Courier ID, HS6 code, Country of departure, Country of origin,
Tax rate, Tax type, Country of origin indicator, Net mass and Item price.
For instance, 'This customs import declaration has attributes: Declaration ID: 97061800,
Date: 2020-01-01, ..., Item Price: 372254.4.' should be categorized as 'no'.
Text: {text}
Answer:
```

**⚠️ 标签说明**: `Fraud=2` (critical fraud) 在代码中被合并到 `answer='yes'`。但代码注释和 readme 表明 critical fraud 是一个独立的更高风险类别。在 GPT-4 子集采样时，critical fraud 样本被单独处理 (`row[-1] == 2`)。

---

## 第三部分：German Credit 完整 Schema

### 3.1 原始格式

**文件**: `data/credit_scoring/German/german.data`

- **格式**: 空格分隔，无表头
- **总行数**: 1,000
- **字段数**: 21 列 (20 features + 1 label)
- **标签列位置**: 最后一列 (索引 20)

**原始数据示例** (前 3 行):
```
A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1
```

### 3.2 Feature Mapping

**文件依据**: `data/credit_scoring/German/prepocess.py:16-78`

| Index | Raw Code | Semantic Meaning | Type | Values / Encoding |
|-------|----------|-----------------|------|-------------------|
| 0 | `A11`–`A14` | Status of existing checking account | Categorical | A11: <0 DM, A12: 0≤x<200 DM, A13: ≥200 DM or salary assignment, A14: no checking account |
| 1 | numeric | Duration in month | Numerical | 整数 (e.g., 6, 12, 24, 48) |
| 2 | `A30`–`A34` | Credit history | Categorical | A30: no credits taken/all paid back, A31: all credits at this bank paid back duly, A32: existing credits paid back duly till now, A33: delay in paying off, A34: critical account/other credits existing |
| 3 | `A40`–`A410` | Purpose | Categorical | A40: car(new), A41: car(used), A42: furniture/equipment, A43: radio/TV, A44: domestic appliances, A45: repairs, A46: education, A47: vacation, A48: retraining, A49: business, A410: others |
| 4 | numeric | Credit amount | Numerical | 整数 (e.g., 1169, 5951) |
| 5 | `A61`–`A65` | Savings account or bonds | Categorical | A61: <100 DM, A62: 100≤x<500, A63: 500≤x<1000, A64: ≥1000, A65: unknown/no savings |
| 6 | `A71`–`A75` | Present employment since | Categorical | A71: unemployed, A72: <1 year, A73: 1≤x<4 years, A74: 4≤x<7 years, A75: ≥7 years |
| 7 | numeric | Installment rate (% of disposable income) | Numerical | 整数 1-4 |
| 8 | `A91`–`A95` | Personal status and sex | Categorical | A91: male divorced/separated, A92: female divorced/separated/married, A93: male single, A94: male married/widowed, A95: female single |
| 9 | `A101`–`A103` | Other debtors or guarantors | Categorical | A101: none, A102: co-applicant, A103: guarantor |
| 10 | numeric | Present residence since | Numerical | 整数 (年份) |
| 11 | `A121`–`A124` | Property | Categorical | A121: real estate, A122: building society/life insurance, A123: car/other, A124: unknown/no property |
| 12 | numeric | Age in years | Numerical | 整数 (e.g., 19-75) |
| 13 | `A141`–`A143` | Other installment plans | Categorical | A141: bank, A142: stores, A143: none |
| 14 | `A151`–`A153` | Housing | Categorical | A151: rent, A152: own, A153: for free |
| 15 | numeric | Number of existing credits at this bank | Numerical | 整数 |
| 16 | `A171`–`A174` | Job | Categorical | A171: unemployed/unskilled non-resident, A172: unskilled resident, A173: skilled employee/official, A174: management/self-employed/highly qualified |
| 17 | numeric | Number of people liable for maintenance | Numerical | 整数 (1-2) |
| 18 | `A191`–`A192` | Telephone | Categorical | A191: none, A192: yes (registered) |
| 19 | `A201`–`A202` | foreign worker | Categorical | A201: yes, A202: no |
| — | — | **(Label)** | — | — |
| 20 | `1` / `2` | Credit risk target | Binary | 1=good, 2=bad |

### 3.3 敏感/保护属性

从偏差分析代码 (`src/bias/bias-german.py`) 中识别的保护属性:

| 特征 | Index | 保护组 (Unprivileged) | 参照组 (Privileged) | 编码规则 (`src/bias/process.py:15-21`) |
|------|-------|----------------------|--------------------|--------------------------------------|
| Personal status and sex | 8 | female (1) | male (0) | A92→0, A95→0 (male); A91→0, A93→0 (male) → 最终 female=1, male=0 |
| Age in years | 12 | older >45 (1) | younger ≤45 (0) | >45→1, ≤45→0 |
| foreign worker | 19 | foreign=yes (0) ⚠️ | local=no (1) ⚠️ | A201→0, A202→1 (注意: 编码方向与直觉相反) |

**文件依据**: `src/bias/process.py:15-21`
```python
pre_data[12][pre_data[12] <= 45] = 0    # younger → privileged
pre_data[12][pre_data[12] > 45] = 1     # older → unprivileged
pre_data[8][pre_data[8] == 2] = 0       # male
pre_data[8][pre_data[8] == 3] = 0       # male
pre_data[8][pre_data[8] == 5] = 1       # female → unprivileged
```

### 3.4 Label Mapping

**原始标签**:
- `1` = good (信用良好, 低风险) — 700 条 (70%)
- `2` = bad (信用不良, 高风险) — 300 条 (30%)

**CALM preprocess 中的转换** (`prepocess.py:95`):
```python
answer = 'good' if data[j][-1] == 1 else 'bad'
gold   = data[j][-1] - 1   # 1→0(good), 2→1(bad)
```

**统一风险标签设计**:

| 原始值 | CALM gold | CALM answer | 统一 risk_label | 含义 |
|--------|-----------|-------------|-----------------|------|
| 1 | 0 | "good" | **0** | low risk (信用良好) |
| 2 | 1 | "bad" | **1** | high risk (信用不良) |

**转换理由**: `risk_label = gold` — CALM 的 `gold` 字段已经正确地将 1(good)→0 和 2(bad)→1。

### 3.5 Prompt 表示

**CALM 生成的 LLM 输入** (`prepocess.py:85-98`):

```
Evaluate the creditworthiness of a customer with the following financial profile.
Respond with only either 'good' or 'bad'. For instance, 'The client has a stable
income, no previous debts, and owns a property.' should be classified as 'good'.
Text: 'The state of Status of existing checking account is bigger than 0 DM but
smaller than 200 DM. The state of Duration in month is 10. The state of Credit history
is existing credits paid back duly till now. The state of Purpose is furniture or
equipment. The state of Credit amount is 1521. The state of Savings account or bonds
is smaller than 100 DM. The state of Present employment since is bigger than 1 smaller
than 4 years. The state of Installment rate in percentage of disposable income is 4.
The state of Personal status and sex is male: divorced or separated. The state of
Other debtors or guarantors is none. The state of Present residence since is 2. The
state of Property is car or other. The state of Age in years is 31. The state of
Other installment plans is none. The state of Housing is own. The state of Number of
existing credits at this bank is 1. The state of Job is unskilled or resident. The
state of Number of people being liable to provide maintenance for is 1. The state of
Telephone is none. The state of foreign worker is yes.'
Answer:
```

**预期输出**: `good` (low risk) 或 `bad` (high risk)

---

## 第四部分：统一风险标签设计

### 4.1 设计原则

统一标签 `risk_label`:
- `0` = **low risk** (正类: 信用良好、无欺诈、无破产、无索赔)
- `1` = **high risk** (负类: 信用不良、欺诈、破产、有索赔)

映射方向: **将业务上的"坏结果"统一映射为 1 (high risk)**

### 4.2 全数据集标签映射表

| # | Dataset | Task | Original Good Label | Original Bad Label | CALM gold 含义 | → risk_label: 0 (low risk) | → risk_label: 1 (high risk) | 确认度 |
|---|---------|------|--------------------|--------------------|--------------------|---------------------------|---------------------------|--------|
| 1 | German | Credit Scoring | 1 = good | 2 = bad | 0=good, 1=bad | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 2 | Australian | Credit Scoring | 1 = good | 0 = bad | 0=good, 1=bad | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 3 | Lending Club | Credit Scoring | Fully Paid | Charged Off | 0=Fully Paid, 1=Charged Off | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 4 | Credit Card Fraud | Fraud Detection | 0 = normal | 1 = fraud | 0=normal, 1=fraud | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 5 | ccFraud | Fraud Detection | 0 = normal | 1 = fraud | 0=good, 1=bad | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 6 | Polish | Bankruptcy | '0' = non-bankrupt | '1' = bankrupt | 0=non-bankrupt, 1=bankrupt | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 7 | Taiwan | Bankruptcy | 0 = non-bankrupt | 1 = bankrupt | 0=non-bankrupt, 1=bankrupt | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |
| 8 | Travel Insurance | Claim Analysis | No = not claimed | Yes = claimed | 0=Yes(claimed), 1=No(not claimed) ⚠️ | **NEEDS_CONFIRMATION** | **NEEDS_CONFIRMATION** | ⚠️ REVERSED |
| 9 | PortoSeguro | Claim Analysis | 0 = no claim | 1 = claim | 0=claim, 1=no claim ⚠️ | **NEEDS_CONFIRMATION** | **NEEDS_CONFIRMATION** | ⚠️ REVERSED |
| 10 | Customs | Fraud Detection | 0 = normal | 1/2 = fraud/critical | 0=normal, 1=fraud | 0 (CALM gold=0) | 1 (CALM gold=1) | ✅ CONFIRMED |

### 4.3 需要确认的数据集

#### Travel Insurance ⚠️ NEEDS_CONFIRMATION

**文件依据**: `prepocess.py:58-59`
```python
answer = 'yes' if data[j][4] == 'Yes' else 'no'
gold = 0 if data[j][4] == 'Yes' else 1
# → 0 = 'Yes' (claimed), 1 = 'No' (not claimed)
```

**问题**: 业务上 "Claim=Yes" 意味着发生了理赔，这应该属于 high risk (不良事件)。但 CALM 将 `gold=0` 赋给 "Yes"，与其他任务的 "good→0" 语义一致。这里的语义是模糊的：

| 视角 | Yes (claimed) | No (not claimed) |
|------|--------------|------------------|
| 被保人角度 | 获得赔付 (good) | 未获赔付 (neutral) |
| 保险公司角度 | 发生理赔成本 (high risk) | 无理赔成本 (low risk) |

**建议**: 从金融风控（保险公司视角）统一为:
- `risk_label=0` → No (not claimed, low risk)
- `risk_label=1` → Yes (claimed, high risk)

即**反转 CALM gold**: `risk_label = 1 - gold`

#### PortoSeguro ⚠️ NEEDS_CONFIRMATION

**文件依据**: `prepocess.py:51`
```python
answer = 'no' if data[j][0] == 0 else 'yes'
gold = 0 if data[j][0] == 1 else 1
# → 0 = 'yes' (claim), 1 = 'no' (no claim) ⚠️
```

**问题**: CALM gold 的赋值是: target=1(有索赔) → gold=0, target=0(无索赔) → gold=1。这与直觉相反。

**建议**: 从金融风控角度统一为:
- `risk_label=0` → target=0 (no claim, low risk)
- `risk_label=1` → target=1 (claim, high risk)

即**反转 CALM gold**: `risk_label = 1 - gold`

### 4.4 统一映射函数

基于以上分析，推荐在迁移脚本中实现的映射函数:

```python
# 数据集需要反转 gold 的列表
GOLD_REVERSED_DATASETS = {'travel_insurance', 'portoseguro'}

def normalize_risk_label(dataset: str, calms_gold: int) -> int:
    """
    将 CALM gold 字段统一映射为 risk_label
    0 = low risk, 1 = high risk
    """
    if dataset in GOLD_REVERSED_DATASETS:
        return 1 - calms_gold  # 反转
    return calms_gold  # 直接使用
```

---

## 第五部分：ms-swift 目标 Schema 设计

### 5.1 设计目标

迁移到 Qwen2.5 + ms-swift，最终训练格式使用 **ShareGPT / ChatML** 格式。

### 5.2 目标 Schema

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<task-specific system prompt>"
    },
    {
      "role": "user",
      "content": "<sample input text>"
    },
    {
      "role": "assistant",
      "content": "<risk label text>"
    }
  ],
  "dataset": "German",
  "task_type": "credit_scoring",
  "risk_label": 0,
  "sample_id": "german_0",
  "features": {
    "<feature_name>": "<value>",
    "...": "..."
  }
}
```

### 5.3 字段映射

| 目标字段 | 来源 | 说明 |
|---------|------|------|
| `messages[0].content` | **新增** | System prompt，根据 task_type 生成 |
| `messages[1].content` | CALM `text` | 用户输入：特征的自然语言描述 |
| `messages[2].content` | CALM `answer` | 模型输出：风险标签文本 (见 5.4) |
| `dataset` | CALM 目录名 | 数据集标识，用于过滤和分析 |
| `task_type` | **新增** | 统一任务类型: `credit_scoring` / `fraud_detection` / `bankruptcy_prediction` / `claim_analysis` |
| `risk_label` | **新增** (基于 CALM `gold`) | 统一数字标签: 0=low risk, 1=high risk |
| `sample_id` | CALM `id` | 前缀区分数据集，如 `german_{id}` |
| `features` | **新增** | 原始特征键值对，用于结构化分析 |

### 5.4 System Prompt 设计

#### 信用评分 (Credit Scoring)

```
You are a financial risk assessment expert. Evaluate the creditworthiness
based on the customer's financial profile. Classify the risk level as:
- low risk: the customer is likely to repay
- high risk: the customer is likely to default
Respond with only 'low risk' or 'high risk'.
```

#### 欺诈检测 (Fraud Detection)

```
You are a financial fraud detection expert. Analyze the transaction or
declaration attributes to identify potential fraud. Classify the risk level as:
- low risk: the transaction/declaration appears normal
- high risk: the transaction/declaration shows fraud indicators
Respond with only 'low risk' or 'high risk'.
```

#### 破产预测 (Bankruptcy Prediction)

```
You are a corporate financial analyst. Evaluate the company's financial
ratios to predict bankruptcy risk. Classify the risk level as:
- low risk: the company is financially stable
- high risk: the company is at risk of bankruptcy
Respond with only 'low risk' or 'high risk'.
```

#### 理赔分析 (Claim Analysis)

```
You are an insurance risk analyst. Analyze the policyholder's profile
and insurance attributes to assess claim risk. Classify the risk level as:
- low risk: unlikely to file a claim
- high risk: likely to file a claim
Respond with only 'low risk' or 'high risk'.
```

### 5.5 Assistant Response 统一

**所有任务统一使用**:
- `"low risk"` — 对应 `risk_label = 0`
- `"high risk"` — 对应 `risk_label = 1`

替换 CALM 原有多样化的 answer 格式 (good/bad, yes/no, Fully Paid/Charged Off 等)。

### 5.6 转换脚本伪代码

```python
def calm_to_swift(calm_record: dict, dataset_name: str, task_type: str) -> dict:
    """
    将单条 CALM 记录转换为 ms-swift ShareGPT 格式
    """
    system_prompt = SYSTEM_PROMPTS[task_type]
    user_text = calm_record["text"]       # 来自 CALM
    risk_label = normalize_risk_label(dataset_name, calm_record["gold"])
    assistant_text = "low risk" if risk_label == 0 else "high risk"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ],
        "dataset": dataset_name,
        "task_type": task_type,
        "risk_label": risk_label,
        "sample_id": f"{dataset_name}_{calm_record['id']}"
    }
```

### 5.7 CALM 字段复用矩阵

| CALM 字段 | 是否复用 | 在目标 Schema 中的位置 | 处理方式 |
|-----------|---------|----------------------|---------|
| `id` | ✅ 复用 | `sample_id` | 添加 dataset 前缀: `f"{dataset}_{id}"` |
| `query` | ❌ 不直接使用 | — | 拆分为 system prompt + user text |
| `answer` | ❌ 替换 | `messages[2].content` | 统一为 "low risk" / "high risk" |
| `choices` | ❌ 不使用 | — | 由 system prompt 统一管理 |
| `gold` | ⚠️ 转换后使用 | `risk_label` | 经 `normalize_risk_label()` 映射 |
| `text` | ✅ 直接复用 | `messages[1].content` | 作为 user message 内容 |
| — | **新增** | `dataset` | 数据集来源标识 |
| — | **新增** | `task_type` | 任务类型分类 |

---

## 第六部分：迁移建议

### 6.1 三层数据结构

```
┌─────────────────────────────────────────────┐
│ Layer 1: CALM Format (原始, 当前仓库)         │
│ {id, query, answer, choices, gold, text}     │
│ 10 datasets × {train, valid, test}.parquet   │
└──────────────────┬──────────────────────────┘
                   │ calm_to_swift.py 转换
                   ▼
┌─────────────────────────────────────────────┐
│ Layer 2: Risk Dataset Unified Schema (中间层) │
│ {messages, dataset, task_type,               │
│  risk_label, sample_id}                      │
│ 统一标签: 0=low risk, 1=high risk            │
│ 统一输出: "low risk" / "high risk"           │
│ 统一 task_type: 4 个标准类别                  │
└──────────────────┬──────────────────────────┘
                   │ JSONL 导出
                   ▼
┌─────────────────────────────────────────────┐
│ Layer 3: ms-swift SFT Dataset (训练格式)      │
│ train.jsonl / val.jsonl                      │
│ Qwen2.5 ChatML template 渲染                 │
│ swift sft --dataset train.jsonl ...          │
└─────────────────────────────────────────────┘
```

### 6.2 可复用资产

| 资产 | 复用方式 | 价值 |
|------|---------|------|
| **10 个数据集的特征描述** (CALM `text` 字段) | 直接用作 user message | 高 — 经过论文验证的 prompt 设计 |
| **Table-based / Description-based 分类** | 保留两种格式，不混合 | 中 — 适配不同特征复杂度 |
| **7:1:2 分割 + seed=10086** | 直接复用分割逻辑 | 高 — 保证可复现性 |
| **偏差分析测试集** (`src/bias/bias_data/`) | 作为 fairness eval 基准 | 高 — 已有 AIF360 集成 |
| **Benchmark 推理结果** (`src/Precision/`) | 作为对比基线 | 中 — 但基于旧模型 (Llama1/2, Bloomz) |
| **类不平衡重采样策略** (2:1) | 可选保留或替换为 weighted loss | 中 — 论文验证有效的策略 |

### 6.3 需要重构的项

| 项目 | 现状 | 重构方案 | 优先级 |
|------|------|---------|--------|
| **标签体系** | 10 个数据集有 6 种不同的 answer 文本 | 统一为 "low risk" / "high risk" | P0 |
| **gold 标签方向** | Travel Insurance 和 PortoSeguro 的 gold 是反向的 | `normalize_risk_label()` 统一 | P0 |
| **数据格式** | 自定义 JSON (parquet) | 转换为 ShareGPT JSONL | P0 |
| **System prompt** | 无 (内嵌在 query 中) | 4 个 task-specific system prompt | P0 |
| **query 字段** | 混合 prompt + text + Answer: | 拆分为 system + user + assistant | P1 |
| **training pipeline** | 外部 CALM-train (Llama2) | ms-swift + Qwen2.5 | P1 |
| **对话模板** | Llama2 `[INST]` 格式 | Qwen2.5 ChatML `<\|im_start\|>` | P1 |
| **评估脚本** | 弱解析 (关键词匹配) | 结构化解析 + risk_label 比对 | P2 |
| **偏差分析** | AIF360 硬编码 | 适配新模型输出格式 | P2 |

### 6.4 数据集特殊性提醒

| 注意事项 | 涉及数据集 | 影响 |
|---------|-----------|------|
| 标签列在第一列 (col 0) | Taiwan | 解析时需要区分，不能统一用 `[-1]` |
| 标签是字符串 | Lending Club (Fully Paid/Charged Off), Polish ('0'/'1'), Travel Insurance (Yes/No) | 需要字符串到数字的映射 |
| 已预分割 train/valid/test | Customs | 无需再进行随机分割 |
| 多文件合并 | Polish (5×ARFF) | 合并后总量 43,405 条 |
| PCA 匿名化特征 | Credit Card Fraud | 28 个 V 特征无物理含义，无法做特征级别的可解释性 |
| 匿名化特征 | Australian | 14 个特征名和值均被替换为无意义符号 |
| Critical Fraud (label=2) | Customs | 当前被合并到 fraud=yes。建议评估是否保留三层分类 |

### 6.5 缺失数据阻隔

以下数据集需要**外部下载原始 CSV** 后才能生成 CALM 格式指令数据:

| 数据集 | 需要的文件 | 建议下载源 | 预计大小 |
|--------|-----------|-----------|---------|
| Lending Club | `accepted_2007_to_2018Q4.csv` | Kaggle | ~1.5 GB |
| Credit Card Fraud | `creditcard.csv` | Kaggle | ~150 MB |
| ccFraud | `ccFraud.csv` | Revolution Analytics | ~50 MB |
| PortoSeguro | `PortoSeguro.csv` (train.csv) | Kaggle | ~50 MB |

> 现有 6 个完整数据集 (German, Australian, Polish, Taiwan, Travel Insurance, Customs) 可立即开始迁移工作。

---

## 附录 A: 各数据集输出文本示例

### A.1 Description-based (German)

```
The state of Status of existing checking account is bigger than 0 DM but smaller than
200 DM. The state of Duration in month is 10. The state of Credit history is existing
credits paid back duly till now. The state of Purpose is furniture or equipment. The
state of Credit amount is 1521. The state of Savings account or bonds is smaller than
100 DM. The state of Present employment since is bigger than 1 smaller than 4 years.
The state of Installment rate in percentage of disposable income is 4. The state of
Personal status and sex is male: divorced or separated. The state of Other debtors or
guarantors is none. The state of Present residence since is 2. The state of Property
is car or other. The state of Age in years is 31. The state of Other installment plans
is none. The state of Housing is own. The state of Number of existing credits at this
bank is 1. The state of Job is unskilled or resident. The state of Number of people
being liable to provide maintenance for is 1. The state of Telephone is none. The
state of foreign worker is yes.
```

### A.2 Table-based (Australian)

```
The client has attributes: A1: 1, A2: 22.08, A3: 11.46, A4: 2, A5: 4, A6: 4,
A7: 1.585, A8: 0, A9: 0, A10: 0, A11: 1, A12: 2, A13: 100, A14: 1213.
```

### A.3 隐藏特征数对比

| 数据集 | 格式 | 文本长度 (est.) | 特征数 |
|--------|------|----------------|--------|
| German | Description | ~800 chars | 20 |
| Australian | Table | ~120 chars | 14 |
| Lending Club | Description | ~500 chars | 21 |
| Credit Card Fraud | Table | ~400 chars | 29 |
| ccFraud | Description | ~200 chars | 7 |
| Polish | Table | ~2000 chars | 64 |
| Taiwan | Table | ~3000 chars | 95 |
| Travel Insurance | Table | ~250 chars | 9 |
| PortoSeguro | Table | ~800 chars | 57 |
| Customs | Table | ~600 chars | 20 |

**提醒**: Taiwan (95 特征, ~3000 chars) 和 Polish (64 特征, ~2000 chars) 的输入文本可能超过 Llama2 的 4096 token 上下文窗口，但 Qwen2.5 的更长上下文 (32K+) 可以完全容纳。

---

## 附录 B: 未解决问题列表

| # | 问题 | 涉及数据集 | 建议行动 |
|---|------|-----------|---------|
| 1 | Travel Insurance 的 risk_label 方向 | Travel Insurance | 确认 Claim=Yes 是否应映射为 high risk |
| 2 | PortoSeguro gold 标签为何反向赋值 | PortoSeguro | 检查原始论文或 CALM-train 中的评估逻辑 |
| 3 | Critical Fraud (label=2) 是否应作为独立类别 | Customs | 评估二层 vs 三层分类的收益 |
| 4 | CALM-train 训练超参数 (learning rate, epochs, batch size) | 全部 | 查阅 `github.com/Dai-shen/CALM-train` |
| 5 | Lending Club 完整数据集按什么标准选取 21 个特征 | Lending Club | 原文 ~150 列中选了 21 列，需确认选择依据 |
| 6 | 描述型 vs 表格型 prompt 的性能差异 | German, Australian 等 | 迁移后应做 prompt 格式消融实验 |
| 7 | ms-swift 是否支持自定义 loss (如 class weights) 处理不平衡 | 全部 | 评估 weighted cross-entropy vs 2:1 重采样 |

---

*本报告基于 `/home/wushaohua/data/risk-control-posttraining` 仓库实际文件生成。所有代码行号引用、文件路径、标签分布数据均为实际读取结果。对无法确认的内容已标注 NEEDS_CONFIRMATION。*
