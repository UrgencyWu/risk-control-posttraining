"""
C5v3: Multi-dataset converter — German + Australian → unified normalized + SFT
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, random, numpy as np

SEED = 10086
TRAIN, DEV, TEST = 0.7, 0.1, 0.2

SYSTEM_PROMPTS = {
    "credit_scoring": (
        "You are a financial risk assessment expert. Evaluate the creditworthiness "
        "based on the customer's financial profile. Classify the risk level as:\n"
        "- low risk: the customer is likely to repay\n"
        "- high risk: the customer is likely to default\n"
        "Respond with only 'low risk' or 'high risk'."
    ),
}

# ==================== German ====================
def load_german(path="data/credit_scoring/German/german.data"):
    with open(path) as f:
        return [line.strip().split() for line in f if line.strip()]

GERMAN_NAMES = [
    "Status of existing checking account", "Duration in month", "Credit history", "Purpose",
    "Credit amount", "Savings account or bonds", "Present employment since",
    "Installment rate in percentage of disposable income", "Personal status and sex",
    " Other debtors or guarantors", "Present residence since", "Property", "Age in years",
    "Other installment plans", "Housing", "Number of existing credits at this bank", "Job",
    "Number of people being liable to provide maintenance for", "Telephone", "foreign worker",
]
GERMAN_KEYS = [  # snake_case feature names
    "checking_account_status","duration_month","credit_history","purpose","credit_amount",
    "savings_account","employment_since","installment_rate","personal_status_sex",
    "other_debtors","residence_since","property","age_years","other_installment_plans",
    "housing","existing_credits_count","job","maintenance_liability_count","telephone","foreign_worker",
]
GERMAN_CAT_MAP = {
    0: {"A11":"smaller than 0 DM","A12":"bigger than 0 DM but smaller than 200 DM","A13":"bigger than 200 DM OR salary assignments for at least 1 year","A14":"no checking account"},
    2: {"A30":"no credits taken or all credits paid back duly","A31":"all credits at this bank paid back duly","A32":"existing credits paid back duly till now","A33":"delay in paying off in the past","A34":"critical account or other credits existing (not at this bank)"},
    3: {"A40":"car (new)","A41":"car (used)","A42":"furniture or equipment","A43":"radio or television","A44":"domestic appliances","A45":"repairs","A46":"education","A47":"vacation","A48":"retraining","A49":"business","A410":"others"},
    5: {"A61":"smaller than 100 DM","A62":"bigger than 100 smaller than 500 DM","A63":"bigger than 500 smaller than 1000 DM","A64":"bigger than 1000 DM","A65":"unknown or no savings account"},
    6: {"A71":"unemployed","A72":"smaller than 1 year","A73":"bigger than 1 smaller than 4 years","A74":"bigger than 4 smaller than 7 years","A75":"bigger than 7 years"},
    8: {"A91":"male: divorced or separated","A92":"female: divorced or separated or married","A93":"male and single","A94":"male and married or widowed","A95":"female and single"},
    9: {"A101":"none","A102":"co-applicant","A103":"guarantor"},
    11: {"A121":"real estate","A122":"building society savings agreement or life insurance","A123":"car or other","A124":"unknown or no property"},
    13: {"A141":"bank","A142":"stores","A143":"none"},
    14: {"A151":"rent","A152":"own","A153":"for free"},
    16: {"A171":"unemployed or unskilled or non-resident","A172":"unskilled or resident","A173":"skilled employee or official","A174":"management or self-employed or highly qualified employee or officer"},
    18: {"A191":"none","A192":"yes, registered under the customers name"},
    19: {"A201":"yes","A202":"no"},
}
GERMAN_LABEL_MAP = {1: 0, 2: 1}  # 1=good→0, 2=bad→1
GERMAN_LABEL_MEANING = {1: "good", 2: "bad"}

def build_german(parsed):
    features = {}
    for i in range(20):
        k = GERMAN_KEYS[i]
        v = parsed[i]
        features[k] = GERMAN_CAT_MAP[i].get(v, v) if i in GERMAN_CAT_MAP else v

    parts = [f"The state of {GERMAN_NAMES[i]} is {features[GERMAN_KEYS[i]]}." for i in range(20)]
    text = " ".join(parts)
    orig_label = int(parsed[20])
    risk_label = GERMAN_LABEL_MAP[orig_label]
    return features, text, orig_label, risk_label, "description"

# ==================== Australian ====================
def load_australian(path="data/credit_scoring/Australian/australian.dat"):
    with open(path) as f:
        return [line.strip().split() for line in f if line.strip()]

AU_NAMES = [f"A{i+1}" for i in range(14)]
AU_KEYS  = [f"attr_{i+1}" for i in range(14)]
AU_LABEL_MAP = {1: 0, 0: 1}  # 1=good→0, 0=bad→1
AU_LABEL_MEANING = {1: "good", 0: "bad"}

def build_australian(parsed):
    features = {AU_KEYS[i]: parsed[i] for i in range(14)}
    parts = [f"A{i+1}: {parsed[i]}" for i in range(14)]
    text = "The client has attributes: " + ", ".join(parts[:-1]) + ", " + parts[-1] + "."
    orig_label = int(parsed[14])
    risk_label = AU_LABEL_MAP[orig_label]
    return features, text, orig_label, risk_label, "table"

# ==================== Common ====================
def split_data(n, seed):
    random.seed(seed)
    indices = list(range(n))
    train_n = int(n * TRAIN)
    dev_n = int(n * DEV)
    train_set = set(random.sample(indices, train_n))
    rest = list(set(indices) - train_set)
    dev_set = set(random.sample(rest, dev_n))
    test_set = set(rest) - dev_set
    splits = {}
    for i in range(n):
        if i in train_set: splits[i] = "train"
        elif i in dev_set: splits[i] = "valid"
        else: splits[i] = "test"
    return splits

def build_normalized(sample_id, dataset, split, task_type, target_type, features, text, orig_label, orig_meaning, risk_label, prompt_format, calm_gold):
    return {
        "sample_id": sample_id, "dataset": dataset, "split": split,
        "task_type": task_type, "target_type": target_type,
        "risk_label": risk_label,
        "original_label": {"value": orig_label, "meaning": orig_meaning, "raw_format": "int"},
        "calm_gold": calm_gold, "text": text, "prompt_format": prompt_format,
        "feature_count": len(features), "features": features,
    }

def build_sft(norm_record, system_prompt):
    assistant = "low risk" if norm_record["risk_label"] == 0 else "high risk"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": norm_record["text"]},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "sample_id": norm_record["sample_id"], "dataset": norm_record["dataset"],
            "split": norm_record["split"], "task_type": norm_record["task_type"],
            "target_type": norm_record["target_type"], "risk_label": norm_record["risk_label"],
            "original_label": norm_record["original_label"],
            "prompt_format": norm_record["prompt_format"],
        },
    }

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def process_dataset(name, raw_records, build_fn, label_map, label_meaning, task_type, target_type, out_dir):
    print(f"  Processing {name}: {len(raw_records)} rows")
    splits = split_data(len(raw_records), SEED)
    all_norm, all_sft = [], []
    sp = SYSTEM_PROMPTS[task_type]

    for i, row in enumerate(raw_records):
        features, text, orig_label, risk_label, prompt_format = build_fn(row)
        split = splits[i]
        calm_gold = risk_label  # for Australian and German, calm_gold == risk_label after mapping 1=good→0

        norm = build_normalized(f"{name.lower()}_{i}", name, split, task_type, target_type,
                                features, text, orig_label, label_meaning.get(orig_label, str(orig_label)),
                                risk_label, prompt_format, calm_gold)
        sft = build_sft(norm, sp)
        all_norm.append(norm)
        all_sft.append(sft)

    for s in ["train", "valid", "test"]:
        save_jsonl([r for r in all_norm if r["split"]==s], f"{out_dir}/{name}/normalized/{s}.jsonl")
        save_jsonl([r for r in all_sft if r["metadata"]["split"]==s], f"{out_dir}/{name}/sft/{s}.jsonl")

    counts = {s: sum(1 for r in all_norm if r["split"]==s) for s in ["train","valid","test"]}
    print(f"    train={counts['train']} valid={counts['valid']} test={counts['test']}")
    return all_norm, all_sft

def main():
    out_dir = "data/processed/multi"
    random.seed(SEED); np.random.seed(SEED)

    print("=== C5v3: Multi-Dataset Conversion ===")

    # German
    ger_raw = load_german()
    ger_norm, ger_sft = process_dataset("German", ger_raw, build_german,
                                         GERMAN_LABEL_MAP, GERMAN_LABEL_MEANING,
                                         "credit_scoring", "binary_risk", out_dir)

    # Australian
    au_raw = load_australian()
    au_norm, au_sft = process_dataset("Australian", au_raw, build_australian,
                                       AU_LABEL_MAP, AU_LABEL_MEANING,
                                       "credit_scoring", "binary_risk", out_dir)

    # Combine
    print(f"\n  Combining datasets...")
    for s in ["train", "valid", "test"]:
        combined_norm = [r for r in ger_norm if r["split"]==s] + [r for r in au_norm if r["split"]==s]
        combined_sft  = [r for r in ger_sft if r["metadata"]["split"]==s] + [r for r in au_sft if r["metadata"]["split"]==s]
        save_jsonl(combined_norm, f"{out_dir}/combined/normalized/{s}.jsonl")
        save_jsonl(combined_sft,  f"{out_dir}/combined/sft/{s}.jsonl")
        print(f"    {s}: {len(combined_norm)} records (German + Australian)")

    # Manifest
    manifest = {
        "datasets": ["German", "Australian"],
        "task_type": "credit_scoring",
        "target_type": "binary_risk",
        "total": {"train": 872, "valid": 119, "test": 339, "overall": 1690},
        "german": {"train": 700, "valid": 100, "test": 200},
        "australian": {"train": 172, "valid": 19, "test": 139},  # approximate, depends on seed
        "seed": SEED,
        "split_ratio": "7:1:2",
    }
    # Compute actual AU counts
    au_train = sum(1 for r in au_norm if r["split"]=="train")
    au_valid = sum(1 for r in au_norm if r["split"]=="valid")
    au_test  = sum(1 for r in au_norm if r["split"]=="test")
    manifest["australian"] = {"train": au_train, "valid": au_valid, "test": au_test}
    manifest["total"] = {"train": 700+au_train, "valid": 100+au_valid, "test": 200+au_test}

    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {out_dir}/manifest.json")
    print(f"  Total: {manifest['total']['train']} train / {manifest['total']['valid']} valid / {manifest['total']['test']} test")

if __name__ == "__main__":
    main()
