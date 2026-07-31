"""
German Credit Converter
CALM Raw → Normalized Layer → SFT ChatML Layer
遵循 docs/German_Credit_Converter_Plan.md 设计
"""
import json
import random
import os

# Fixed timestamp for reproducibility
FROZEN_TIMESTAMP = "2026-07-21T00:00:00Z"

# ============================================================
# Config
# ============================================================
INPUT_FILE = "data/credit_scoring/German/german.data"
OUTPUT_DIR = "data/processed/german"
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
SEED = 10086

# ============================================================
# Feature definitions (from prepocess.py)
# ============================================================
FEATURE_NAMES = [
    "checking_account_status",  # 0
    "duration_month",           # 1
    "credit_history",           # 2
    "purpose",                  # 3
    "credit_amount",            # 4
    "savings_account",          # 5
    "employment_since",         # 6
    "installment_rate",         # 7
    "personal_status_sex",      # 8
    "other_debtors",            # 9
    "residence_since",          # 10
    "property",                 # 11
    "age_years",                # 12
    "other_installment_plans",  # 13
    "housing",                  # 14
    "existing_credits_count",   # 15
    "job",                      # 16
    "maintenance_liability_count",  # 17
    "telephone",                # 18
    "foreign_worker",           # 19
]

# Categorical feature index -> {raw_code: human_description}
CATEGORY_MAP = {
    0: {"A11": "smaller than 0 DM", "A12": "bigger than 0 DM but smaller than 200 DM",
        "A13": "bigger than 200 DM OR salary assignments for at least 1 year", "A14": "no checking account"},
    2: {"A30": "no credits taken or all credits paid back duly", "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
        "A34": "critical account or other credits existing (not at this bank)"},
    3: {"A40": "car (new)", "A41": "car (used)", "A42": "furniture or equipment",
        "A43": "radio or television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "others"},
    5: {"A61": "smaller than 100 DM", "A62": "bigger than 100 smaller than  500 DM",
        "A63": "bigger than 500 smaller than 1000 DM", "A64": "bigger than 1000 DM",
        "A65": "unknown or no savings account"},
    6: {"A71": "unemployed", "A72": "smaller than 1 year", "A73": "bigger than 1  smaller than 4 years",
        "A74": "bigger than 4  smaller than 7 years", "A75": "bigger than 7 years"},
    8: {"A91": "male: divorced or separated", "A92": "female: divorced or separated or married",
        "A93": "male and single", "A94": "male and married or widowed", "A95": "female and single"},
    9: {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    11: {"A121": "real estate", "A122": "building society savings agreement or life insurance",
         "A123": "car or other", "A124": "unknown or no property"},
    13: {"A141": "bank", "A142": "stores", "A143": "none"},
    14: {"A151": "rent", "A152": "own", "A153": "for free"},
    16: {"A171": "unemployed or unskilled or non-resident", "A172": "unskilled or resident",
         "A173": "skilled employee or official",
         "A174": "management or self-employed or highly qualified employee or officer"},
    18: {"A191": "none", "A192": "yes, registered under the customers name"},
    19: {"A201": "yes", "A202": "no"},
}

# System prompt
SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)

# Label mapping (GermanAdapter)
LABEL_MEANING = {1: "good", 2: "bad"}
RISK_LABEL_MAP = {1: 0, 2: 1}  # GermanAdapter.map_risk_label()

# Protected attribute computation
def compute_protected_attributes(features):
    pa = {}
    # gender: A91/A93/A94 → male, A92/A95 → female
    raw_gender = features.get("personal_status_sex", "")
    if "female" in raw_gender:
        pa["gender"] = "female"
    else:
        pa["gender"] = "male"
    # age_group
    age = int(features.get("age_years", 0))
    pa["age_group"] = "old" if age > 45 else "young"
    # foreign_status
    raw_foreign = features.get("foreign_worker", "")
    pa["foreign_status"] = "yes" if raw_foreign == "yes" else "no"
    return pa


# ============================================================
# Step 1: Load raw data
# ============================================================
def load_raw(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            records.append(parts)
    return records


# ============================================================
# Step 2: Build features dict
# ============================================================
def build_features(parsed):
    features = {}
    for i in range(20):
        raw_val = parsed[i]
        name = FEATURE_NAMES[i]
        if i in CATEGORY_MAP:
            features[name] = CATEGORY_MAP[i].get(raw_val, raw_val)
        else:
            features[name] = raw_val
    return features


# ============================================================
# Step 3: Build text description
# ============================================================
ORIGINAL_FEATURE_NAMES = [
    "Status of existing checking account", "Duration in month", "Credit history", "Purpose",
    "Credit amount", "Savings account or bonds", "Present employment since",
    "Installment rate in percentage of disposable income", "Personal status and sex",
    " Other debtors or guarantors", "Present residence since", "Property", "Age in years",
    "Other installment plans", "Housing", "Number of existing credits at this bank", "Job",
    "Number of people being liable to provide maintenance for", "Telephone", "foreign worker",
]

def build_text(features):
    parts = []
    for i in range(20):
        name = ORIGINAL_FEATURE_NAMES[i]
        value = features[FEATURE_NAMES[i]]
        parts.append(f"The state of {name} is {value}.")
    return " ".join(parts)


# ============================================================
# Step 4: Extract label
# ============================================================
def extract_label(parsed):
    value = int(parsed[20])
    return {"value": value, "meaning": LABEL_MEANING[value], "raw_format": "int"}


def map_risk_label(original_value):
    return RISK_LABEL_MAP[original_value]


# ============================================================
# Step 5: Split data (CALM-compatible: random.sample, 7:1:2)
# ============================================================
def split_data(records, seed):
    random.seed(seed)
    n = len(records)
    indices = list(range(n))

    train_ind = set(random.sample(indices, int(n * TRAIN_RATIO)))
    remaining = list(set(indices) - train_ind)
    dev_ind = set(random.sample(remaining, int(n * DEV_RATIO)))
    test_ind = set(remaining) - dev_ind

    splits = {}
    for i in range(n):
        if i in train_ind:
            splits[i] = "train"
        elif i in dev_ind:
            splits[i] = "valid"
        else:
            splits[i] = "test"
    return splits


# ============================================================
# Step 6: Build normalized record
# ============================================================
def build_normalized(idx, parsed, split):
    features = build_features(parsed)
    original_label = extract_label(parsed)
    risk_label = map_risk_label(original_label["value"])
    text = build_text(features)
    protected = compute_protected_attributes(features)

    return {
        "sample_id": f"german_{idx}",
        "dataset": "German",
        "split": split,
        "task_type": "credit_scoring",
        "target_type": "binary_risk",
        "risk_label": risk_label,
        "original_label": original_label,
        "calm_gold": risk_label,  # German: same mapping as risk_label
        "text": text,
        "prompt_format": "description",
        "feature_count": 20,
        "features": features,
        "protected_attributes": protected,
    }


# ============================================================
# Step 7: Build SFT record
# ============================================================
def build_sft(norm_record):
    risk_label = norm_record["risk_label"]
    assistant_content = "low risk" if risk_label == 0 else "high risk"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": norm_record["text"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "sample_id": norm_record["sample_id"],
            "dataset": norm_record["dataset"],
            "split": norm_record["split"],
            "task_type": norm_record["task_type"],
            "target_type": norm_record["target_type"],
            "risk_label": norm_record["risk_label"],
            "original_label": norm_record["original_label"],
            "prompt_format": norm_record["prompt_format"],
        },
    }


# ============================================================
# Step 8: Validate
# ============================================================
def validate(normalized_records, sft_records):
    errors = []

    # V1: sample count
    assert len(normalized_records) == 1000, f"V1 FAIL: expected 1000, got {len(normalized_records)}"

    # V2: label domain
    for r in normalized_records:
        if r["risk_label"] not in (0, 1):
            errors.append(f"V2 FAIL: {r['sample_id']} risk_label={r['risk_label']}")

    # V3: field completeness
    required_normalized = ["sample_id", "dataset", "split", "task_type", "target_type",
                           "risk_label", "original_label", "calm_gold", "text",
                           "prompt_format", "feature_count", "features"]
    for r in normalized_records:
        for f in required_normalized:
            if f not in r:
                errors.append(f"V3 FAIL: {r['sample_id']} missing field {f}")
        if len(r["features"]) != 20:
            errors.append(f"V3 FAIL: {r['sample_id']} feature_count={len(r['features'])}")

    # V4: no split overlap
    train_ids = {r["sample_id"] for r in normalized_records if r["split"] == "train"}
    valid_ids = {r["sample_id"] for r in normalized_records if r["split"] == "valid"}
    test_ids = {r["sample_id"] for r in normalized_records if r["split"] == "test"}
    if train_ids & valid_ids:
        errors.append("V4 FAIL: train/valid overlap")
    if train_ids & test_ids:
        errors.append("V4 FAIL: train/test overlap")
    if valid_ids & test_ids:
        errors.append("V4 FAIL: valid/test overlap")

    # V5: ChatML format
    for r in sft_records:
        msgs = r["messages"]
        if len(msgs) != 3:
            errors.append(f"V5 FAIL: {r['metadata']['sample_id']} messages len={len(msgs)}")
        if msgs[0]["role"] != "system":
            errors.append(f"V5 FAIL: {r['metadata']['sample_id']} msg[0] role")
        if msgs[1]["role"] != "user":
            errors.append(f"V5 FAIL: {r['metadata']['sample_id']} msg[1] role")
        if msgs[2]["role"] != "assistant":
            errors.append(f"V5 FAIL: {r['metadata']['sample_id']} msg[2] role")
        if msgs[2]["content"] not in ("low risk", "high risk"):
            errors.append(f"V5 FAIL: {r['metadata']['sample_id']} assistant content={msgs[2]['content']}")

    # V6: metadata traceability
    for n, s in zip(normalized_records, sft_records):
        m = s["metadata"]
        if m["sample_id"] != n["sample_id"]:
            errors.append(f"V6 FAIL: {n['sample_id']} sample_id mismatch")
        if m["risk_label"] != n["risk_label"]:
            errors.append(f"V6 FAIL: {n['sample_id']} risk_label mismatch")
        expected_risk = RISK_LABEL_MAP[n["original_label"]["value"]]
        if n["risk_label"] != expected_risk:
            errors.append(f"V6 FAIL: {n['sample_id']} risk_label={n['risk_label']} expected={expected_risk}")

    # V7: feature mapping (no raw codes in categorical features)
    for r in normalized_records:
        for i in CATEGORY_MAP:
            name = FEATURE_NAMES[i]
            val = r["features"][name]
            if val.startswith("A") and any(c.isdigit() for c in val[1:]):
                errors.append(f"V7 FAIL: {r['sample_id']} raw code in {name}={val}")

    if errors:
        for e in errors:
            print(f"  {e}")
        raise AssertionError(f"Validation failed: {len(errors)} errors")
    print("  All validations passed (V1-V7)")


# ============================================================
# Step 9: Save
# ============================================================
def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def compute_distribution(records):
    dist = {"low_risk": 0, "high_risk": 0}
    for r in records:
        if r["risk_label"] == 0:
            dist["low_risk"] += 1
        else:
            dist["high_risk"] += 1
    return dist

# ============================================================
# Main
# ============================================================
def main():
    print("=== German Credit Converter ===")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Split: {TRAIN_RATIO}:{DEV_RATIO}:{TEST_RATIO}, seed={SEED}")

    # Load
    print("\n[1] Loading raw data...")
    records = load_raw(INPUT_FILE)
    print(f"  Loaded {len(records)} records")

    # Split
    print("\n[2] Splitting data...")
    splits = split_data(records, SEED)
    train_count = sum(1 for v in splits.values() if v == "train")
    valid_count = sum(1 for v in splits.values() if v == "valid")
    test_count = sum(1 for v in splits.values() if v == "test")
    print(f"  train={train_count}, valid={valid_count}, test={test_count}")

    # Build normalized
    print("\n[3] Building normalized records...")
    normalized_records = []
    for idx, parsed in enumerate(records):
        split = splits[idx]
        norm = build_normalized(idx, parsed, split)
        normalized_records.append(norm)

    # Build SFT
    print("\n[4] Building SFT records...")
    sft_records = [build_sft(n) for n in normalized_records]

    # Validate
    print("\n[5] Validating...")
    validate(normalized_records, sft_records)

    # Save normalized
    print("\n[6] Saving normalized/ ...")
    norm_dir = os.path.join(OUTPUT_DIR, "normalized")
    os.makedirs(norm_dir, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        subset = [r for r in normalized_records if r["split"] == split_name]
        path = os.path.join(norm_dir, f"{split_name}.jsonl")
        save_jsonl(subset, path)
        print(f"  {path}: {len(subset)} records")

    # Save SFT
    print("\n[7] Saving sft/ ...")
    sft_dir = os.path.join(OUTPUT_DIR, "sft")
    os.makedirs(sft_dir, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        subset = [r for r in sft_records if r["metadata"]["split"] == split_name]
        path = os.path.join(sft_dir, f"{split_name}.jsonl")
        save_jsonl(subset, path)
        print(f"  {path}: {len(subset)} records")

    # Manifest
    print("\n[8] Generating manifest.json ...")
    manifest = {
        "dataset": "German",
        "task_type": "credit_scoring",
        "target_type": "binary_risk",
        "created_at": FROZEN_TIMESTAMP,
        "source_file": INPUT_FILE,
        "converter": "convert_german.py",
        "random_seed": SEED,
        "split_ratio": "7:1:2",
        "total_samples": len(normalized_records),
        "feature_count": 20,
        "label_distribution": {
            "overall": compute_distribution(normalized_records),
            "train": compute_distribution([r for r in normalized_records if r["split"] == "train"]),
            "valid": compute_distribution([r for r in normalized_records if r["split"] == "valid"]),
            "test": compute_distribution([r for r in normalized_records if r["split"] == "test"]),
        },
        "files_generated": [
            "normalized/train.jsonl", "normalized/valid.jsonl", "normalized/test.jsonl",
            "sft/train.jsonl", "sft/valid.jsonl", "sft/test.jsonl",
            "manifest.json",
        ],
    }
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  {manifest_path}")

    print("\n=== Done ===")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Normalized: {len(normalized_records)} records → normalized/{{train,valid,test}}.jsonl")
    print(f"  SFT: {len(sft_records)} records → sft/{{train,valid,test}}.jsonl")
    print(f"  Manifest: manifest.json")


if __name__ == "__main__":
    main()
