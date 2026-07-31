import json
import unittest

from src.evaluation.c7_final import REPOSITORY_ROOT


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class GermanDatasetContractTests(unittest.TestCase):
    def test_frozen_split_sizes_and_sample_ids(self):
        root = REPOSITORY_ROOT / "data/processed/german"
        normalized_splits = {
            split: load_jsonl(root / "normalized" / f"{split}.jsonl")
            for split in ("train", "valid", "test")
        }

        self.assertEqual({"train": 700, "valid": 100, "test": 200}, {
            split: len(records) for split, records in normalized_splits.items()
        })
        split_ids = [
            {record["sample_id"] for record in records}
            for records in normalized_splits.values()
        ]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])

    def test_sft_metadata_matches_normalized_labels(self):
        root = REPOSITORY_ROOT / "data/processed/german"
        for split in ("train", "valid", "test"):
            with self.subTest(split=split):
                normalized = load_jsonl(root / "normalized" / f"{split}.jsonl")
                sft = load_jsonl(root / "sft" / f"{split}.jsonl")
                self.assertEqual(
                    [(record["sample_id"], record["risk_label"]) for record in normalized],
                    [
                        (record["metadata"]["sample_id"], record["metadata"]["risk_label"])
                        for record in sft
                    ],
                )


if __name__ == "__main__":
    unittest.main()
