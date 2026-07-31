import copy
import unittest

from src.evaluation.c7_final import (
    REPOSITORY_ROOT,
    evaluate_at_frozen_threshold,
    evaluate_model,
    load_prediction_records,
    run_evaluation,
    validate_ground_truth_alignment,
)


class C7EvaluationTests(unittest.TestCase):
    def test_committed_artifacts_reproduce_validation_only_operating_points(self):
        results = run_evaluation()

        expected = {
            "Majority": (0.05, 325),
            "LogisticRegression": (0.20, 113),
            "Qwen-ZeroShot": (0.80, 140),
            "SFT-seed7": (0.15, 100),
            "SFT-multi": (0.25, 142),
        }
        self.assertEqual(set(expected), set(results))
        for model_name, (threshold, cost) in expected.items():
            with self.subTest(model=model_name):
                self.assertAlmostEqual(threshold, results[model_name]["threshold"])
                self.assertEqual(cost, results[model_name]["cost"])
                self.assertEqual("validation_cost_minimization", results[model_name]["threshold_source"])

    def test_test_evaluator_requires_a_frozen_threshold(self):
        records = load_prediction_records(
            REPOSITORY_ROOT / "outputs/baselines/qwen_zero_shot_test.jsonl"
        )
        with self.assertRaises(ValueError):
            evaluate_at_frozen_threshold(records, None)

    def test_test_label_changes_cannot_change_the_selected_threshold(self):
        valid_records = load_prediction_records(
            REPOSITORY_ROOT / "outputs/baselines/qwen_zero_shot_valid.jsonl"
        )
        test_records = load_prediction_records(
            REPOSITORY_ROOT / "outputs/baselines/qwen_zero_shot_test.jsonl"
        )
        altered_test_records = copy.deepcopy(test_records)
        for record in altered_test_records:
            record["ground_truth"] = 1 - record["ground_truth"]

        original = evaluate_model(valid_records, test_records)
        altered = evaluate_model(valid_records, altered_test_records)
        self.assertEqual(original["threshold"], altered["threshold"])
        self.assertEqual(original["validation_cost"], altered["validation_cost"])

    def test_prediction_artifact_labels_must_match_the_frozen_split(self):
        records = load_prediction_records(
            REPOSITORY_ROOT / "outputs/baselines/qwen_zero_shot_test.jsonl"
        )
        records[0]["ground_truth"] = 1 - records[0]["ground_truth"]
        with self.assertRaises(ValueError):
            validate_ground_truth_alignment(records, "test")


if __name__ == "__main__":
    unittest.main()
