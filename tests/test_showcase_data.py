import json
import re
import unittest

from src.evaluation.c7_final import REPOSITORY_ROOT


SHOWCASE_MODEL_TO_C7 = {
    "majority": "Majority",
    "zero-shot": "Qwen-ZeroShot",
    "logistic-regression": "LogisticRegression",
    "sft-seed7": "SFT-seed7",
    "multi-sft": "SFT-multi",
}


class ShowcaseDataTests(unittest.TestCase):
    def setUp(self):
        self.html = (REPOSITORY_ROOT / "docs/index.html").read_text(encoding="utf-8")
        self.readme_en = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.readme_zh = (REPOSITORY_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        data_match = re.search(
            r'<script id="showcase-data" type="application/json">\s*(.*?)\s*</script>',
            self.html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(data_match, "showcase data payload is missing")
        self.showcase_data = json.loads(data_match.group(1))
        self.c7_metrics = json.loads(
            (REPOSITORY_ROOT / "outputs/c7_final_metrics.json").read_text(encoding="utf-8")
        )

    def test_showcase_metrics_match_the_frozen_c7_artifact(self):
        self.assertEqual(
            set(SHOWCASE_MODEL_TO_C7),
            {model["id"] for model in self.showcase_data["models"]},
        )
        for model in self.showcase_data["models"]:
            with self.subTest(model=model["id"]):
                source = self.c7_metrics[SHOWCASE_MODEL_TO_C7[model["id"]]]
                for metric in (
                    "roc_auc",
                    "pr_auc",
                    "nll",
                    "brier",
                    "ece",
                    "cost",
                    "threshold",
                    "high_risk_recall",
                    "low_risk_recall",
                ):
                    self.assertAlmostEqual(source[metric], model[metric], places=4)

    def test_showcase_exposes_bilingual_and_interactive_controls(self):
        self.assertIn('data-language="en"', self.html)
        self.assertIn('data-language="zh"', self.html)
        self.assertIn('id="metric-select"', self.html)
        self.assertIn('id="metric-chart"', self.html)
        self.assertIn('src="./assets/showcase.js"', self.html)

    def test_showcase_leads_with_three_core_findings(self):
        findings = (
            ('id="finding-sft"', "+0.232"),
            ('id="finding-baseline"', "0.757 <span>vs</span> 0.747"),
            ('id="finding-preference"', "6 / 6"),
        )
        results_position = self.html.index('id="results-heading"')
        for card_id, quantitative_text in findings:
            with self.subTest(card=card_id):
                self.assertIn(card_id, self.html)
                self.assertIn(quantitative_text, self.html)
                self.assertLess(self.html.index(card_id), results_position)

    def test_readmes_lead_with_question_findings_and_contributions(self):
        english_sections = (
            "## Research Question",
            "## Four Quantitative Findings",
            "## Project Contributions",
            "## Evidence and Implementation Status",
        )
        chinese_sections = (
            "## 研究问题",
            "## 四条量化结论",
            "## 项目贡献",
            "## 证据与实现状态",
        )

        for document, sections in (
            (self.readme_en, english_sections),
            (self.readme_zh, chinese_sections),
        ):
            positions = [document.index(section) for section in sections]
            self.assertEqual(positions, sorted(positions))

        self.assertGreater(
            self.readme_en.index("## Interactive Showcase and Deployment"),
            self.readme_en.index("## Quick Start"),
        )
        self.assertGreater(
            self.readme_zh.index("## 交互式展示与部署"),
            self.readme_zh.index("## 快速复现"),
        )


if __name__ == "__main__":
    unittest.main()
