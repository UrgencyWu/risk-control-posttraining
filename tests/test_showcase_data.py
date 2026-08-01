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
        self.javascript = (REPOSITORY_ROOT / "docs/assets/showcase.js").read_text(encoding="utf-8")
        self.stylesheet = (REPOSITORY_ROOT / "docs/assets/showcase.css").read_text(encoding="utf-8")
        self.readme_zh = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.readme_en = (REPOSITORY_ROOT / "README.en.md").read_text(encoding="utf-8")
        self.readme_legacy_zh = (REPOSITORY_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        self.static_chart = REPOSITORY_ROOT / "docs/assets/roc_auc_trend.svg"

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

    def test_chinese_is_the_default_language_with_english_switches(self):
        self.assertIn('<html lang="zh-CN">', self.html)
        self.assertIn('class="language-button is-active" data-language="zh" aria-pressed="true"', self.html)
        self.assertIn('class="language-button" data-language="en" aria-pressed="false"', self.html)
        self.assertIn('const state = { language: "zh"', self.javascript)

        self.assertTrue(self.readme_zh.startswith("# 大语言模型风控后训练"))
        self.assertIn('<a href="./README.md"><strong>中文</strong></a>', self.readme_zh)
        self.assertIn('<a href="./README.en.md">English</a>', self.readme_zh)
        self.assertIn('<a href="./README.en.md"><strong>English</strong></a>', self.readme_en)
        self.assertIn('<a href="./README.md">中文</a>', self.readme_en)
        self.assertEqual(self.readme_zh, self.readme_legacy_zh)

    def test_showcase_sections_follow_the_required_order(self):
        ordered_ids = (
            'id="summary"',
            'id="research-question"',
            'id="technical-route"',
            'id="implementation"',
            'id="experiment-data"',
            'id="conclusions"',
            'id="reproduction"',
        )
        positions = [self.html.index(section_id) for section_id in ordered_ids]
        self.assertEqual(positions, sorted(positions))

    def test_readmes_follow_the_required_order(self):
        chinese_sections = (
            "## 项目摘要",
            "## 研究问题",
            "## 技术路线",
            "## 实施细节",
            "## 实验数据",
            "## 实验结论",
            "## 复现指引",
        )
        english_sections = (
            "## Project Summary",
            "## Research Questions",
            "## Technical Route",
            "## Implementation Details",
            "## Experimental Data",
            "## Experimental Conclusions",
            "## Reproduction Guide",
        )

        for document, sections in (
            (self.readme_zh, chinese_sections),
            (self.readme_en, english_sections),
        ):
            positions = [document.index(section) for section in sections]
            self.assertEqual(positions, sorted(positions))

    def test_experimental_data_uses_line_charts_and_tables(self):
        self.assertIn('id="metric-chart"', self.html)
        self.assertIn('data-chart-type="line"', self.html)
        self.assertIn('id="metric-table-body"', self.html)
        self.assertIn('createSvgElement("polyline"', self.javascript)
        self.assertTrue(self.static_chart.is_file())
        self.assertIn("./docs/assets/roc_auc_trend.svg", self.readme_zh)
        self.assertIn("./docs/assets/roc_auc_trend.svg", self.readme_en)

    def test_monochrome_visual_system_uses_color_only_as_an_accent(self):
        self.assertIn("--ink: #111111;", self.stylesheet)
        self.assertIn("--paper: #ffffff;", self.stylesheet)
        self.assertIn("--canvas: #f4f4f1;", self.stylesheet)
        self.assertIn("--accent: #2563eb;", self.stylesheet)
        self.assertIn("border: 1px solid var(--line);", self.stylesheet)
        self.assertNotIn("radial-gradient", self.stylesheet)

    def test_live_showcase_and_reproduction_links_are_exposed(self):
        live_url = "https://urgencywu.github.io/risk-control-posttraining/"
        self.assertIn(live_url, self.readme_zh)
        self.assertIn(live_url, self.readme_en)
        self.assertIn("git clone https://github.com/UrgencyWu/risk-control-posttraining.git", self.readme_zh)
        self.assertIn("git clone https://github.com/UrgencyWu/risk-control-posttraining.git", self.readme_en)


if __name__ == "__main__":
    unittest.main()
