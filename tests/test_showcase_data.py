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
        self.javascript = (REPOSITORY_ROOT / "docs/assets/showcase.js").read_text(
            encoding="utf-8"
        )
        self.stylesheet = (REPOSITORY_ROOT / "docs/assets/showcase.css").read_text(
            encoding="utf-8"
        )
        self.readme_zh = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.readme_en = (REPOSITORY_ROOT / "README.en.md").read_text(
            encoding="utf-8"
        )
        self.readme_legacy_zh = (
            REPOSITORY_ROOT / "README.zh-CN.md"
        ).read_text(encoding="utf-8")
        self.static_chart = REPOSITORY_ROOT / "docs/assets/metric_grouped_bars.svg"
        self.static_chart_text = self.static_chart.read_text(encoding="utf-8")

        match = re.search(
            r'<script id="showcase-data" type="application/json">\s*(.*?)\s*</script>',
            self.html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "showcase data payload is missing")
        self.showcase_data = json.loads(match.group(1))
        self.c7_metrics = json.loads(
            (REPOSITORY_ROOT / "outputs/c7_final_metrics.json").read_text(
                encoding="utf-8"
            )
        )

    def test_showcase_metrics_match_the_frozen_c7_artifact(self):
        self.assertEqual(
            set(SHOWCASE_MODEL_TO_C7),
            {model["id"] for model in self.showcase_data["models"]},
        )
        for model in self.showcase_data["models"]:
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
                with self.subTest(model=model["id"], metric=metric):
                    self.assertAlmostEqual(source[metric], model[metric], places=4)

    def test_chinese_is_default_and_english_switch_remains(self):
        self.assertIn('<html lang="zh-CN">', self.html)
        self.assertIn(
            'class="language-button is-active" data-language="zh"', self.html
        )
        self.assertIn('class="language-button" data-language="en"', self.html)
        self.assertIn('const state = { language: "zh"', self.javascript)
        self.assertEqual(self.readme_zh, self.readme_legacy_zh)

    def test_sections_follow_required_order(self):
        html_ids = (
            'id="summary"',
            'id="research-question"',
            'id="technical-route"',
            'id="implementation"',
            'id="experiment-data"',
            'id="conclusions"',
            'id="reproduction"',
        )
        positions = [self.html.index(item) for item in html_ids]
        self.assertEqual(positions, sorted(positions))

        zh = (
            "## 项目摘要",
            "## 研究问题",
            "## 技术路线",
            "## 实施细节",
            "## 实验数据",
            "## 实验结论",
            "## 复现指引",
        )
        en = (
            "## Project Summary",
            "## Research Questions",
            "## Technical Route",
            "## Implementation Details",
            "## Experimental Data",
            "## Experimental Conclusions",
            "## Reproduction Guide",
        )
        self.assertEqual(
            [self.readme_zh.index(section) for section in zh],
            sorted(self.readme_zh.index(section) for section in zh),
        )
        self.assertEqual(
            [self.readme_en.index(section) for section in en],
            sorted(self.readme_en.index(section) for section in en),
        )

    def test_experimental_data_uses_fixed_grouped_bar_chart_and_table(self):
        self.assertIn('data-chart-type="grouped-bar"', self.html)
        self.assertNotIn('id="metric-select"', self.html)
        self.assertIn('id="metric-table-body"', self.html)
        self.assertIn("const CHART_METRICS = [", self.javascript)
        for metric in (
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
            "high_risk_recall",
            "low_risk_recall",
        ):
            self.assertIn(f'key: "{metric}"', self.javascript)
        self.assertIn('createSvgElement("rect"', self.javascript)
        self.assertNotIn('createSvgElement("polyline"', self.javascript)
        self.assertTrue(self.static_chart.is_file())
        self.assertIn(
            "./docs/assets/metric_grouped_bars.svg", self.readme_zh
        )
        self.assertIn(
            "./docs/assets/metric_grouped_bars.svg", self.readme_en
        )

    def test_chart_localizes_metric_labels_and_marks_direction(self):
        for key in (
            "metricRoc",
            "metricPr",
            "metricBrier",
            "metricEce",
            "metricHighRecall",
            "metricLowRecall",
            "higherBetter",
            "lowerBetter",
        ):
            self.assertIn(f"{key}:", self.javascript)
        self.assertIn('metric.direction === "up"', self.javascript)
        self.assertIn("↑", self.javascript)
        self.assertIn("↓", self.javascript)
        self.assertIn("↑", self.static_chart_text)
        self.assertIn("↓", self.static_chart_text)

    def test_chart_displays_selected_values_and_keeps_zero_values_visible(self):
        self.assertIn("const minVisibleHeight = 2.5;", self.javascript)
        self.assertIn(
            "Math.max(minVisibleHeight, rawHeight)", self.javascript
        )
        self.assertIn('const valueLabel = createSvgElement("text"', self.javascript)
        self.assertIn("rawValue.toFixed(3)", self.javascript)
        self.assertIn('height="2.5"', self.static_chart_text)
        self.assertIn("0.0000", self.static_chart_text)

    def test_chart_has_immediate_tooltip_and_interactive_legend(self):
        self.assertIn('tooltip.className = "chart-tooltip"', self.javascript)
        self.assertIn('rect.addEventListener("pointerenter"', self.javascript)
        self.assertIn('button.className = "legend-item"', self.javascript)
        self.assertIn('button.setAttribute("aria-pressed"', self.javascript)
        self.assertIn(".chart-tooltip", self.stylesheet)
        self.assertIn(".legend-item.is-selected", self.stylesheet)

    def test_monochrome_system_and_distinct_algorithm_colors(self):
        for token in (
            "--ink: #111111;",
            "--paper: #ffffff;",
            "--canvas: #f4f4f1;",
        ):
            self.assertIn(token, self.stylesheet)
        for token in (
            "--series-majority",
            "--series-zero-shot",
            "--series-lr",
            "--series-sft",
            "--series-multi",
        ):
            self.assertIn(token, self.stylesheet)
        self.assertIn("overflow: hidden;", self.stylesheet)
        self.assertNotIn("gradient", self.stylesheet)

    def test_live_showcase_and_reproduction_links_are_exposed(self):
        url = "https://urgencywu.github.io/risk-control-posttraining/"
        self.assertIn(url, self.readme_zh)
        self.assertIn(url, self.readme_en)
        clone = "git clone https://github.com/UrgencyWu/risk-control-posttraining.git"
        self.assertIn(clone, self.readme_zh)
        self.assertIn(clone, self.readme_en)


if __name__ == "__main__":
    unittest.main()
