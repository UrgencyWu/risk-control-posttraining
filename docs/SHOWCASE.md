# Interactive Showcase

`docs/index.html` is a static, bilingual project showcase. It presents the
frozen C7 model results as an interactive chart without a server, model weights,
or external analytics. Visitors can switch between English and Simplified
Chinese, compare ROC-AUC / PR-AUC / Cost, and select a model for its threshold,
role, and evidence-backed interpretation.

## Publish with GitHub Pages

The repository includes `.github/workflows/deploy-showcase.yml`. In GitHub:

1. open **Settings → Pages**;
2. choose **GitHub Actions** as the build and deployment source;
3. push to `main` or `agent/readme-verified-results`, or run the workflow
   manually from the Actions tab.

The workflow deploys the contents of `docs/` as a static site. GitHub displays
the resulting public URL in the workflow deployment summary.

## Data contract

The values embedded in the showcase are intentionally limited to the committed
`outputs/c7_final_metrics.json` artifact. `tests/test_showcase_data.py` checks
the interactive data against that artifact, so displayed metrics cannot drift
from the leakage-safe final evaluation unnoticed.

To validate the display locally:

```bash
python -m unittest discover -s tests -v
node --check docs/assets/showcase.js
python -m http.server --directory docs 8000
```

Then open `http://localhost:8000` in a browser. No network access is required
for the finished showcase page.
