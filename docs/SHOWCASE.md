# Interactive Showcase

`docs/index.html` is a static, bilingual presentation of the project's frozen research findings. It requires no backend, model weights, external analytics, or network access after the files are loaded.

## What the first screen communicates

The showcase leads with the three conclusions that define the project:

1. **SFT effect:** German Credit ROC-AUC improves by `+0.232`, from `0.515` zero-shot to `0.747` after LoRA SFT.
2. **Honest classical comparison:** Logistic Regression remains slightly higher on ranking, `0.757` versus SFT's `0.747`; the project does not claim stable superiority.
3. **Preference-optimization boundary:** all `6 / 6` principal DPO/SimPO variants fail to exceed SFT, and the mechanism audit links the failures to global label-prior movement or decision collapse.

After these headline findings, visitors can switch between English and Simplified Chinese, compare ROC-AUC / PR-AUC / Cost, select a model, inspect its frozen validation-selected threshold, and follow the evidence chain behind the conclusions.

## Data contract

The interactive model values are limited to the committed `outputs/c7_final_metrics.json` artifact. `tests/test_showcase_data.py` checks every embedded model metric against that artifact and also verifies that the three headline finding cards remain present.

The page therefore separates two kinds of content:

- **frozen quantitative evidence**, checked against committed artifacts;
- **bounded interpretation**, written to avoid claiming that one favorable seed establishes statistical superiority or that DPO/SimPO is ineffective outside the tested setting.

## Validate locally

```bash
python -m unittest discover -s tests -v
node --check docs/assets/showcase.js
python -m http.server --directory docs 8000
```

Then open `http://localhost:8000` in a browser. No network access is required for the finished page.

## Publish with GitHub Pages

Deployment is intentionally documented after the research content because publication is an implementation detail, not the project result.

The repository includes `.github/workflows/deploy-showcase.yml`. In GitHub:

1. open **Settings → Pages**;
2. choose **GitHub Actions** as the build and deployment source;
3. push the approved changes to `main`, or run the deployment workflow manually from the Actions tab.

The workflow deploys the contents of `docs/` as a static site. GitHub displays the public URL in the workflow deployment summary.
