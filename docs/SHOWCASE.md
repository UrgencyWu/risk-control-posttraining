# 交互式展示页

在线地址：<https://urgencywu.github.io/risk-control-posttraining/>

`docs/index.html` 是无需后端、模型权重或外部分析服务的静态项目展示页。页面默认中文，右上角可切换英文。

## 内容结构

1. 项目摘要；
2. 研究问题；
3. 技术路线；
4. 实施细节；
5. 实验数据（多指标分组柱状图与冻结指标表）；
6. 实验结论；
7. 复现指引。

## 图表设计

实验数据区使用一张固定的分组柱状图，不提供指标切换按钮：

- 横轴依次为 `ROC-AUC`、`PR-AUC`、`Brier`、`ECE`、High-risk Recall、Low-risk Recall；
- 六个指标均处于 `0–1` 尺度，可在同一纵轴下直接显示原始数值；
- 每个指标组包含 Majority、Zero-shot、Logistic Regression、SFT、Multi-SFT 五种算法；
- 五种算法使用固定的灰、深灰、蓝、青、橙辅助色；
- 点击柱形、模型按钮或表格行可查看模型角色、冻结阈值与关键指标；
- `NLL` 与 `Cost` 因量纲差异不进入公共纵轴，但完整保留在精确表格中。

## 视觉原则

- 黑、白、灰构成主要视觉层级；
- 彩色仅用于算法区分与语义强调；
- 卡片、表格和路线使用简洁 1px 线条；
- 不使用渐变、装饰阴影或指标切换控件。

## 数据合同与测试

展示页指标仅来自 `outputs/c7_final_metrics.json`。`tests/test_showcase_data.py` 检查：

- 页面内嵌指标与 C7 冻结工件一致；
- 中文默认与英文切换；
- README 与页面章节顺序；
- 图表类型为 grouped bar，且不存在指标切换按钮；
- 固定六个图表指标、五种算法系列与静态 README 柱状图；
- 黑白主视觉与算法辅助色。

## 本地验证

```bash
python -m unittest discover -s tests -v
node --check docs/assets/showcase.js
python -m http.server --directory docs 8000
```

然后访问 `http://localhost:8000`。

## GitHub Pages 部署

仓库包含 `.github/workflows/deploy-showcase.yml`。将 Pages 发布来源设置为 GitHub Actions 后，向 `main` 推送 `docs/` 修改即可部署。
