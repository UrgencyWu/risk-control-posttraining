# 交互式展示页

在线地址：<https://urgencywu.github.io/risk-control-posttraining/>

`docs/index.html` 是无需后端、模型权重或外部分析服务的静态项目展示页。页面默认使用中文，右上角可切换为英文。

## 内容结构

展示页与默认中文 README 使用相同的信息顺序：

1. 项目摘要；
2. 研究问题；
3. 技术路线；
4. 实施细节；
5. 实验数据（交互式折线图与指标表）；
6. 实验结论；
7. 复现指引。

该顺序先回答“项目研究得到了什么”，再展开证据、实现与复现细节。

## 视觉与交互

- 黑、白、灰构成主要视觉层级；
- 蓝色只用于当前选中数据、正向结果和可操作入口；
- 橙色仅用于方法边界与负结果提示；
- 卡片、表格和路线均使用简洁的 1px 线条，避免大面积渐变和装饰；
- 实验数据区可在 ROC-AUC、PR-AUC 与 Cost 之间切换；
- 折线连接离散模型结果，不表示连续训练轨迹；
- 点击折线数据点、模型按钮或表格行可查看实验角色、阈值与解释；
- 页面在桌面与移动端均保持可用。

## 数据合同

展示页中的模型指标仅来自已提交的 `outputs/c7_final_metrics.json`。`tests/test_showcase_data.py` 会检查：

- 页面内嵌指标与 C7 冻结工件一致；
- 中文为默认语言并保留英文切换；
- 页面章节顺序符合项目叙事要求；
- 实验数据区域使用折线图和表格；
- README 中英文版本使用相同章节顺序；
- 静态 README 折线图存在并被引用。

因此页面将“冻结量化证据”和“范围受限的研究解释”分开维护，避免展示文案与实验工件漂移。

## 本地验证

```bash
python -m unittest discover -s tests -v
node --check docs/assets/showcase.js
python -m http.server --directory docs 8000
```

然后访问 `http://localhost:8000`。完成后的展示页不依赖网络请求。

## GitHub Pages 部署

仓库包含 `.github/workflows/deploy-showcase.yml`：

1. 在 GitHub 打开 **Settings → Pages**；
2. 将发布来源设置为 **GitHub Actions**；
3. 向 `main` 推送经过审查的 `docs/` 修改，或在 Actions 页面手动运行部署工作流。

工作流会将 `docs/` 目录发布为静态站点，并在部署摘要中给出公开地址。
