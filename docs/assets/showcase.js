(() => {
  const dataset = JSON.parse(document.getElementById("showcase-data").textContent);
  const state = { language: "zh", selectedModel: "sft-seed7" };
  const CHART_METRICS = [
    ["roc_auc", "ROC-AUC"],
    ["pr_auc", "PR-AUC"],
    ["brier", "Brier"],
    ["ece", "ECE"],
    ["high_risk_recall", "HR Recall"],
    ["low_risk_recall", "LR Recall"]
  ];
  const SERIES_COLORS = {
    majority: "var(--series-majority)",
    "zero-shot": "var(--series-zero-shot)",
    "logistic-regression": "var(--series-lr)",
    "sft-seed7": "var(--series-sft)",
    "multi-sft": "var(--series-multi)"
  };

  const copy = {
    zh: {
      documentTitle: "大语言模型风控后训练 | 交互式展示", skipLink: "跳至正文", brand: "风控后训练研究", navSummary: "摘要", navQuestion: "研究问题", navRoute: "技术路线", navData: "实验数据", navReproduce: "复现",
      summaryEyebrow: "项目摘要 · 信用风险后训练", summaryTitle: "SFT 学到了有效风险排序，偏好优化暴露出明确的方法边界。", summaryCopy: "项目以 Qwen3.5-4B 和 German Credit 为主实验对象，系统比较 LoRA SFT、DPO/SimPO 与 Logistic Regression，并用无测试泄漏的成本敏感评测区分真实收益与标签先验偏移。",
      summaryStatSft: "SFT ROC-AUC 增量", summaryStatBaseline: "LR 与 SFT", summaryStatCost: "最佳 SFT Cost", summaryStatPreference: "主 DPO/SimPO 实验未超过 SFT",
      questionEyebrow: "研究问题", questionTitle: "后训练能否在强传统基线面前产生真实、稳定的风险判别价值？", questionOne: "指令模型能否从自然语言形式的表格记录中学习可泛化的风险排序？", questionTwo: "SFT 与 DPO/SimPO 改善的是条件判别能力，还是只改变标签先验？", questionThree: "失败根因来自数据、优化、阈值，还是目标函数与短标签任务的错配？",
      routeEyebrow: "技术路线", routeTitle: "从数据协议到机制审计的完整实验链", route1Title: "仓库与数据审计", route2Title: "统一数据协议", route3Title: "确定性数据构造", route4Title: "传统与 LLM 基线", route5Title: "三随机种子 LoRA SFT", route6Title: "DPO / SimPO", route7Title: "验证集冻结业务阈值", route8Title: "C7 评测与机制审计",
      implementationEyebrow: "实施细节", implementationTitle: "可复现的数据、训练与评测实现", implDataTitle: "数据治理", implDataCopy: "German Credit 冻结为 700/100/200；通过 Schema、manifest、V1–V7 与 SHA-256 检查保证确定性。", implTrainTitle: "LoRA SFT", implTrainCopy: "Qwen3.5-4B 使用 response-only loss、LoRA r=16、alpha=32、dropout=0.05，训练 5 个 epoch、三个随机种子。", implPreferenceTitle: "偏好优化", implPreferenceCopy: "构造 Oracle、Hard 与多数据集 Preference，运行六组主 DPO/SimPO 实验及成本敏感对照。", implEvalTitle: "无泄漏评测", implEvalCopy: "统一计算 AUC、校准、召回、混淆矩阵与 Cost；阈值只从验证预测中选择。",
      dataEyebrow: "实验数据 · German Credit 测试集 N = 200", dataTitle: "多指标分组柱状图与冻结指标表", dataCopy: "柱状图固定展示六个 0–1 尺度指标；每个指标组包含五种算法。NLL 与 Cost 保留在下方精确表格中。", chartCaption: "ROC-AUC、PR-AUC 与 Recall 越高越好；Brier 与 ECE 越低越好。柱顶与悬停提示显示原始数值。", selectionLabel: "当前模型", tableModel: "模型", tableHighRecall: "高风险召回", tableLowRecall: "低风险召回", tableThreshold: "阈值",
      conclusionEyebrow: "实验结论", conclusionTitle: "正向结果、诚实基线与方法边界", conclusion1Title: "SFT ROC-AUC 提升 +0.232", conclusion1Copy: "SFT 是唯一持续产生有效风险排序的后训练方法。", conclusion2Title: "LR 0.757 仍高于 SFT 0.747", conclusion2Copy: "项目声称接近，不声称稳定超越。", conclusion3Title: "成本阈值有效但不能替代排序", conclusion3Copy: "验证集阈值降低 FN 成本，AUC 与校准指标防止伪改进。", conclusion4Title: "6 / 6 组 DPO/SimPO 未超过 SFT", conclusion4Copy: "短标签偏好目标主要移动全局标签先验。",
      reproEyebrow: "复现指引", reproTitle: "先复现冻结指标，再按需运行完整 GPU 链路", cpuTitle: "CPU：复现最终指标", gpuTitle: "GPU：训练与推理", footerText: "仅用于研究、教育与作品展示，不适用于真实高影响决策。", footerLanguage: "English README",
      roles: { majority: "类别不平衡下界", "zero-shot": "未适配 LLM 基线", "logistic-regression": "强非 LLM 基线", "sft-seed7": "冻结 SFT Checkpoint", "multi-sft": "多数据集迁移实验" },
      notes: { majority: "67.5% Accuracy 来自类别不平衡，但漏掉全部高风险样本。", "zero-shot": "排序接近随机，并严重偏向 high risk。", "logistic-regression": "排序和概率质量最强。", "sft-seed7": "按验证集 PR-AUC 规则选择，冻结测试 Cost 最低。", "multi-sft": "提升跨数据集覆盖，但 German 上存在负迁移。" },
      shortNames: { majority: "Majority", "zero-shot": "Zero-shot", "logistic-regression": "LR", "sft-seed7": "SFT", "multi-sft": "Multi-SFT" },
      labels: { threshold: "阈值", cost: "Cost", roc: "ROC-AUC", highRecall: "高风险召回" }
    },
    en: {
      documentTitle: "Risk-Control Post-Training | Interactive Showcase", skipLink: "Skip to content", brand: "Risk-Control Post-Training", navSummary: "Summary", navQuestion: "Questions", navRoute: "Route", navData: "Data", navReproduce: "Reproduce",
      summaryEyebrow: "Project Summary · Credit-Risk Post-Training", summaryTitle: "SFT learned useful risk ranking; preference optimization exposed a clear method boundary.", summaryCopy: "Using Qwen3.5-4B and German Credit, the project compares LoRA SFT, DPO/SimPO, and Logistic Regression under leakage-safe cost-sensitive evaluation.",
      summaryStatSft: "SFT ROC-AUC gain", summaryStatBaseline: "LR vs SFT", summaryStatCost: "Best SFT Cost", summaryStatPreference: "Principal DPO/SimPO runs below SFT",
      questionEyebrow: "Research Questions", questionTitle: "Does post-training create real and stable value against a strong statistical baseline?", questionOne: "Can an instruction model learn generalizable ranking from natural-language tabular records?", questionTwo: "Do SFT and DPO/SimPO improve conditional discrimination or only label priors?", questionThree: "Are failures caused by data, optimization, thresholds, or objective-task mismatch?",
      routeEyebrow: "Technical Route", routeTitle: "A complete chain from data contracts to mechanism audits", route1Title: "Repository and data audit", route2Title: "Unified data contract", route3Title: "Deterministic construction", route4Title: "Classical and LLM baselines", route5Title: "Three-seed LoRA SFT", route6Title: "DPO / SimPO", route7Title: "Freeze validation thresholds", route8Title: "C7 evaluation and audit",
      implementationEyebrow: "Implementation Details", implementationTitle: "Reproducible data, training, and evaluation", implDataTitle: "Data governance", implDataCopy: "German Credit is frozen to 700/100/200 with schema, manifest, V1–V7, and SHA-256 checks.", implTrainTitle: "LoRA SFT", implTrainCopy: "Qwen3.5-4B uses response-only loss, LoRA r=16, alpha=32, dropout=0.05, five epochs, and three seeds.", implPreferenceTitle: "Preference optimization", implPreferenceCopy: "Oracle, hard, and multi-dataset pairs support six principal DPO/SimPO runs and cost-sensitive controls.", implEvalTitle: "Leakage-safe evaluation", implEvalCopy: "AUC, calibration, recall, confusion matrices, and Cost are computed with validation-selected thresholds.",
      dataEyebrow: "Experimental Data · German Credit Test N = 200", dataTitle: "Grouped multi-metric bar chart and frozen table", dataCopy: "The chart fixes six metrics on a common 0–1 scale; each metric group contains all five algorithms. NLL and Cost remain in the precise table below.", chartCaption: "Higher is better for ROC-AUC, PR-AUC, and Recall; lower is better for Brier and ECE. Labels and tooltips show raw values.", selectionLabel: "Selected model", tableModel: "Model", tableHighRecall: "High-risk recall", tableLowRecall: "Low-risk recall", tableThreshold: "Threshold",
      conclusionEyebrow: "Experimental Conclusions", conclusionTitle: "Positive result, honest baseline, and method boundary", conclusion1Title: "SFT ROC-AUC improves by +0.232", conclusion1Copy: "SFT is the only post-training method that consistently creates useful ranking.", conclusion2Title: "LR 0.757 remains above SFT 0.747", conclusion2Copy: "The project claims competitiveness, not stable superiority.", conclusion3Title: "Cost thresholds help but do not replace ranking", conclusion3Copy: "Validation thresholds reduce FN cost while AUC and calibration prevent trivial gains.", conclusion4Title: "6 / 6 DPO/SimPO runs remain below SFT", conclusion4Copy: "Short-label preference objectives mainly shift the global label prior.",
      reproEyebrow: "Reproduction Guide", reproTitle: "Reproduce frozen metrics first, then run the GPU workflow if needed", cpuTitle: "CPU: reproduce final metrics", gpuTitle: "GPU: training and inference", footerText: "For research, education, and portfolio use only; not for high-impact decisions.", footerLanguage: "中文 README",
      roles: { majority: "Class-imbalance lower bound", "zero-shot": "Unadapted LLM baseline", "logistic-regression": "Strong non-LLM baseline", "sft-seed7": "Frozen SFT checkpoint", "multi-sft": "Multi-dataset transfer experiment" },
      notes: { majority: "67.5% Accuracy comes from imbalance while every high-risk case is missed.", "zero-shot": "Ranking is near random and predictions are strongly biased toward high risk.", "logistic-regression": "Strongest ranking and probability quality.", "sft-seed7": "Selected by validation PR-AUC and has the lowest frozen test Cost.", "multi-sft": "Improves coverage but shows negative transfer on German." },
      shortNames: { majority: "Majority", "zero-shot": "Zero-shot", "logistic-regression": "LR", "sft-seed7": "SFT", "multi-sft": "Multi-SFT" },
      labels: { threshold: "Threshold", cost: "Cost", roc: "ROC-AUC", highRecall: "High-risk recall" }
    }
  };

  const svgNS = "http://www.w3.org/2000/svg";
  const chart = document.getElementById("metric-chart");
  const tableBody = document.getElementById("metric-table-body");
  const modelButtons = document.getElementById("model-buttons");
  const legend = document.getElementById("chart-legend");
  const modelById = (id) => dataset.models.find((model) => model.id === id);
  const strings = () => copy[state.language];
  const createSvgElement = (name, attrs = {}) => {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  };
  const displayName = (model) => strings().shortNames[model.id] || model.name;

  function setLanguageText() {
    const text = strings();
    document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
    document.title = text.documentTitle;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const value = text[node.dataset.i18n];
      if (value) node.textContent = value;
    });
    document.querySelectorAll(".language-button").forEach((button) => {
      const active = button.dataset.language === state.language;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelector(".site-footer a").href = state.language === "zh" ? "../README.en.md" : "../README.md";
  }

  function renderLegend() {
    legend.replaceChildren();
    dataset.models.forEach((model) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = SERIES_COLORS[model.id];
      const label = document.createElement("span");
      label.textContent = displayName(model);
      item.append(swatch, label);
      legend.append(item);
    });
  }

  function renderChart() {
    const width = Math.max(chart.clientWidth || 900, 520);
    const height = width < 700 ? 500 : 540;
    const margin = { top: 28, right: 20, bottom: 92, left: 52 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const groupWidth = plotWidth / CHART_METRICS.length;
    const innerWidth = Math.min(groupWidth * .78, 115);
    const gap = 2;
    const barWidth = Math.max(5, (innerWidth - gap * (dataset.models.length - 1)) / dataset.models.length);

    chart.replaceChildren();
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const title = createSvgElement("title");
    title.textContent = strings().dataTitle;
    chart.append(title);

    for (let tick = 0; tick <= 5; tick += 1) {
      const value = tick / 5;
      const y = margin.top + plotHeight * (1 - value);
      chart.append(createSvgElement("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": ".13" }));
      const label = createSvgElement("text", { x: margin.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": "11", "fill-opacity": ".65" });
      label.textContent = value.toFixed(1);
      chart.append(label);
    }
    chart.append(createSvgElement("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotHeight, stroke: "currentColor", "stroke-width": "1.2" }));
    chart.append(createSvgElement("line", { x1: margin.left, x2: width - margin.right, y1: margin.top + plotHeight, y2: margin.top + plotHeight, stroke: "currentColor", "stroke-width": "1.2" }));

    CHART_METRICS.forEach(([metric, labelText], groupIndex) => {
      const groupStart = margin.left + groupIndex * groupWidth + (groupWidth - innerWidth) / 2;
      dataset.models.forEach((model, modelIndex) => {
        const value = model[metric];
        const x = groupStart + modelIndex * (barWidth + gap);
        const barHeight = plotHeight * Math.max(0, Math.min(1, value));
        const y = margin.top + plotHeight - barHeight;
        const selected = model.id === state.selectedModel;
        const rect = createSvgElement("rect", {
          x, y, width: barWidth, height: barHeight, rx: 2,
          fill: SERIES_COLORS[model.id], opacity: selected ? "1" : ".78",
          stroke: selected ? "#111111" : "none", "stroke-width": selected ? "1.4" : "0",
          tabindex: "0", role: "button", "aria-label": `${displayName(model)} ${labelText}: ${value.toFixed(3)}`
        });
        const tooltip = createSvgElement("title");
        tooltip.textContent = `${displayName(model)} · ${labelText}: ${value.toFixed(4)}`;
        rect.append(tooltip);
        rect.addEventListener("click", () => selectModel(model.id));
        rect.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectModel(model.id); }
        });
        chart.append(rect);
      });
      const xLabel = createSvgElement("text", { x: margin.left + groupIndex * groupWidth + groupWidth / 2, y: margin.top + plotHeight + 27, "text-anchor": "middle", fill: "currentColor", "font-size": width < 700 ? "10" : "12", "font-weight": "650" });
      xLabel.textContent = labelText;
      chart.append(xLabel);
    });
  }

  function renderSelection() {
    const model = modelById(state.selectedModel);
    const text = strings();
    document.getElementById("selected-model-name").textContent = model.name;
    document.getElementById("selected-model-role").textContent = text.roles[model.id];
    document.getElementById("selected-model-note").textContent = text.notes[model.id];
    const values = [[text.labels.threshold, model.threshold.toFixed(2)], [text.labels.cost, model.cost], [text.labels.roc, model.roc_auc.toFixed(3)], [text.labels.highRecall, model.high_risk_recall.toFixed(3)]];
    const container = document.getElementById("selected-model-metrics");
    container.replaceChildren();
    values.forEach(([label, value]) => {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label; dd.textContent = value; wrapper.append(dt, dd); container.append(wrapper);
    });
  }

  function renderButtons() {
    modelButtons.replaceChildren();
    dataset.models.forEach((model) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "model-button";
      button.classList.toggle("is-selected", model.id === state.selectedModel);
      button.setAttribute("aria-pressed", String(model.id === state.selectedModel));
      button.textContent = displayName(model);
      button.addEventListener("click", () => selectModel(model.id));
      modelButtons.append(button);
    });
  }

  function renderTable() {
    tableBody.replaceChildren();
    dataset.models.forEach((model) => {
      const row = document.createElement("tr");
      row.tabIndex = 0;
      row.classList.toggle("is-selected", model.id === state.selectedModel);
      const values = [model.name, model.roc_auc.toFixed(3), model.pr_auc.toFixed(3), model.nll.toFixed(3), model.brier.toFixed(3), model.ece.toFixed(3), String(model.cost), model.high_risk_recall.toFixed(3), model.low_risk_recall.toFixed(3), model.threshold.toFixed(2)];
      values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
      row.addEventListener("click", () => selectModel(model.id));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectModel(model.id); } });
      tableBody.append(row);
    });
  }

  function selectModel(id) {
    state.selectedModel = id;
    renderChart(); renderSelection(); renderButtons(); renderTable();
  }

  function renderAll() {
    setLanguageText(); renderLegend(); renderChart(); renderSelection(); renderButtons(); renderTable();
  }

  document.querySelectorAll(".language-button").forEach((button) => {
    button.addEventListener("click", () => { state.language = button.dataset.language; renderAll(); });
  });
  window.addEventListener("resize", renderChart);
  renderAll();
})();
