(() => {
  const dataset = JSON.parse(document.getElementById("showcase-data").textContent);
  const state = { language: "en", metric: "roc_auc", selectedModel: "sft-seed7" };

  const copy = {
    en: {
      eyebrow: "APPLIED LLM POST-TRAINING · CREDIT RISK",
      heroTitle: "Evidence-led post-training, not a benchmark claim.",
      heroCopy: "An end-to-end study of how LoRA SFT, preference optimization, and validation-only decision rules behave on small tabular credit-risk data.",
      readmeLink: "Read the technical README",
      protocolLink: "Review the evaluation protocol",
      evidenceLabel: "FROZEN RESULT",
      evidenceCaption: "Qwen3.5-4B zero-shot to LoRA SFT ROC-AUC",
      factCost: "SFT test Cost",
      factThreshold: "Threshold source",
      factThresholdValue: "Validation only",
      resultsEyebrow: "GERMAN CREDIT · TEST N = 200",
      resultsTitle: "Explore the frozen final metrics",
      metricLabel: "Compare by",
      chartCaption: "Select a metric to compare the committed prediction artifacts. Higher is better for AUC metrics; lower is better for Cost.",
      selectionLabel: "SELECTED MODEL",
      tableModel: "Model",
      tableThreshold: "Threshold",
      pathEyebrow: "EXPERIMENT PATH",
      pathTitle: "A complete result includes its boundaries",
      pathCopy: "The project preserves successful adaptation, strong baselines, and negative preference-optimization results as one evidence chain.",
      protocolEyebrow: "EVALUATION GUARDRAIL",
      protocolTitle: "The operating point is frozen before test evaluation.",
      protocolCopy: "Every threshold is selected by minimizing 5 × false negatives + 1 × false positives on validation predictions. The committed test prediction artifact is then evaluated once at that frozen threshold.",
      protocolReadMore: "Read the complete protocol →",
      flowValid: "Validation predictions",
      flowThreshold: "Freeze threshold",
      flowTest: "Test metrics once",
      footerText: "Research, education, and portfolio demonstration only. Not for high-impact decisions.",
      footerLanguage: "中文 README",
      modelPicker: "Choose a model",
      metricsCaption: "Frozen C7 final metrics for all evaluated models",
      chartComparison: "comparison",
      protocolDiagram: "Validation predictions select a threshold, which is frozen and applied to test predictions for final metrics.",
      metrics: { roc_auc: "ROC-AUC", pr_auc: "PR-AUC", cost: "Cost" },
      lowerBetter: "Lower is better",
      higherBetter: "Higher is better",
      labels: { role: "Role", threshold: "Threshold", cost: "Cost", highRecall: "High-risk recall" },
      roles: {
        majority: "Sanity-check lower bound",
        "zero-shot": "Unadapted LLM baseline",
        "logistic-regression": "Strong non-LLM reference",
        "sft-seed7": "Frozen downstream SFT checkpoint",
        "multi-sft": "German–Australian transfer experiment"
      },
      notes: {
        majority: "Class imbalance creates superficial accuracy while missing every high-risk applicant.",
        "zero-shot": "Near-random ranking and severe high-risk over-prediction establish the adaptation baseline.",
        "logistic-regression": "Best final ranking and probability quality; SFT is compared against this strong reference.",
        "sft-seed7": "Selected by validation PR-AUC, not test performance; it has the lowest frozen test Cost.",
        "multi-sft": "Cross-dataset training improves coverage but does not improve the primary German benchmark."
      },
      path: [
        ["Data governance", "Frozen 700 / 100 / 200 split, ChatML records, schema checks.", "Verified", "success"],
        ["Strong baselines", "Majority, Logistic Regression, and zero-shot Qwen establish the comparison floor.", "Verified", "success"],
        ["LoRA SFT", "Three random seeds create useful risk ranking; seed 7 is the frozen downstream checkpoint.", "Verified", "success"],
        ["Preference optimization", "DPO and SimPO variants are retained as negative evidence of label-prior collapse.", "Boundary", "boundary"],
        ["Leakage-safe C7", "Validation chooses the threshold; committed test predictions produce final metrics once.", "Verified", "success"]
      ]
    },
    zh: {
      eyebrow: "应用型 LLM 后训练 · 信用风险",
      heroTitle: "以证据驱动后训练，而非夸大基准结论。",
      heroCopy: "端到端考察 LoRA SFT、偏好优化与仅使用验证集的决策规则，在小样本表格信用风险数据上的实际行为。",
      readmeLink: "阅读技术 README",
      protocolLink: "查看评测协议",
      evidenceLabel: "冻结结果",
      evidenceCaption: "Qwen3.5-4B 零样本到 LoRA SFT 的 ROC-AUC",
      factCost: "SFT 测试 Cost",
      factThreshold: "阈值来源",
      factThresholdValue: "仅验证集",
      resultsEyebrow: "GERMAN CREDIT · 测试集 N = 200",
      resultsTitle: "交互探索冻结后的最终指标",
      metricLabel: "比较指标",
      chartCaption: "切换指标以比较已提交的预测工件。AUC 指标越高越好，Cost 越低越好。",
      selectionLabel: "当前模型",
      tableModel: "模型",
      tableThreshold: "阈值",
      pathEyebrow: "实验路径",
      pathTitle: "完整的结果，也要呈现它的边界",
      pathCopy: "项目将成功的适配、强传统基线和偏好优化的负结果保留为同一条证据链。",
      protocolEyebrow: "评测护栏",
      protocolTitle: "在测试评估前，operating point 已被冻结。",
      protocolCopy: "每个阈值均在验证预测上最小化 5 × 假阴性 + 1 × 假阳性；之后只在冻结阈值下对已提交的测试预测工件评估一次。",
      protocolReadMore: "阅读完整评测协议 →",
      flowValid: "验证集预测",
      flowThreshold: "冻结阈值",
      flowTest: "单次测试指标",
      footerText: "仅用于研究、教育与作品展示，不适用于高影响决策。",
      footerLanguage: "English README",
      modelPicker: "选择模型",
      metricsCaption: "所有模型的冻结 C7 最终指标",
      chartComparison: "对比",
      protocolDiagram: "验证集预测选择阈值，阈值被冻结后应用于测试预测，最终指标只计算一次。",
      metrics: { roc_auc: "ROC-AUC", pr_auc: "PR-AUC", cost: "Cost" },
      lowerBetter: "越低越好",
      higherBetter: "越高越好",
      labels: { role: "实验角色", threshold: "阈值", cost: "Cost", highRecall: "高风险召回" },
      roles: {
        majority: "Sanity-check 下界",
        "zero-shot": "未适配的 LLM 基线",
        "logistic-regression": "强非 LLM 参考模型",
        "sft-seed7": "冻结的下游 SFT checkpoint",
        "multi-sft": "German–Australian 迁移实验"
      },
      notes: {
        majority: "类别不平衡会带来表面准确率，但该模型漏掉了所有高风险申请人。",
        "zero-shot": "近似随机的排序和严重的高风险过预测，构成后训练前的 LLM 基线。",
        "logistic-regression": "最终排序与概率质量最优；SFT 始终与这一强参考模型比较。",
        "sft-seed7": "按验证集 PR-AUC 而非测试表现选择；其冻结测试 Cost 最低。",
        "multi-sft": "跨数据集训练扩大了覆盖范围，但没有提升主 German 基准。"
      },
      path: [
        ["数据治理", "冻结 700 / 100 / 200 划分、ChatML 记录与 Schema 校验。", "已验证", "success"],
        ["强基线", "Majority、Logistic Regression 与零样本 Qwen 建立比较下界。", "已验证", "success"],
        ["LoRA SFT", "三个随机种子产生有效风险排序；seed 7 为冻结下游 checkpoint。", "已验证", "success"],
        ["偏好优化", "DPO 与 SimPO 作为标签先验坍缩的负向证据被完整保留。", "方法边界", "boundary"],
        ["无泄漏 C7", "验证集选择阈值；提交的测试预测只生成一次最终指标。", "已验证", "success"]
      ]
    }
  };

  const svgNamespace = "http://www.w3.org/2000/svg";
  const metricSelect = document.getElementById("metric-select");
  const chart = document.getElementById("metric-chart");
  const buttonContainer = document.getElementById("model-buttons");
  const tableBody = document.getElementById("metric-table-body");
  const pathContainer = document.getElementById("experiment-path");

  function currentCopy() { return copy[state.language]; }
  function modelById(id) { return dataset.models.find((model) => model.id === id); }
  function formatMetric(metric, value) { return metric === "cost" ? String(value) : value.toFixed(3); }
  function displayName(model) {
    if (state.language !== "zh") return model.name;
    return {
      majority: "Majority",
      "zero-shot": "Qwen3.5-4B 零样本",
      "logistic-regression": "Logistic Regression",
      "sft-seed7": "Qwen3.5-4B LoRA SFT · seed 7",
      "multi-sft": "Qwen3.5-4B 多数据集 SFT · German"
    }[model.id];
  }

  function setText() {
    const strings = currentCopy();
    document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach((element) => {
      const value = strings[element.dataset.i18n];
      if (value) element.textContent = value;
    });
    document.querySelectorAll(".language-button").forEach((button) => {
      const active = button.dataset.language === state.language;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    [...metricSelect.options].forEach((option) => { option.textContent = strings.metrics[option.value]; });
    buttonContainer.setAttribute("aria-label", strings.modelPicker);
    document.getElementById("metrics-caption").textContent = strings.metricsCaption;
    document.getElementById("protocol-diagram").setAttribute("aria-label", strings.protocolDiagram);
    document.getElementById("footer-language-link").href = state.language === "zh" ? "../README.md" : "../README.zh-CN.md";
  }

  function createSvgElement(name, attributes = {}) {
    const element = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function renderChart() {
    const strings = currentCopy();
    const metric = state.metric;
    const lowerIsBetter = metric === "cost";
    const sortedModels = [...dataset.models].sort((a, b) => lowerIsBetter ? a[metric] - b[metric] : b[metric] - a[metric]);
    const width = Math.max(chart.clientWidth || 650, 330);
    const height = 318;
    const margin = { top: 24, right: 55, bottom: 22, left: width < 500 ? 122 : 164 };
    const plotWidth = width - margin.left - margin.right;
    const rowHeight = (height - margin.top - margin.bottom) / sortedModels.length;
    const maxValue = lowerIsBetter ? Math.max(...sortedModels.map((model) => model[metric])) : 1;

    chart.replaceChildren();
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    chart.setAttribute("aria-label", `${strings.metrics[metric]} ${strings.chartComparison}. ${strings[lowerIsBetter ? "lowerBetter" : "higherBetter"]}.`);

    const title = createSvgElement("title");
    title.textContent = `${strings.metrics[metric]} ${strings.chartComparison}`;
    chart.append(title);
    const descriptor = createSvgElement("desc");
    descriptor.textContent = strings.chartCaption;
    chart.append(descriptor);

    [0, .25, .5, .75, 1].forEach((ratio) => {
      const x = margin.left + plotWidth * ratio;
      chart.append(createSvgElement("line", { x1: x, x2: x, y1: margin.top - 7, y2: height - margin.bottom, stroke: "currentColor", "stroke-opacity": ".12" }));
      const tick = createSvgElement("text", { x, y: height - 4, "text-anchor": "middle", fill: "currentColor", "fill-opacity": ".64", "font-size": "11" });
      tick.textContent = lowerIsBetter ? String(Math.round(maxValue * ratio)) : ratio.toFixed(2);
      chart.append(tick);
    });

    sortedModels.forEach((model, index) => {
      const y = margin.top + index * rowHeight + rowHeight * .19;
      const selected = model.id === state.selectedModel;
      const scaled = model[metric] / maxValue;
      const bar = createSvgElement("rect", {
        x: margin.left,
        y,
        width: Math.max(2, plotWidth * scaled),
        height: rowHeight * .54,
        rx: 5,
        fill: selected ? "var(--teal)" : "var(--blue)",
        opacity: selected ? "1" : ".78",
        tabindex: "0",
        role: "button",
        "aria-label": `${displayName(model)}: ${formatMetric(metric, model[metric])}`
      });
      bar.addEventListener("click", () => selectModel(model.id));
      bar.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectModel(model.id); } });
      chart.append(bar);

      const label = createSvgElement("text", { x: margin.left - 10, y: y + rowHeight * .36, "text-anchor": "end", fill: "currentColor", "font-size": width < 500 ? "10" : "12", "font-weight": selected ? "700" : "500" });
      label.textContent = displayName(model);
      chart.append(label);

      const value = createSvgElement("text", { x: Math.min(margin.left + plotWidth * scaled + 8, width - 38), y: y + rowHeight * .36, fill: "currentColor", "font-size": "12", "font-weight": "750" });
      value.textContent = formatMetric(metric, model[metric]);
      chart.append(value);
    });
  }

  function renderSelection() {
    const strings = currentCopy();
    const model = modelById(state.selectedModel);
    document.getElementById("selected-model-name").textContent = displayName(model);
    document.getElementById("selected-model-role").textContent = strings.roles[model.id];
    document.getElementById("selected-model-note").textContent = strings.notes[model.id];
    const metrics = [
      [strings.labels.threshold, model.threshold.toFixed(2)],
      [strings.labels.cost, model.cost],
      ["ROC-AUC", model.roc_auc.toFixed(3)],
      [strings.labels.highRecall, model.high_risk_recall.toFixed(3)]
    ];
    const container = document.getElementById("selected-model-metrics");
    container.replaceChildren();
    metrics.forEach(([label, value]) => {
      const wrapper = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      wrapper.append(term, detail);
      container.append(wrapper);
    });
  }

  function renderModelButtons() {
    buttonContainer.replaceChildren();
    dataset.models.forEach((model) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "model-button";
      button.textContent = displayName(model);
      const selected = model.id === state.selectedModel;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.addEventListener("click", () => selectModel(model.id));
      buttonContainer.append(button);
    });
  }

  function renderTable() {
    tableBody.replaceChildren();
    dataset.models.forEach((model) => {
      const row = document.createElement("tr");
      row.tabIndex = 0;
      row.classList.toggle("is-selected", model.id === state.selectedModel);
      row.setAttribute("aria-label", displayName(model));
      [displayName(model), model.roc_auc.toFixed(3), model.pr_auc.toFixed(3), model.nll.toFixed(3), model.brier.toFixed(3), model.ece.toFixed(3), String(model.cost), model.threshold.toFixed(2)].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      row.addEventListener("click", () => selectModel(model.id));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectModel(model.id); } });
      tableBody.append(row);
    });
  }

  function renderPath() {
    pathContainer.replaceChildren();
    currentCopy().path.forEach(([title, detail, status, kind]) => {
      const item = document.createElement("li");
      item.className = kind === "boundary" ? "is-boundary" : "is-success";
      const heading = document.createElement("span");
      const body = document.createElement("p");
      const badge = document.createElement("span");
      heading.className = "path-step-title";
      body.className = "path-step-copy";
      badge.className = "path-step-status";
      heading.textContent = title;
      body.textContent = detail;
      badge.textContent = status;
      item.append(heading, body, badge);
      pathContainer.append(item);
    });
  }

  function selectModel(modelId) {
    state.selectedModel = modelId;
    renderChart();
    renderSelection();
    renderModelButtons();
    renderTable();
  }

  function renderAll() {
    setText();
    renderChart();
    renderSelection();
    renderModelButtons();
    renderTable();
    renderPath();
  }

  metricSelect.addEventListener("change", () => { state.metric = metricSelect.value; renderChart(); });
  document.querySelectorAll(".language-button").forEach((button) => {
    button.addEventListener("click", () => { state.language = button.dataset.language; renderAll(); });
  });
  window.addEventListener("resize", () => renderChart());
  renderAll();
})();
