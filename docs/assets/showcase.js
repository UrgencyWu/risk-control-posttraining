(() => {
  const dataset = JSON.parse(document.getElementById("showcase-data").textContent);
  const state = { language: "zh", metric: "roc_auc", selectedModel: "sft-seed7" };

  const copy = {
    zh: {
      documentTitle: "大语言模型风控后训练 | 交互式展示",
      description: "大语言模型风控后训练项目：研究问题、技术路线、实验数据、结论与复现指引。",
      skipLink: "跳至正文",
      brand: "风控后训练研究",
      navSummary: "摘要",
      navQuestion: "研究问题",
      navRoute: "技术路线",
      navData: "实验数据",
      navReproduce: "复现",
      summaryEyebrow: "项目摘要 · 信用风险后训练",
      summaryTitle: "SFT 学到了有效风险排序，偏好优化暴露出明确的方法边界。",
      summaryCopy: "项目以 Qwen3.5-4B 和 German Credit 为主实验对象，系统比较 LoRA SFT、DPO/SimPO 与 Logistic Regression，并用无测试泄漏的成本敏感评测区分真实收益与标签先验偏移。",
      viewData: "查看实验数据",
      viewRepo: "查看 GitHub 仓库",
      summaryStatSft: "SFT 的 ROC-AUC 增量",
      summaryStatBaseline: "Logistic Regression 与 SFT",
      summaryStatBaselineDetail: "差距 0.010，不声称稳定超越",
      summaryStatCost: "最佳 SFT 测试 Cost",
      summaryStatCostDetail: "阈值仅由验证集选择",
      summaryStatPreference: "主 DPO/SimPO 实验未超过 SFT",
      summaryStatPreferenceDetail: "标签先验偏移或决策坍缩",
      questionEyebrow: "研究问题",
      questionTitle: "后训练能否在强传统基线面前产生真实、稳定的风险判别价值？",
      questionLead: "项目不只比较最终 Accuracy，而是同时检查样本级排序、概率质量、非对称成本和跨随机种子稳定性。",
      questionOne: "指令模型能否从自然语言形式的表格信用记录中学习可泛化的风险排序？",
      questionTwo: "LoRA SFT 与 DPO/SimPO 分别改变了样本判别能力，还是只改变了两个标签的整体概率？",
      questionThree: "当后训练失败时，根因是数据、优化、决策阈值，还是目标函数与二标签短回答任务的错配？",
      routeEyebrow: "技术路线",
      routeTitle: "从数据协议到机制审计的完整实验链",
      routeCopy: "表示学习与决策 operating point 分离；所有业务阈值在验证集冻结后再进入测试评估。",
      route1Title: "仓库与数据审计",
      route1Copy: "核查 CALM 数据、标签与训练链路。",
      route2Title: "统一数据协议",
      route2Copy: "定义 RiskDataset Schema 与标签语义。",
      route3Title: "确定性数据构造",
      route3Copy: "冻结划分并生成 Normalized / ChatML 数据。",
      route4Title: "强弱基线",
      route4Copy: "比较 Majority、Logistic Regression 与 Zero-shot Qwen。",
      route5Title: "LoRA SFT",
      route5Copy: "三随机种子验证风险排序与稳定性。",
      route6Title: "偏好优化",
      route6Copy: "构造 Oracle / Hard Preference，运行 DPO/SimPO。",
      route7Title: "成本敏感决策",
      route7Copy: "在验证集最小化 5×FN + FP 并冻结阈值。",
      route8Title: "最终评测与审计",
      route8Copy: "C7 统一指标与 log-probability 机制分析。",
      implementationEyebrow: "实施细节",
      implementationTitle: "可复现的数据、训练与评测实现",
      implementationCopy: "仓库公开处理后数据、训练入口、预测工件、指标、日志和审计报告；大模型与 LoRA 权重未提交。",
      implDataTitle: "数据治理",
      implDataCopy: "审计 10 个数据集；German Credit 冻结为 700/100/200；通过 Schema、manifest、V1–V7 与 SHA-256 检查保证可追溯和确定性。",
      implDataFact1: "German 记录",
      implDataFact2: "输入特征",
      implTrainTitle: "LoRA SFT",
      implTrainCopy: "Qwen3.5-4B 使用 response-only loss、LoRA r=16、alpha=32、dropout=0.05，训练 5 个 epoch，并完成三个随机种子。",
      implTrainFact1: "训练 GPU",
      implTrainFact2: "随机种子",
      implPreferenceTitle: "偏好优化",
      implPreferenceCopy: "构造 Oracle、Hard 与多数据集 Preference，覆盖 DPO、SimPO、Cost-sensitive SFT、Anchored DPO 和 Risk-DPO Pilot。",
      implPreferenceFact1: "Oracle 对",
      implPreferenceFact2: "主实验",
      implEvalTitle: "无泄漏评测",
      implEvalCopy: "统一计算 AUC、校准、召回、混淆矩阵与非对称 Cost；阈值仅从已提交验证预测中选择，再固定用于测试集。",
      implEvalFact1: "FN : FP 成本",
      implEvalFact2: "测试调阈值",
      implEvalNo: "禁止",
      dataEyebrow: "实验数据 · German Credit 测试集 N = 200",
      dataTitle: "折线图与冻结指标表",
      dataCopy: "折线图按模型显示同一指标；点击数据点、模型按钮或表格行可以查看具体实验角色和阈值。",
      metricLabel: "比较指标",
      chartCaption: "AUC 越高越好，Cost 越低越好。折线用于比较离散实验结果，不表示连续训练过程。",
      selectionLabel: "当前模型",
      tableModel: "模型",
      tableThreshold: "阈值",
      conclusionEyebrow: "实验结论",
      conclusionTitle: "正向结果、诚实基线与方法边界",
      conclusionCopy: "项目保留成功结果、稳定性限制和机制性负结果，不以单次最佳指标替代完整证据。",
      conclusion1Kicker: "SFT 有效",
      conclusion1Title: "ROC-AUC 提升 +0.232",
      conclusion1Copy: "SFT 是唯一持续产生有效风险排序的后训练方法，并同步改善 PR-AUC、NLL 与 Brier。",
      conclusion2Kicker: "诚实对比",
      conclusion2Title: "0.757 仍高于 0.747",
      conclusion2Copy: "Logistic Regression 的排序与概率质量仍略强；最佳 SFT Cost 更低，但三随机种子均值不支持稳定超越。",
      conclusion3Kicker: "决策规则",
      conclusion3Title: "成本阈值有效，但不能替代排序",
      conclusion3Copy: "验证集阈值降低高风险漏判成本；AUC 与校准指标用于识别全部预测 high risk 的伪改进。",
      conclusion4Kicker: "方法边界",
      conclusion4Title: "6 / 6 组 DPO/SimPO 未超过 SFT",
      conclusion4Copy: "短标签偏好目标主要移动全局标签先验，未直接监督申请人之间的风险顺序，并多次导致单类别坍缩。",
      reproEyebrow: "复现指引",
      reproTitle: "先复现冻结指标，再按需运行完整 GPU 链路",
      reproCopy: "公开指标可以完全由已提交的 valid/test 预测工件在 CPU 上重建；完整训练需要本地 Qwen3.5-4B 与 GPU 环境。",
      cpuTitle: "CPU：复现最终指标",
      cpuCopy: "不需要模型权重或 GPU。",
      gpuTitle: "GPU：训练与推理",
      gpuCopy: "需要本地模型路径、训练依赖和 Slurm 环境。",
      repoLink: "GitHub 仓库",
      showcaseLink: "展示页与部署说明",
      protocolLink: "完整评测协议",
      englishReadmeLink: "English README",
      footerText: "仅用于研究、教育与作品展示，不适用于真实高影响决策。",
      footerLanguage: "English README",
      modelPicker: "选择模型",
      metricsCaption: "冻结 C7 最终指标",
      chartComparison: "折线图",
      lowerBetter: "越低越好",
      higherBetter: "越高越好",
      labels: { threshold: "阈值", cost: "Cost", highRecall: "高风险召回" },
      metrics: { roc_auc: "ROC-AUC", pr_auc: "PR-AUC", cost: "Cost" },
      roles: {
        majority: "类别不平衡下界",
        "zero-shot": "未适配的 LLM 基线",
        "logistic-regression": "强非 LLM 参考模型",
        "sft-seed7": "按验证集规则冻结的 SFT Checkpoint",
        "multi-sft": "German–Australian 多数据集迁移实验"
      },
      notes: {
        majority: "67.5% Accuracy 来自类别不平衡，但漏掉所有高风险样本。",
        "zero-shot": "排序接近随机，并严重偏向预测 high risk。",
        "logistic-regression": "ROC-AUC 与概率质量最强，是 SFT 必须面对的诚实基线。",
        "sft-seed7": "按验证集 PR-AUC 规则选择，而非按测试表现挑选；冻结测试 Cost 最低。",
        "multi-sft": "扩大跨数据集覆盖，但没有提升主 German 基准。"
      },
      shortNames: {
        majority: "Majority",
        "zero-shot": "Zero-shot",
        "logistic-regression": "LR",
        "sft-seed7": "SFT",
        "multi-sft": "Multi-SFT"
      }
    },
    en: {
      documentTitle: "Risk-Control Post-Training | Interactive Showcase",
      description: "Risk-control LLM post-training project: summary, research questions, technical route, experimental data, conclusions, and reproduction guide.",
      skipLink: "Skip to content",
      brand: "Risk-Control Post-Training",
      navSummary: "Summary",
      navQuestion: "Questions",
      navRoute: "Route",
      navData: "Data",
      navReproduce: "Reproduce",
      summaryEyebrow: "Project Summary · Credit-Risk Post-Training",
      summaryTitle: "SFT learned useful risk ranking; preference optimization exposed a clear method boundary.",
      summaryCopy: "Using Qwen3.5-4B and German Credit, the project compares LoRA SFT, DPO/SimPO, and Logistic Regression under leakage-safe cost-sensitive evaluation to separate real discrimination gains from global label-prior shifts.",
      viewData: "View experimental data",
      viewRepo: "Open GitHub repository",
      summaryStatSft: "SFT ROC-AUC gain",
      summaryStatBaseline: "Logistic Regression vs SFT",
      summaryStatBaselineDetail: "0.010 gap; no stable-superiority claim",
      summaryStatCost: "Best SFT test Cost",
      summaryStatCostDetail: "Threshold selected on validation only",
      summaryStatPreference: "Principal DPO/SimPO runs below SFT",
      summaryStatPreferenceDetail: "Label-prior shift or decision collapse",
      questionEyebrow: "Research Questions",
      questionTitle: "Does post-training create real and stable risk-discrimination value against a strong statistical baseline?",
      questionLead: "The study evaluates sample-level ranking, probability quality, asymmetric cost, and cross-seed stability rather than relying on final Accuracy alone.",
      questionOne: "Can an instruction-tuned model learn generalizable risk ranking from tabular credit records expressed as natural-language instructions?",
      questionTwo: "Do LoRA SFT and DPO/SimPO improve conditional discrimination, or only change the global probability of the two labels?",
      questionThree: "When post-training fails, is the root cause data, optimization, decision thresholds, or an objective mismatch with two-label short answers?",
      routeEyebrow: "Technical Route",
      routeTitle: "A complete chain from data contracts to mechanism audits",
      routeCopy: "Representation learning is separated from the decision operating point; business thresholds are frozen on validation before test evaluation.",
      route1Title: "Repository and data audit",
      route1Copy: "Inspect CALM data, labels, and training paths.",
      route2Title: "Unified data contract",
      route2Copy: "Define RiskDataset schema and label semantics.",
      route3Title: "Deterministic construction",
      route3Copy: "Freeze splits and generate Normalized / ChatML records.",
      route4Title: "Strong and weak baselines",
      route4Copy: "Compare Majority, Logistic Regression, and Zero-shot Qwen.",
      route5Title: "LoRA SFT",
      route5Copy: "Use three seeds to evaluate ranking and stability.",
      route6Title: "Preference optimization",
      route6Copy: "Build Oracle / Hard pairs and run DPO/SimPO.",
      route7Title: "Cost-sensitive decisions",
      route7Copy: "Minimize 5×FN + FP on validation and freeze thresholds.",
      route8Title: "Final evaluation and audit",
      route8Copy: "Run C7 metrics and log-probability mechanism analysis.",
      implementationEyebrow: "Implementation Details",
      implementationTitle: "Reproducible data, training, and evaluation",
      implementationCopy: "The repository publishes processed data, entry points, prediction artifacts, metrics, logs, and audit reports; large model and LoRA weight binaries are excluded.",
      implDataTitle: "Data governance",
      implDataCopy: "Audit 10 datasets; freeze German Credit to 700/100/200; enforce schema, manifest, V1–V7, and SHA-256 checks for traceability and determinism.",
      implDataFact1: "German records",
      implDataFact2: "Input features",
      implTrainTitle: "LoRA SFT",
      implTrainCopy: "Train Qwen3.5-4B with response-only loss, LoRA r=16, alpha=32, dropout=0.05, five epochs, and three random seeds.",
      implTrainFact1: "Training GPUs",
      implTrainFact2: "Random seeds",
      implPreferenceTitle: "Preference optimization",
      implPreferenceCopy: "Construct Oracle, Hard, and multi-dataset preference data for DPO, SimPO, Cost-sensitive SFT, Anchored DPO, and Risk-DPO pilots.",
      implPreferenceFact1: "Oracle pairs",
      implPreferenceFact2: "Principal runs",
      implEvalTitle: "Leakage-safe evaluation",
      implEvalCopy: "Compute AUC, calibration, recall, confusion matrices, and asymmetric Cost; select thresholds only from committed validation predictions and freeze them for test.",
      implEvalFact1: "FN : FP cost",
      implEvalFact2: "Test threshold tuning",
      implEvalNo: "Forbidden",
      dataEyebrow: "Experimental Data · German Credit Test N = 200",
      dataTitle: "Line chart and frozen metric table",
      dataCopy: "The line chart compares one metric across discrete model results. Select a point, model button, or table row to inspect its role and threshold.",
      metricLabel: "Compare by",
      chartCaption: "Higher is better for AUC; lower is better for Cost. The line connects discrete experiments and does not represent a continuous training trajectory.",
      selectionLabel: "Selected model",
      tableModel: "Model",
      tableThreshold: "Threshold",
      conclusionEyebrow: "Experimental Conclusions",
      conclusionTitle: "Positive evidence, an honest baseline, and a method boundary",
      conclusionCopy: "The project preserves successful results, stability limits, and mechanism-level negative evidence rather than replacing the full record with one best run.",
      conclusion1Kicker: "SFT works",
      conclusion1Title: "ROC-AUC improves by +0.232",
      conclusion1Copy: "SFT is the only post-training method that consistently creates useful risk ranking and also improves PR-AUC, NLL, and Brier.",
      conclusion2Kicker: "Honest comparison",
      conclusion2Title: "0.757 remains above 0.747",
      conclusion2Copy: "Logistic Regression still has slightly stronger ranking and probability quality; lower best-seed SFT Cost does not establish stable superiority across seeds.",
      conclusion3Kicker: "Decision rule",
      conclusion3Title: "Cost thresholds help but cannot replace ranking",
      conclusion3Copy: "Validation thresholds reduce high-risk false-negative cost, while AUC and calibration expose trivial all-high-risk behavior.",
      conclusion4Kicker: "Method boundary",
      conclusion4Title: "6 / 6 DPO/SimPO runs fail to exceed SFT",
      conclusion4Copy: "Short-label preference objectives mainly move global label priors, do not directly supervise applicant-to-applicant risk order, and repeatedly collapse decisions.",
      reproEyebrow: "Reproduction Guide",
      reproTitle: "Reproduce frozen metrics first, then run the full GPU path if needed",
      reproCopy: "Published metrics can be rebuilt on CPU from committed validation/test predictions. Full training requires a local Qwen3.5-4B model and GPU environment.",
      cpuTitle: "CPU: reproduce final metrics",
      cpuCopy: "No model weights or GPU are required.",
      gpuTitle: "GPU: training and inference",
      gpuCopy: "Requires a local model path, training dependencies, and Slurm.",
      repoLink: "GitHub repository",
      showcaseLink: "Showcase and deployment notes",
      protocolLink: "Full evaluation protocol",
      englishReadmeLink: "中文 README",
      footerText: "For research, education, and portfolio demonstration only. Not for real high-impact decisions.",
      footerLanguage: "中文 README",
      modelPicker: "Choose a model",
      metricsCaption: "Frozen C7 final metrics",
      chartComparison: "line chart",
      lowerBetter: "Lower is better",
      higherBetter: "Higher is better",
      labels: { threshold: "Threshold", cost: "Cost", highRecall: "High-risk recall" },
      metrics: { roc_auc: "ROC-AUC", pr_auc: "PR-AUC", cost: "Cost" },
      roles: {
        majority: "Class-imbalance lower bound",
        "zero-shot": "Unadapted LLM baseline",
        "logistic-regression": "Strong non-LLM reference",
        "sft-seed7": "SFT checkpoint frozen by the validation rule",
        "multi-sft": "German–Australian multi-dataset transfer experiment"
      },
      notes: {
        majority: "Its 67.5% Accuracy comes from class imbalance while missing every high-risk case.",
        "zero-shot": "Ranking is near random and predictions are heavily biased toward high risk.",
        "logistic-regression": "It has the strongest ROC-AUC and probability quality and serves as the honest reference for SFT.",
        "sft-seed7": "Selected by validation PR-AUC rather than test performance; it has the lowest frozen test Cost.",
        "multi-sft": "It broadens cross-dataset coverage but does not improve the primary German benchmark."
      },
      shortNames: {
        majority: "Majority",
        "zero-shot": "Zero-shot",
        "logistic-regression": "LR",
        "sft-seed7": "SFT",
        "multi-sft": "Multi-SFT"
      }
    }
  };

  const svgNamespace = "http://www.w3.org/2000/svg";
  const metricSelect = document.getElementById("metric-select");
  const chart = document.getElementById("metric-chart");
  const buttonContainer = document.getElementById("model-buttons");
  const tableBody = document.getElementById("metric-table-body");

  function currentCopy() { return copy[state.language]; }
  function modelById(id) { return dataset.models.find((model) => model.id === id); }
  function formatMetric(metric, value) { return metric === "cost" ? String(value) : value.toFixed(3); }
  function formatAxis(metric, value) { return metric === "cost" ? String(Math.round(value)) : value.toFixed(2); }

  function displayName(model) {
    if (state.language === "en") return model.name;
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
    document.title = strings.documentTitle;
    const description = document.querySelector('meta[name="description"]');
    if (description) description.setAttribute("content", strings.description);

    document.querySelectorAll("[data-i18n]").forEach((element) => {
      const value = strings[element.dataset.i18n];
      if (value) element.textContent = value;
    });

    document.querySelectorAll(".language-button").forEach((button) => {
      const active = button.dataset.language === state.language;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    [...metricSelect.options].forEach((option) => {
      option.textContent = strings.metrics[option.value];
    });

    buttonContainer.setAttribute("aria-label", strings.modelPicker);
    document.getElementById("metrics-caption").textContent = strings.metricsCaption;

    const alternateReadme = document.querySelector('[data-i18n="englishReadmeLink"]');
    if (alternateReadme) alternateReadme.href = state.language === "zh" ? "../README.en.md" : "../README.md";
    const footerLanguage = document.getElementById("footer-language-link");
    footerLanguage.href = state.language === "zh" ? "../README.en.md" : "../README.md";
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
    const models = dataset.models;
    const width = Math.max(chart.clientWidth || 720, 340);
    const height = 350;
    const margin = { top: 30, right: 30, bottom: 82, left: width < 520 ? 52 : 64 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minValue = lowerIsBetter ? 0 : 0.30;
    const maxValue = lowerIsBetter ? 350 : 0.80;
    const xStep = plotWidth / Math.max(models.length - 1, 1);
    const xFor = (index) => margin.left + index * xStep;
    const yFor = (value) => margin.top + ((maxValue - value) / (maxValue - minValue)) * plotHeight;
    const points = models.map((model, index) => ({ model, x: xFor(index), y: yFor(model[metric]) }));

    chart.replaceChildren();
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    chart.setAttribute("aria-label", `${strings.metrics[metric]} ${strings.chartComparison}. ${strings[lowerIsBetter ? "lowerBetter" : "higherBetter"]}.`);

    const title = createSvgElement("title");
    title.textContent = `${strings.metrics[metric]} ${strings.chartComparison}`;
    chart.append(title);
    const descriptor = createSvgElement("desc");
    descriptor.textContent = strings.chartCaption;
    chart.append(descriptor);

    const tickCount = 5;
    for (let index = 0; index <= tickCount; index += 1) {
      const ratio = index / tickCount;
      const value = minValue + (maxValue - minValue) * ratio;
      const y = yFor(value);
      chart.append(createSvgElement("line", {
        x1: margin.left,
        x2: width - margin.right,
        y1: y,
        y2: y,
        stroke: "currentColor",
        "stroke-opacity": ".12",
        "stroke-width": "1"
      }));
      const tick = createSvgElement("text", {
        x: margin.left - 9,
        y: y + 4,
        "text-anchor": "end",
        fill: "currentColor",
        "fill-opacity": ".62",
        "font-size": "11"
      });
      tick.textContent = formatAxis(metric, value);
      chart.append(tick);
    }

    chart.append(createSvgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: margin.top + plotHeight,
      y2: margin.top + plotHeight,
      stroke: "currentColor",
      "stroke-width": "1.3"
    }));

    const polyline = createSvgElement("polyline", {
      points: points.map((point) => `${point.x},${point.y}`).join(" "),
      fill: "none",
      stroke: "currentColor",
      "stroke-width": "2.5",
      "stroke-linecap": "round",
      "stroke-linejoin": "round"
    });
    chart.append(polyline);

    points.forEach(({ model, x, y }) => {
      const selected = model.id === state.selectedModel;
      const point = createSvgElement("circle", {
        cx: x,
        cy: y,
        r: selected ? 7.5 : 6,
        fill: selected ? "var(--accent)" : "var(--paper)",
        stroke: selected ? "var(--accent)" : "currentColor",
        "stroke-width": selected ? "3" : "2.2",
        tabindex: "0",
        role: "button",
        "aria-label": `${displayName(model)}: ${formatMetric(metric, model[metric])}`
      });
      point.addEventListener("click", () => selectModel(model.id));
      point.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectModel(model.id);
        }
      });
      chart.append(point);

      const value = createSvgElement("text", {
        x,
        y: Math.max(y - 13, 15),
        "text-anchor": "middle",
        fill: selected ? "var(--accent)" : "currentColor",
        "font-size": "11.5",
        "font-weight": selected ? "800" : "650"
      });
      value.textContent = formatMetric(metric, model[metric]);
      chart.append(value);

      const label = createSvgElement("text", {
        x,
        y: margin.top + plotHeight + 28,
        "text-anchor": "middle",
        fill: "currentColor",
        "font-size": width < 520 ? "10" : "11.5",
        "font-weight": selected ? "800" : "600"
      });
      label.textContent = strings.shortNames[model.id];
      chart.append(label);
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
      [
        displayName(model),
        model.roc_auc.toFixed(3),
        model.pr_auc.toFixed(3),
        model.nll.toFixed(3),
        model.brier.toFixed(3),
        model.ece.toFixed(3),
        String(model.cost),
        model.threshold.toFixed(2)
      ].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      row.addEventListener("click", () => selectModel(model.id));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectModel(model.id);
        }
      });
      tableBody.append(row);
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
  }

  metricSelect.addEventListener("change", () => {
    state.metric = metricSelect.value;
    renderChart();
  });

  document.querySelectorAll(".language-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.language = button.dataset.language;
      renderAll();
    });
  });

  window.addEventListener("resize", () => renderChart());
  renderAll();
})();
