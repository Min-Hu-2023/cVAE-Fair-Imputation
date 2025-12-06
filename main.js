// main.js

let DATA = null;            // { groundTruth, imputations }
let CURRENT_METHOD_KEY = null;

// ---------------------------
// 1. Load JSON + initialize
// ---------------------------
// Consistent colors for each imputation method
const METHOD_COLORS = {
  mean: "#1f77b4",
  median: "#ff7f0e",
  MissForest: "#2ca02c",
  cVAE: "#d62728",
};

document.addEventListener("DOMContentLoaded", () => {
  // Initialize dataset-related UI (purely presentational)
  initDatasetControls();

  fetch("imputation_viz_data.json")
    .then((resp) => resp.json())
    .then((json) => {
      DATA = json;
      const keys = Object.keys(DATA.imputations);
      CURRENT_METHOD_KEY = keys[0]; // default = first method in JSON

      initMethodSelect();
      initResultTypeControls();
      updateAllForMethod(CURRENT_METHOD_KEY);
    })
    .catch((err) => {
      console.error("Failed to load imputation_viz_data.json:", err);
    });
});

// ---------------------------
// 1b. Dataset controls (UI only)
// ---------------------------

function initDatasetControls() {
  const uploadConfig = document.getElementById("uploadConfig");
  const syntheticConfig = document.getElementById("syntheticConfig");
  const datasetRadios = document.querySelectorAll("input[name='datasetMode']");
  const startBtn = document.getElementById("startImputationBtn");
  const resultsWrapper = document.getElementById("resultsWrapper");

  if (datasetRadios && datasetRadios.length > 0) {
    datasetRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        if (radio.value === "upload") {
          if (uploadConfig) uploadConfig.style.display = "";
          if (syntheticConfig) syntheticConfig.style.display = "none";
        } else {
          if (uploadConfig) uploadConfig.style.display = "none";
          if (syntheticConfig) syntheticConfig.style.display = "";
        }
      });
    });
  }

  if (startBtn && resultsWrapper) {
    startBtn.addEventListener("click", () => {
      // Reveal all the analysis sections
      resultsWrapper.style.display = "";

      // Re-render all plots now that the wrapper is visible
      if (DATA && CURRENT_METHOD_KEY) {
        updateAllForMethod(CURRENT_METHOD_KEY);
      }

      // Smooth scroll down
      resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
}

// ---------------------------
// 2. Method dropdown
// ---------------------------

function prettyLabelFromKey(key) {
  if (key === "mean") return "Mean";
  if (key === "cVAE") return "cVAE";
  if (key === "MissForest") return "MissForest";
  // fallback
  return key;
}

function initMethodSelect() {
  const select = document.getElementById("methodSelect");
  if (!select || !DATA) return;

  select.innerHTML = "";

  Object.keys(DATA.imputations).forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = prettyLabelFromKey(key);
    select.appendChild(opt);
  });

  select.value = CURRENT_METHOD_KEY;

  select.addEventListener("change", () => {
    CURRENT_METHOD_KEY = select.value;
    updateAllForMethod(CURRENT_METHOD_KEY);
  });
}

// ---------------------------------
// 2b. Result-type controls
// ---------------------------------

function initResultTypeControls() {
  const resultTypeSelect = document.getElementById("resultTypeSelect");
  const predictionTypeWrapper = document.getElementById("predictionTypeWrapper");
  const predictionTypeSelect = document.getElementById("predictionTypeSelect");

  const vizSection = document.getElementById("vizSection");
  const predictionSection = document.getElementById("predictionSection");
  const classificationSection = document.getElementById("classificationSection");
  const regressionSection = document.getElementById("regressionSection");

  if (!resultTypeSelect) return;

  // Switch between viz/clustering and prediction
  resultTypeSelect.addEventListener("change", () => {
    const mode = resultTypeSelect.value;

    if (mode === "viz") {
      if (vizSection) vizSection.style.display = "";
      if (predictionSection) predictionSection.style.display = "none";
      if (predictionTypeWrapper) predictionTypeWrapper.style.display = "none";
    } else {
      if (vizSection) vizSection.style.display = "none";
      if (predictionSection) predictionSection.style.display = "";
      if (predictionTypeWrapper) predictionTypeWrapper.style.display = "";

      // Show whichever prediction task is selected
      const task = predictionTypeSelect ? predictionTypeSelect.value : "classification";
      if (task === "classification") {
        if (classificationSection) classificationSection.style.display = "";
        if (regressionSection) regressionSection.style.display = "none";
      } else {
        if (classificationSection) classificationSection.style.display = "none";
        if (regressionSection) regressionSection.style.display = "";

        // Re-render prediction plots so regression gets correct width
        if (DATA && CURRENT_METHOD_KEY) {
          const imp = DATA.imputations[CURRENT_METHOD_KEY];
          if (imp && imp.metrics) {
            renderPredictionSection(imp.metrics);
          }
        }
      }

      // Render prediction plots now that the prediction card is visible
      if (DATA && CURRENT_METHOD_KEY) {
        const imp = DATA.imputations[CURRENT_METHOD_KEY];
        if (imp && imp.metrics) {
          renderPredictionSection(imp.metrics);
        }
      }
    }
  });

  // Switch between classification / regression
  if (predictionTypeSelect) {
    predictionTypeSelect.addEventListener("change", () => {
      const task = predictionTypeSelect.value;

      if (task === "classification") {
        if (classificationSection) classificationSection.style.display = "";
        if (regressionSection) regressionSection.style.display = "none";
      } else {
        // 🔑 Make regression visible *first*
        if (classificationSection) classificationSection.style.display = "none";
        if (regressionSection) regressionSection.style.display = "";

        // 🔑 Then re-render prediction plots so regression gets correct width
        if (DATA && CURRENT_METHOD_KEY) {
          const imp = DATA.imputations[CURRENT_METHOD_KEY];
          if (imp && imp.metrics) {
            renderPredictionSection(imp.metrics);
          }
        }
      }
    });
  }
}

// ---------------------------------
// 3. Orchestrator: update all views
// ---------------------------------

function updateAllForMethod(methodKey) {
  if (!DATA) return;

  const gt = DATA.groundTruth;
  const imp = DATA.imputations[methodKey];
  const metrics = imp.metrics;

  // Section 1: Fairness
  renderFairnessSection(metrics);

  // Section 2: UMAPs + ARI/NMI
  renderVisualizationSection(gt.umap, imp.umap, metrics);

  // Only render prediction plots if the prediction view is actually active
  const resultTypeSelect = document.getElementById("resultTypeSelect");
  if (resultTypeSelect && resultTypeSelect.value === "pred") {
    renderPredictionSection(metrics);
  }
}


// ---------------------------------
// 4. Generic tiny barplot helper
// ---------------------------------
function renderMetricBarAllMethods(containerId, metricKey, title, yAxisTitle, fixed01 = false) {
  const el = document.getElementById(containerId);
  if (!el || !DATA || !DATA.imputations) return;

  const methodKeys = Object.keys(DATA.imputations);

  const xs = [];
  const ys = [];

  methodKeys.forEach((mKey) => {
    const metrics = DATA.imputations[mKey].metrics;
    if (metrics && typeof metrics[metricKey] === "number") {
      xs.push(mKey);                      // method name (e.g. "mean", "cVAE" …)
      ys.push(metrics[metricKey]);        // the metric value
    }
  });

  if (xs.length === 0) return;

  const trace = {
    x: xs,
    y: ys,
    type: "bar",
    text: ys.map((v) => v.toFixed(3)),
    textposition: "auto",
    marker: {
      color: xs.map((method) => METHOD_COLORS[method] || "#888"),
    },
    hovertemplate: "%{x}<br>" + yAxisTitle + ": %{y:.3f}<extra></extra>",
  };

  const yaxis = { title: yAxisTitle };
  if (fixed01) {
    yaxis.range = [0, 1];
  }

  const layout = {
    title,
    margin: { t: 40, r: 10, b: 60, l: 50 },
    xaxis: {
      title: "Imputation method",
      tickangle: -30,
    },
    yaxis,
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true });
}

function renderSingleMetricBar(containerId, value, title, yAxisTitle, fixed01 = false) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const trace = {
    x: [""],
    y: [value],
    type: "bar",
    text: [value.toFixed(3)],
    textposition: "auto",
    hovertemplate: `${title}<br>${yAxisTitle}: %{y:.3f}<extra></extra>`,
  };

  const yaxis = {
    title: yAxisTitle,
  };
  if (fixed01) {
    yaxis.range = [0, 1];
  }

  const layout = {
    title,
    margin: { t: 40, r: 10, b: 40, l: 50 },
    xaxis: { showticklabels: false },
    yaxis,
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true });
}

// ---------------------------------
// 5. Section 1: Fairness metrics
// ---------------------------------

function renderFairnessSection(m) {
  renderMetricBarAllMethods("msieTrainPlot", "MSIE_train", "MSIE (train)", "MSIE");
  renderMetricBarAllMethods("msieTestPlot", "MSIE_test", "MSIE (test)", "MSIE");

  renderMetricBarAllMethods("ifrTrainPlot", "IFR_train", "IFR (train)", "IFR");
  renderMetricBarAllMethods("ifrTestPlot", "IFR_test", "IFR (test)", "IFR");

  renderMetricBarAllMethods("fisTrainPlot", "FIS_train", "FIS (train)", "FIS");
  renderMetricBarAllMethods("fisTestPlot", "FIS_test", "FIS (test)", "FIS");
}

// ---------------------------------
// 6. Section 2: UMAP + ARI/NMI
// ---------------------------------

function renderVisualizationSection(gtUmap, impUmap, metrics) {
  renderUMAPPanels(gtUmap, impUmap);
  renderClusterMetricPanels(metrics);
}

// 6.1 UMAPs

function renderUMAPPanels(gtUmap, impUmap) {
  if (!gtUmap || !impUmap) return;

  const all = gtUmap.concat(impUmap);
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xRange = [Math.min(...xs), Math.max(...xs)];
  const yRange = [Math.min(...ys), Math.max(...ys)];

  plotUmap("umapGtTrue", gtUmap, "Ground truth Dataset – true clusters", "true", xRange, yRange);
  plotUmap("umapGtCluster", gtUmap, "Ground truth Dataset – Agglomerative clustering", "cluster", xRange, yRange);
  plotUmap("umapImpTrue", impUmap, "Imputed – true clusters", "true", xRange, yRange);
  plotUmap("umapImpCluster", impUmap, "Imputed – Agglomerative clustering", "cluster", xRange, yRange);
}

function plotUmap(containerId, points, title, colorField, xRange, yRange) {
  const el = document.getElementById(containerId);
  if (!el || !points || points.length === 0) return;

  const groups = Array.from(new Set(points.map((p) => p[colorField]))).sort();
  const traces = groups.map((g) => {
    const subset = points.filter((p) => p[colorField] === g);
    return {
      x: subset.map((p) => p.x),
      y: subset.map((p) => p.y),
      mode: "markers",
      type: "scattergl",
      name: `${colorField === "true" ? "True" : "Cluster"} ${g}`,
      text: subset.map((p) => `true: ${p.true}, cluster: ${p.cluster}`),
      hovertemplate: "%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
      marker: {
        size: 5,
        opacity: 0.8,
      },
    };
  });

  const layout = {
    title,
    margin: { t: 40, r: 10, b: 40, l: 40 },
    xaxis: { title: "UMAP 1", range: xRange },
    yaxis: { title: "UMAP 2", range: yRange },
    legend: { orientation: "h", y: -0.2 },
  };

  Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

// 6.2 ARI / NMI (UMI) panels

function renderClusterMetricPanels(/* m */) {
  renderMetricBarAllMethods("ariMajorPlot", "ARI_major", "ARI (majority)", "ARI", true);
  renderMetricBarAllMethods("ariMinorPlot", "ARI_minor", "ARI (minority)", "ARI", true);

  // "UMI" in text but stored as NMI_* in the JSON
  renderMetricBarAllMethods("nmiMajorPlot", "NMI_major", "NMI (majority)", "NMI", true);
  renderMetricBarAllMethods("nmiMinorPlot", "NMI_minor", "NMI (minority)", "NMI", true);
}

// ---------------------------------
// 7. Section 3: Prediction performance
// ---------------------------------

function renderPredictionSection(/* m */) {
  // Classification: accuracy in [0, 1]
  renderMetricBarAllMethods(
    "accTrainOverallPlot",
    "accuracy_train_overall",
    "Accuracy (train – overall)",
    "Accuracy",
    true
  );
  renderMetricBarAllMethods(
    "accTrainMajorPlot",
    "accuracy_train_major",
    "Accuracy (train – majority)",
    "Accuracy",
    true
  );
  renderMetricBarAllMethods(
    "accTrainMinorPlot",
    "accuracy_train_minor",
    "Accuracy (train – minority)",
    "Accuracy",
    true
  );
  renderMetricBarAllMethods(
    "accTestOverallPlot",
    "accuracy_test_overall",
    "Accuracy (test – overall)",
    "Accuracy",
    true
  );
  renderMetricBarAllMethods(
    "accTestMajorPlot",
    "accuracy_test_major",
    "Accuracy (test – majority)",
    "Accuracy",
    true
  );
  renderMetricBarAllMethods(
    "accTestMinorPlot",
    "accuracy_test_minor",
    "Accuracy (test – minority)",
    "Accuracy",
    true
  );

  // Regression: MSE (no fixed 0–1 range)
  renderMetricBarAllMethods(
    "mseTrainOverallPlot",
    "MSE_train_overall",
    "MSE (train – overall)",
    "MSE"
  );
  renderMetricBarAllMethods(
    "mseTrainMajorPlot",
    "MSE_train_major",
    "MSE (train – majority)",
    "MSE"
  );
  renderMetricBarAllMethods(
    "mseTrainMinorPlot",
    "MSE_train_minor",
    "MSE (train – minority)",
    "MSE"
  );
  renderMetricBarAllMethods(
    "mseTestOverallPlot",
    "MSE_test_overall",
    "MSE (test – overall)",
    "MSE"
  );
  renderMetricBarAllMethods(
    "mseTestMajorPlot",
    "MSE_test_major",
    "MSE (test – majority)",
    "MSE"
  );
  renderMetricBarAllMethods(
    "mseTestMinorPlot",
    "MSE_test_minor",
    "MSE (test – minority)",
    "MSE"
  );
}