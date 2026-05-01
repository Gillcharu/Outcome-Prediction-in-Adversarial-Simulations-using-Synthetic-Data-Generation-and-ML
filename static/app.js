function applyBasePreset(values) {
  Object.entries(values).forEach(([key, value]) => {
    const field = document.querySelector(`[name="${key}"]`);
    if (field) {
      field.value = value;
    }
  });
}

function setupPresets() {
  const presetButtons = document.querySelectorAll(".preset-card");
  presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const raw = button.getAttribute("data-preset");
      if (!raw) {
        return;
      }
      applyBasePreset(JSON.parse(raw));
    });
  });
}

function renderBarChart(rootId) {
  const chartRoot = document.getElementById(rootId);
  if (!chartRoot) {
    return;
  }

  const chartCard = chartRoot.closest(".chart-card");
  if (!chartCard) {
    return;
  }

  const raw = chartCard.getAttribute("data-chart");
  if (!raw) {
    return;
  }

  const data = JSON.parse(raw);
  chartRoot.innerHTML = "";

  data.labels.forEach((label, index) => {
    const value = data.values[index];
    const row = document.createElement("div");
    row.className = "bar-row";

    const meta = document.createElement("div");
    meta.className = "bar-meta";

    const name = document.createElement("span");
    name.textContent = label;

    const score = document.createElement("strong");
    score.textContent = data.kind === "feature-importance"
      ? `${value.toFixed(2)}`
      : `${value.toFixed(1)}%`;

    meta.appendChild(name);
    meta.appendChild(score);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, value))}%`;

    track.appendChild(fill);
    row.appendChild(meta);
    row.appendChild(track);
    chartRoot.appendChild(row);
  });
}

setupPresets();
renderBarChart("comparison-chart");
renderBarChart("feature-importance-chart");
