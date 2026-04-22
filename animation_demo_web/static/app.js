const pipelineEl = document.getElementById('pipeline');
const logsEl = document.getElementById('logs');
const runBtn = document.getElementById('runBtn');
const stageExplainEl = document.getElementById('stageExplain');
const detailStatsEl = document.getElementById('detailStats');
const detailVizEl = document.getElementById('detailViz');

const synMapeEl = document.getElementById('synMape');
const synRmseEl = document.getElementById('synRmse');
const baseMapeEl = document.getElementById('baseMape');
const baseRmseEl = document.getElementById('baseRmse');
const rowsEl = document.getElementById('rows');
const outDaysEl = document.getElementById('outDays');
const totalKwhEl = document.getElementById('totalKwh');
const runStatusEl = document.getElementById('runStatus');

let ws = null;
let stages = [];
let stageDetails = {};
let currentStageName = '';

function addLog(line) {
  logsEl.textContent += line + '\n';
  logsEl.scrollTop = logsEl.scrollHeight;
}

function stageClass(status) {
  if (status === 'running') return 'stage running';
  if (status === 'done') return 'stage done';
  if (status === 'failed') return 'stage failed';
  return 'stage';
}

function renderStages() {
  pipelineEl.innerHTML = '';
  stages.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = stageClass(s.status);
    div.innerHTML = `<h3>${i + 1}. ${s.name}</h3><p>${s.status}</p>`;
    div.addEventListener('click', () => {
      currentStageName = s.name;
      renderDetailForStage(currentStageName);
    });
    pipelineEl.appendChild(div);
  });
}

function setMetric(el, value, suffix = '') {
  if (value === undefined || value === null) return;
  el.textContent = `${value}${suffix}`;
}

function getConfig() {
  return {
    days: Number(document.getElementById('days').value),
    targetKwh: Number(document.getElementById('targetKwh').value),
    targetMode: document.getElementById('targetMode').value,
    k: Number(document.getElementById('k').value),
    skipTrain: document.getElementById('skipTrain').checked,
  };
}

function clearDetail() {
  detailStatsEl.innerHTML = '';
  detailVizEl.innerHTML = '';
}

function addStat(label, value) {
  const div = document.createElement('div');
  div.className = 'stat-pill';
  div.innerHTML = `<b>${label}</b><br/>${value}`;
  detailStatsEl.appendChild(div);
}

function renderLineChart(values, title) {
  if (!values || values.length === 0) {
    detailVizEl.innerHTML = '<div class="viz-caption">No preview data yet.</div>';
    return;
  }

  const width = Math.max(700, detailVizEl.clientWidth - 20);
  const height = 180;
  const pad = 24;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-9, max - min);

  const pts = values.map((v, i) => {
    const x = pad + (i / Math.max(1, values.length - 1)) * (width - 2 * pad);
    const y = height - pad - ((v - min) / span) * (height - 2 * pad);
    return `${x},${y}`;
  }).join(' ');

  detailVizEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#0b121a"></rect>
      <polyline points="${pts}" fill="none" stroke="#4fc3f7" stroke-width="2"></polyline>
    </svg>
    <div class="viz-caption">${title} | min=${min.toFixed(4)} max=${max.toFixed(4)}</div>
  `;
}

function renderCentersChart(centers) {
  if (!centers || centers.length === 0) {
    detailVizEl.innerHTML = '<div class="viz-caption">No cluster centers available yet.</div>';
    return;
  }

  const width = Math.max(700, detailVizEl.clientWidth - 20);
  const height = 180;
  const pad = 24;
  const flat = centers.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = Math.max(1e-9, max - min);
  const colors = ['#4fc3f7', '#81c784', '#ffb74d', '#ba68c8', '#64b5f6', '#e57373', '#ffd54f', '#a1887f', '#90a4ae'];

  const lines = centers.slice(0, 9).map((arr, idx) => {
    const pts = arr.map((v, i) => {
      const x = pad + (i / Math.max(1, arr.length - 1)) * (width - 2 * pad);
      const y = height - pad - ((v - min) / span) * (height - 2 * pad);
      return `${x},${y}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="1.7"></polyline>`;
  }).join('');

  detailVizEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#0b121a"></rect>
      ${lines}
    </svg>
    <div class="viz-caption">Cluster center daily profiles (24-hour vectors)</div>
  `;
}

function renderHeatmap(matrix) {
  if (!matrix || matrix.length === 0) {
    detailVizEl.innerHTML = '<div class="viz-caption">No transition matrix available yet.</div>';
    return;
  }
  const n = matrix.length;
  detailVizEl.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'heatmap';
  wrap.style.gridTemplateColumns = `repeat(${n}, minmax(36px, 1fr))`;

  let max = 0;
  matrix.forEach(r => r.forEach(v => { if (v > max) max = v; }));
  max = Math.max(max, 1e-9);

  matrix.forEach(row => {
    row.forEach(v => {
      const c = Math.floor((v / max) * 200);
      const cell = document.createElement('div');
      cell.className = 'heat-cell';
      cell.style.background = `rgb(${20}, ${40 + c}, ${80 + c / 2})`;
      cell.textContent = Number(v).toFixed(2);
      wrap.appendChild(cell);
    });
  });

  detailVizEl.appendChild(wrap);
  const cap = document.createElement('div');
  cap.className = 'viz-caption';
  cap.textContent = 'Markov transition matrix (rows: current state, columns: next state)';
  detailVizEl.appendChild(cap);
}

function renderDetailForStage(stageName) {
  const detail = stageDetails[stageName] || {};
  const summary = detail.summary || {};
  stageExplainEl.textContent = detail.explain || 'Waiting for stage details...';
  clearDetail();

  if (stageName === 'Preprocess') {
    addStat('Rows', summary.rows ?? '-');
    addStat('Total kWh', summary.total_kwh != null ? Number(summary.total_kwh).toFixed(3) : '-');
    addStat('Mean kWh', summary.mean_kwh != null ? Number(summary.mean_kwh).toFixed(4) : '-');
    renderLineChart(summary.preview || [], 'Hourly load preview (last ~7 days)');
    return;
  }

  if (stageName === 'Train KMeans') {
    addStat('Clusters', summary.clusters ?? '-');
    addStat('Center min', summary.center_min != null ? Number(summary.center_min).toFixed(4) : '-');
    addStat('Center max', summary.center_max != null ? Number(summary.center_max).toFixed(4) : '-');
    renderCentersChart(summary.centers || []);
    return;
  }

  if (stageName === 'Build Markov') {
    addStat('Shape', summary.shape ? `${summary.shape[0]}x${summary.shape[1]}` : '-');
    addStat('Row-sum mean', summary.row_sum_mean != null ? Number(summary.row_sum_mean).toFixed(4) : '-');
    addStat('Row-sum std', summary.row_sum_std != null ? Number(summary.row_sum_std).toFixed(6) : '-');
    renderHeatmap(summary.matrix || []);
    return;
  }

  if (stageName === 'Validate') {
    addStat('Synthetic MAPE', summary.synthetic_mape != null ? `${Number(summary.synthetic_mape).toFixed(2)}%` : '-');
    addStat('Synthetic RMSE', summary.synthetic_rmse != null ? Number(summary.synthetic_rmse).toFixed(3) : '-');
    addStat('Baseline MAPE', summary.baseline_mape != null ? `${Number(summary.baseline_mape).toFixed(2)}%` : '-');
    addStat('Baseline RMSE', summary.baseline_rmse != null ? Number(summary.baseline_rmse).toFixed(3) : '-');
    addStat('Noise', summary.noise != null ? Number(summary.noise).toFixed(3) : '-');
    addStat('Blend alpha', summary.blend_alpha != null ? Number(summary.blend_alpha).toFixed(2) : '-');
    detailVizEl.innerHTML = '<div class="viz-caption">Validation compares synthetic generation against baseline on holdout data.</div>';
    return;
  }

  if (stageName === 'Generate') {
    addStat('Rows', summary.rows ?? '-');
    addStat('Days', summary.days ?? '-');
    addStat('Total kWh', summary.total_kwh != null ? Number(summary.total_kwh).toFixed(3) : '-');
    addStat('Mean kWh', summary.mean_kwh != null ? Number(summary.mean_kwh).toFixed(4) : '-');
    renderLineChart(summary.preview || [], 'Generated profile preview (first ~7 days)');
    return;
  }

  detailVizEl.innerHTML = '<div class="viz-caption">No stage details yet.</div>';
}

function startRun() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  logsEl.textContent = '';
  stageDetails = {};
  currentStageName = '';
  runStatusEl.textContent = 'Running';
  runBtn.disabled = true;

  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${scheme}://${window.location.host}/ws/run`);

  ws.onopen = () => {
    addLog('Connected. Starting pipeline...');
    ws.send(JSON.stringify(getConfig()));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'pipeline_start') {
      stages = msg.stages.map((name) => ({ name, status: 'pending' }));
      renderStages();
      addLog('Pipeline started.');
      if (stages.length > 0) {
        currentStageName = stages[0].name;
        renderDetailForStage(currentStageName);
      }
      return;
    }

    if (msg.type === 'stage_start') {
      const idx = msg.index - 1;
      stages[idx].status = 'running';
      stageDetails[msg.stage] = stageDetails[msg.stage] || {};
      stageDetails[msg.stage].explain = msg.explain || '';
      currentStageName = msg.stage;
      renderStages();
      renderDetailForStage(currentStageName);
      addLog(`$ ${msg.command}`);
      return;
    }

    if (msg.type === 'log') {
      addLog(msg.line);
      return;
    }

    if (msg.type === 'metric') {
      const d = msg.data || {};
      setMetric(synMapeEl, d.synthetic_mape, '%');
      setMetric(synRmseEl, d.synthetic_rmse);
      setMetric(baseMapeEl, d.baseline_mape, '%');
      setMetric(baseRmseEl, d.baseline_rmse);
      setMetric(rowsEl, d.rows);
      setMetric(outDaysEl, d.days);
      setMetric(totalKwhEl, d.total_kwh);
      return;
    }

    if (msg.type === 'stage_done') {
      const idx = msg.index - 1;
      stages[idx].status = msg.ok ? 'done' : 'failed';
      stageDetails[msg.stage] = stageDetails[msg.stage] || {};
      stageDetails[msg.stage].summary = msg.summary || {};
      currentStageName = msg.stage;
      renderStages();
      renderDetailForStage(currentStageName);
      addLog(`[${msg.ok ? 'OK' : 'FAILED'}] ${msg.stage}`);
      return;
    }

    if (msg.type === 'pipeline_done') {
      runStatusEl.textContent = msg.ok ? 'Completed' : 'Failed';
      addLog(msg.ok ? 'Pipeline completed.' : 'Pipeline failed.');
      runBtn.disabled = false;
      return;
    }

    if (msg.type === 'error') {
      runStatusEl.textContent = 'Error';
      addLog(`Error: ${msg.message}`);
      runBtn.disabled = false;
    }
  };

  ws.onclose = () => {
    runBtn.disabled = false;
  };

  ws.onerror = () => {
    runStatusEl.textContent = 'Error';
    addLog('WebSocket error');
    runBtn.disabled = false;
  };
}

runBtn.addEventListener('click', startRun);
