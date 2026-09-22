/* ---------------- dashboard price chart ---------------- */
async function loadPriceChart(canvasId, days = 180) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const res = await fetch(`/api/history?n=${days}`);
  const data = await res.json();

  new Chart(canvas, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        {
          label: 'BTC close (USD)',
          data: data.close,
          borderColor: '#f0b90b',
          backgroundColor: 'rgba(240,185,11,.12)',
          fill: true,
          tension: 0.25,
          pointRadius: 0,
          yAxisID: 'y',
        },
        {
          label: '|daily return| %',
          data: data.absret,
          borderColor: '#3ddc84',
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.25,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#e6e6e6' } } },
      scales: {
        x: { ticks: { color: '#8a8f98', maxTicksLimit: 8 } },
        y: { position: 'left', ticks: { color: '#f0b90b' }, grid: { color: '#1c2026' } },
        y1: { position: 'right', ticks: { color: '#3ddc84' }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

/* ---------------- analysis charts ---------------- */
function initAnalysisCharts() {
  const A = window.__ANALYTICS__;

  // Weekday chart
  const wd = document.getElementById('weekdayChart');
  if (wd) {
    new Chart(wd, {
      type: 'bar',
      data: {
        labels: A.weekday.map(d => d.day),
        datasets: [
          { label: 'Mean return %',   data: A.weekday.map(d => d.mean_ret),
            backgroundColor: 'rgba(240,185,11,.8)' },
          { label: 'Mean |move| %',   data: A.weekday.map(d => d.mean_abs),
            backgroundColor: 'rgba(61,220,132,.65)' },
        ],
      },
      options: {
        plugins: { legend: { labels: { color: '#e6e6e6' } } },
        scales: {
          x: { ticks: { color: '#8a8f98' }, grid: { color: '#1c2026' } },
          y: { ticks: { color: '#8a8f98' }, grid: { color: '#1c2026' } },
        },
      },
    });
  }

  // Fear & Greed chart
  const fg = document.getElementById('fgChart');
  if (fg && A.fear_greed && A.fear_greed.length) {
    new Chart(fg, {
      type: 'bar',
      data: {
        labels: A.fear_greed.map(d => d.label),
        datasets: [{
          label: 'Mean next-day return %',
          data: A.fear_greed.map(d => d.mean_ret),
          backgroundColor: A.fear_greed.map(d =>
            d.mean_ret >= 0 ? 'rgba(61,220,132,.75)' : 'rgba(255,107,107,.75)'),
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#8a8f98' }, grid: { color: '#1c2026' } },
          y: { ticks: { color: '#8a8f98' }, grid: { color: '#1c2026' } },
        },
      },
    });
  }
}

/* ---------------- What-if form ---------------- */
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const body = Object.fromEntries(fd.entries());

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const out = await res.json();
    if (!res.ok) { alert(out.error || 'Error'); return; }

    const f = out.forecast;
    document.getElementById('resultPlaceholder').classList.add('d-none');
    document.getElementById('resultPanel').classList.remove('d-none');

    document.getElementById('rExpected').textContent = f.expected_abs_return.toFixed(2);
    document.getElementById('rSigma').textContent    = f.sigma.toFixed(2);

    const fmt = r => `${r[0].toFixed(2)}% to +${r[1].toFixed(2)}%`;
    document.getElementById('r68').textContent = fmt(f.range_68);
    document.getElementById('r90').textContent = fmt(f.range_90);
    document.getElementById('r95').textContent = fmt(f.range_95);
  });
});