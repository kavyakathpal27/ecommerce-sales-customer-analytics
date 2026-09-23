"""
Builds reports/dashboard.html: a self-contained, static KPI dashboard (no
server, no fetch) that mirrors the Power BI model in powerbi/README.md, for
an immediate visual preview of the project's headline findings.

All chart data is baked in as JSON at build time from data/processed/*.csv
and reports/findings.json -- open the HTML file directly in a browser.

Chart.js version pinned below was verified live against cdnjs
(https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.1/chart.umd.min.js
returns HTTP 200; cross-checked against api.cdnjs.com/libraries/Chart.js)
before hardcoding, per the dataviz skill's guidance to never assume a CDN
path resolves.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

RNG_SEED = 7

# Fixed categorical color slots (validated palette, dataviz skill), assigned
# to RFM segments in a stable identity order and reused across every chart
# that shows segment identity in this dashboard -- never re-cycled by rank.
SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk (was frequent)", "Hibernating / Lost"]
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL_LIGHT = "#2a78d6"
SEQUENTIAL_DARK = "#3987e5"


def build_delay_review_scatter():
    """Re-derive the (state-demeaned) delivery-delay-vs-review-score dataset
    directly from the processed star schema, matching src/analysis.py's
    delay_review_regression() method, for an order-level scatter."""
    orders = pd.read_csv(PROCESSED_DIR / "fact_orders.csv")
    reviews = pd.read_csv(PROCESSED_DIR / "fact_reviews.csv")
    customers = pd.read_csv(PROCESSED_DIR / "dim_customer.csv")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    df = (
        delivered.merge(reviews[["order_id", "review_score"]], on="order_id", how="inner")
        .merge(customers[["customer_unique_id", "customer_state"]], on="customer_unique_id", how="left")
        .dropna(subset=["delivery_delay_days", "review_score", "customer_state"])
    )
    state_x_mean = df.groupby("customer_state")["delivery_delay_days"].transform("mean")
    state_y_mean = df.groupby("customer_state")["review_score"].transform("mean")
    df["x_demeaned"] = df["delivery_delay_days"] - state_x_mean
    df["y_demeaned"] = df["review_score"] - state_y_mean
    return df


def main():
    findings = json.loads((REPORTS_DIR / "findings.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(RNG_SEED)

    color_map_light = {seg: CATEGORICAL["light"][i % 8] for i, seg in enumerate(SEGMENT_ORDER)}
    color_map_dark = {seg: CATEGORICAL["dark"][i % 8] for i, seg in enumerate(SEGMENT_ORDER)}

    seg_df = pd.DataFrame(findings["rfm_segmentation"]["segments"])
    seg_df = seg_df.set_index("segment").reindex(SEGMENT_ORDER).reset_index()

    lorenz = findings["revenue_concentration"]["lorenz_curve"]

    cat_df = pd.read_csv(PROCESSED_DIR / "category_summary.csv").sort_values("revenue", ascending=False).head(10)

    monthly = pd.read_csv(PROCESSED_DIR / "monthly_trend.csv")

    cohort = pd.read_csv(PROCESSED_DIR / "cohort_retention.csv")
    cohort_curve = (
        cohort[cohort["month_index"] <= 8]
        .groupby("month_index")
        .agg(avg_retention=("retention_pct", "mean"), n_cohorts=("cohort_month", "count"))
        .reset_index()
    )
    # Keep only month_index values with a reasonably-sized cohort sample
    cohort_curve = cohort_curve[cohort_curve["n_cohorts"] >= 5]

    scatter_df = build_delay_review_scatter()
    reg = findings["delay_review_regression"]
    sample = scatter_df.sample(n=min(1000, len(scatter_df)), random_state=RNG_SEED)
    scatter_points = [{"x": round(r.x_demeaned, 2), "y": round(r.y_demeaned, 2)} for r in sample.itertuples()]
    x_min, x_max = float(scatter_df["x_demeaned"].min()), float(scatter_df["x_demeaned"].max())
    slope = reg["state_demeaned_slope_per_day"]
    reg_line = [{"x": round(x_min, 2), "y": round(slope * x_min, 2)},
                {"x": round(x_max, 2), "y": round(slope * x_max, 2)}]

    ov = findings["dataset_overview"]
    conc = findings["revenue_concentration"]
    risk = findings["at_risk_revenue"]
    tt = findings["ttest_slow_states_review_score"]
    anova_cat = findings["anova_price_by_category"]
    cohort_finding = findings["cohort_retention"]

    data = {
        "segmentOrder": SEGMENT_ORDER,
        "colorLight": color_map_light,
        "colorDark": color_map_dark,
        "kpis": {
            "totalOrders": ov["total_orders"],
            "deliveredOrders": ov["delivered_orders"],
            "totalCustomers": ov["total_customers"],
            "totalRevenue": ov["total_revenue"],
            "dateRange": ov["date_range"],
            "topSegmentPctRevenue": findings["rfm_segmentation"]["top_segment_pct_of_revenue"],
            "top10PctRevenueShare": conc["top_10pct_customers_revenue_share"],
            "gini": conc["gini_coefficient"],
            "atRiskPctRevenue": risk["pct_of_revenue_at_risk"],
            "atRiskRevenue": risk["at_risk_revenue"],
            "avgReviewScore": round(float(pd.read_csv(PROCESSED_DIR / "fact_reviews.csv")["review_score"].mean()), 2),
        },
        "segmentRevenue": [
            {"segment": r.segment, "revenue": r.total_revenue, "pctOfRevenue": r.pct_of_revenue,
             "pctOfCustomers": r.pct_of_customers}
            for r in seg_df.itertuples()
        ],
        "lorenz": lorenz,
        "categoryChart": [
            {"name": r.product_category_name, "revenue": r.revenue, "marginPct": r.margin_pct}
            for r in cat_df.itertuples()
        ],
        "monthlyTrend": [
            {"month": r.order_month, "revenue": r.revenue, "orders": r.order_count}
            for r in monthly.itertuples()
        ],
        "cohortCurve": [
            {"monthIndex": int(r.month_index), "avgRetention": round(float(r.avg_retention), 1)}
            for r in cohort_curve.itertuples()
        ],
        "scatter": scatter_points,
        "regressionLine": reg_line,
        "regression": {
            "pooledSlope": reg["pooled_slope_per_day"],
            "slope": reg["state_demeaned_slope_per_day"],
            "r2": reg["state_demeaned_r_squared"],
            "p": reg["state_demeaned_p_value"],
            "n": reg["n_orders"],
        },
        "narrative": {
            "topSegment": findings["rfm_segmentation"]["top_segment_name"],
            "slowStatesReview": tt["slow_states_avg_review"],
            "fastStatesReview": tt["fast_states_avg_review"],
            "anovaF": anova_cat["f_statistic"],
            "month1Retention": cohort_finding["avg_month1_retention_pct"],
        },
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    out_path = REPORTS_DIR / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out_path.relative_to(ROOT)}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-Commerce Sales &amp; Customer Analytics Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.1/chart.umd.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;
    --good:           #0ca30c;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --good:           #0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --good:           #0ca30c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 16px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
  h1 { font-size: 20px; margin: 0; }
  .subtitle { color: var(--text-secondary); font-size: 13px; margin: 4px 0 20px; }
  .theme-toggle {
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; cursor: pointer;
  }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .kpi {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .kpi .label { font-size: 12px; color: var(--text-secondary); }
  .kpi .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .kpi .sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
  .card {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px;
  }
  .card h2 { font-size: 14px; margin: 0 0 2px; }
  .card .desc { font-size: 12px; color: var(--text-secondary); margin: 0 0 12px; }
  .card canvas { max-height: 280px; }
  .legend-row { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--text-secondary); margin: 0 0 10px; }
  .legend-row span.dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  .findings { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin-top: 14px; }
  .findings h2 { font-size: 14px; margin: 0 0 10px; }
  .findings ul { margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.7; color: var(--text-secondary); }
  .findings li b { color: var(--text-primary); }
  footer { text-align: center; font-size: 11px; color: var(--text-muted); margin: 24px 0 8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>E-Commerce Sales &amp; Customer Analytics</h1>
    <button class="theme-toggle" id="themeToggle">Toggle dark mode</button>
  </header>
  <p class="subtitle" id="subtitle"></p>

  <div class="kpi-row" id="kpiRow"></div>

  <div class="grid">
    <div class="card">
      <h2>Revenue by RFM segment</h2>
      <p class="desc">Segment colors are fixed identities, reused in the customer legend below.</p>
      <div class="legend-row" id="segmentLegend"></div>
      <canvas id="segmentChart"></canvas>
    </div>
    <div class="card">
      <h2>Revenue concentration (Lorenz curve)</h2>
      <p class="desc">Cumulative % of revenue (y) held by the bottom X% of customers by spend (x); dashed line = perfect equality.</p>
      <canvas id="lorenzChart"></canvas>
    </div>
    <div class="card">
      <h2>Delivery delay vs. review score (state-demeaned)</h2>
      <p class="desc" id="scatterDesc"></p>
      <canvas id="scatterChart"></canvas>
    </div>
    <div class="card">
      <h2>Monthly revenue trend</h2>
      <p class="desc">Nov/Dec seasonality bump + gradual platform growth over the dataset's 2.5 years.</p>
      <canvas id="trendChart"></canvas>
    </div>
    <div class="card">
      <h2>Cohort retention curve</h2>
      <p class="desc">Average % of a monthly cohort still placing orders N months after their first purchase.</p>
      <canvas id="cohortChart"></canvas>
    </div>
    <div class="card" style="grid-column: 1 / -1;">
      <h2>Top 10 product categories by revenue &amp; margin</h2>
      <p class="desc">Bars = revenue (left axis via bar length); label = profit margin % per category.</p>
      <canvas id="categoryChart"></canvas>
    </div>
  </div>

  <div class="findings">
    <h2>Key findings</h2>
    <ul id="findingsList"></ul>
  </div>

  <footer>Synthetic dataset generated for this project (see README.md) &middot; Built with Python/Pandas/SciPy + Chart.js</footer>
</div>

<script>
const DATA = __DATA__;

function isDark() {
  const stamp = document.documentElement.getAttribute('data-theme');
  if (stamp === 'dark') return true;
  if (stamp === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}
function segColor(name) {
  const map = isDark() ? DATA.colorDark : DATA.colorLight;
  return map[name] || '#898781';
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

document.getElementById('subtitle').textContent =
  `${DATA.kpis.totalOrders.toLocaleString()} orders (${DATA.kpis.deliveredOrders.toLocaleString()} delivered), ` +
  `${DATA.kpis.totalCustomers.toLocaleString()} customers, ${DATA.kpis.dateRange[0]} to ${DATA.kpis.dateRange[1]} (synthetic data)`;

const kpiRow = document.getElementById('kpiRow');
const kpis = [
  { label: 'Total revenue', value: `$${DATA.kpis.totalRevenue.toLocaleString(undefined,{maximumFractionDigits:0})}`, sub: `${DATA.kpis.totalOrders.toLocaleString()} orders` },
  { label: 'Total customers', value: DATA.kpis.totalCustomers.toLocaleString(), sub: 'unique customer_unique_id' },
  { label: 'Top segment revenue', value: `${DATA.kpis.topSegmentPctRevenue}%`, sub: `from Champions` },
  { label: 'Top 10% of customers', value: `${DATA.kpis.top10PctRevenueShare}%`, sub: `of revenue (Gini ${DATA.kpis.gini})` },
  { label: 'At-risk revenue', value: `${DATA.kpis.atRiskPctRevenue}%`, sub: `$${DATA.kpis.atRiskRevenue.toLocaleString(undefined,{maximumFractionDigits:0})} (180d+ no reorder)` },
  { label: 'Avg review score', value: `${DATA.kpis.avgReviewScore} / 5`, sub: 'all delivered reviews' },
];
kpiRow.innerHTML = kpis.map(k => `
  <div class="kpi">
    <div class="label">${k.label}</div>
    <div class="value">${k.value}</div>
    <div class="sub">${k.sub}</div>
  </div>`).join('');

document.getElementById('segmentLegend').innerHTML = DATA.segmentOrder.map(s =>
  `<span><span class="dot" style="background:${segColor(s)}"></span>${s}</span>`
).join('');

document.getElementById('scatterDesc').textContent =
  `Each point = one delivered, reviewed order; delay & score demeaned by customer state to remove the ` +
  `state-level confound. Slope ${DATA.regression.slope} pts/day (pooled, unconfounded: ${DATA.regression.pooledSlope}), ` +
  `R²=${DATA.regression.r2}, p<0.001, n=${DATA.regression.n.toLocaleString()}.`;

const gridColor = () => cssVar('--gridline');
const tickColor = () => cssVar('--text-muted');
const textColor = () => cssVar('--text-primary');

Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";
Chart.defaults.color = tickColor();

function baseScales(extra) {
  return Object.assign({
    x: { grid: { color: gridColor() }, ticks: { color: tickColor() } },
    y: { grid: { color: gridColor() }, ticks: { color: tickColor() }, beginAtZero: true },
  }, extra || {});
}

const charts = [];

function buildSegmentChart() {
  const ctx = document.getElementById('segmentChart');
  const sorted = [...DATA.segmentRevenue].sort((a,b) => b.revenue - a.revenue);
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(d => d.segment),
      datasets: [{
        label: 'Revenue ($)',
        data: sorted.map(d => d.revenue),
        backgroundColor: sorted.map(d => segColor(d.segment)),
        borderRadius: 4,
        barThickness: 26,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `$${c.raw.toLocaleString()} (${sorted[c.dataIndex].pctOfRevenue}% of revenue)` } },
      },
      scales: baseScales(),
    },
  });
  charts.push(chart);
}

function buildLorenzChart() {
  const ctx = document.getElementById('lorenzChart');
  const points = DATA.lorenz.map(p => ({ x: p.cum_pct_customers * 100, y: p.cum_pct_revenue * 100 }));
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Actual revenue distribution',
          data: points,
          borderColor: cssVar('--series-1'),
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: 'Perfect equality',
          data: [{x:0,y:0},{x:100,y:100}],
          borderColor: cssVar('--text-muted'),
          borderWidth: 2,
          borderDash: [6,4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { color: tickColor() } } },
      scales: baseScales({
        x: { type: 'linear', grid: { color: gridColor() }, ticks: { color: tickColor() }, title: { display: true, text: '% of customers (lowest spend -> highest)', color: tickColor() } },
        y: { grid: { color: gridColor() }, ticks: { color: tickColor() }, title: { display: true, text: '% of revenue', color: tickColor() } },
      }),
    },
  });
  charts.push(chart);
}

function buildScatterChart() {
  const ctx = document.getElementById('scatterChart');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Orders (sampled)',
          data: DATA.scatter,
          backgroundColor: cssVar('--series-1'),
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: 'Fitted regression (state-demeaned)',
          data: DATA.regressionLine,
          type: 'line',
          borderColor: cssVar('--text-primary'),
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { color: tickColor() } } },
      scales: baseScales({
        x: { grid: { color: gridColor() }, ticks: { color: tickColor() }, title: { display: true, text: 'Delivery delay, days (state-demeaned)', color: tickColor() } },
        y: { grid: { color: gridColor() }, ticks: { color: tickColor() }, title: { display: true, text: 'Review score (state-demeaned)', color: tickColor() } },
      }),
    },
  });
  charts.push(chart);
}

function buildTrendChart() {
  const ctx = document.getElementById('trendChart');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: DATA.monthlyTrend.map(d => d.month),
      datasets: [{
        label: 'Revenue ($)',
        data: DATA.monthlyTrend.map(d => d.revenue),
        borderColor: cssVar('--series-1'),
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: baseScales({
        x: { grid: { color: gridColor() }, ticks: { color: tickColor(), maxRotation: 60, minRotation: 60, autoSkip: true } },
        y: { grid: { color: gridColor() }, ticks: { color: tickColor() }, beginAtZero: true },
      }),
    },
  });
  charts.push(chart);
}

function buildCohortChart() {
  const ctx = document.getElementById('cohortChart');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: DATA.cohortCurve.map(d => `+${d.monthIndex}mo`),
      datasets: [{
        label: 'Avg retention %',
        data: DATA.cohortCurve.map(d => d.avgRetention),
        borderColor: cssVar('--series-1'),
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: baseScales({ y: { grid: { color: gridColor() }, ticks: { color: tickColor() }, beginAtZero: true, max: 100 } }),
    },
  });
  charts.push(chart);
}

function buildCategoryChart() {
  const ctx = document.getElementById('categoryChart');
  const sorted = [...DATA.categoryChart].sort((a,b) => b.revenue - a.revenue);
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(d => d.name),
      datasets: [{
        label: 'Revenue ($)',
        data: sorted.map(d => d.revenue),
        backgroundColor: cssVar('--series-1'),
        borderRadius: 4,
        barThickness: 20,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `$${c.raw.toLocaleString()} revenue, ${sorted[c.dataIndex].marginPct}% margin` } },
      },
      scales: baseScales(),
    },
  });
  charts.push(chart);
}

function buildFindings() {
  const n = DATA.narrative;
  const items = [
    `<b>${n.topSegment}</b> customers are the top revenue segment, driving <b>${DATA.kpis.topSegmentPctRevenue}%</b> of revenue; independently, the top 10% of customers by raw spend drive <b>${DATA.kpis.top10PctRevenueShare}%</b> of revenue (Gini=${DATA.kpis.gini}).`,
    `State-demeaned regression: each additional day of delivery delay costs <b>${Math.abs(DATA.regression.slope)} review-score points</b> (R²=${DATA.regression.r2}, p&lt;0.001) &mdash; the pooled (unconfounded-for) estimate of ${DATA.regression.pooledSlope} overstates this due to a customer-state confound.`,
    `Empirically slow-delivery states average <b>${n.slowStatesReview}</b> stars vs <b>${n.fastStatesReview}</b> for faster states (Welch's t-test, p&lt;0.001).`,
    `Item price differs significantly by product category (ANOVA F=${n.anovaF}, p&lt;0.001) &mdash; category is a genuine price/revenue driver.`,
    `<b>${DATA.kpis.atRiskPctRevenue}%</b> of historical revenue ($${DATA.kpis.atRiskRevenue.toLocaleString()}) sits with customers flagged at-risk (180+ days, no reorder).`,
    `Average month-1 cohort retention is only <b>${n.month1Retention}%</b> &mdash; most customers are one-time buyers, consistent with real Olist-style e-commerce.`,
  ];
  document.getElementById('findingsList').innerHTML = items.map(i => `<li>${i}</li>`).join('');
}

function rebuildAll() {
  charts.forEach(c => c.destroy());
  charts.length = 0;
  buildSegmentChart();
  buildLorenzChart();
  buildScatterChart();
  buildTrendChart();
  buildCohortChart();
  buildCategoryChart();
}

buildFindings();
rebuildAll();

document.getElementById('themeToggle').addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  document.getElementById('segmentLegend').innerHTML = DATA.segmentOrder.map(s =>
    `<span><span class="dot" style="background:${segColor(s)}"></span>${s}</span>`
  ).join('');
  setTimeout(rebuildAll, 0);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
