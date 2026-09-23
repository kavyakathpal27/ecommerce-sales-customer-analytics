# Power BI Model — E-Commerce Sales & Customer Analytics

This folder documents the Power BI data model built on top of `data/processed/`.
Power BI Desktop is a GUI application, so the `.pbix` itself isn't checked into
source control here — instead this is the exact spec to rebuild it in ~15-20
minutes: which tables to import, how to relate them, and the DAX measure
library to paste in.

## 1. Get Data

In Power BI Desktop: **Get Data → Text/CSV**, import each file from
`data/processed/`:

| File                          | Role in model                                             |
|--------------------------------|-------------------------------------------------------------|
| `dim_customer.csv`            | Dimension (1 row / customer_unique_id)                     |
| `dim_product.csv`             | Dimension (1 row / product, incl. synthetic `product_cost`) |
| `dim_seller.csv`               | Dimension (1 row / seller)                                  |
| `dim_date.csv`                 | Dimension (calendar)                                        |
| `fact_orders.csv`              | Fact (grain: 1 row / order)                                  |
| `fact_order_items.csv`        | Fact (grain: 1 row / order line item)                        |
| `fact_payments.csv`           | Fact (grain: 1 row / payment installment/split)              |
| `fact_reviews.csv`             | Fact (grain: 1 row / review)                                 |
| `customer_rfm.csv`             | Pre-aggregated (1 row / customer; R/F/M scores + segment)    |
| `rfm_segment_summary.csv`      | Pre-aggregated (1 row / RFM segment)                          |
| `category_summary.csv`         | Pre-aggregated (1 row / product category; revenue + margin)  |
| `cohort_retention.csv`         | Pre-aggregated (1 row / cohort_month x month_index)           |
| `monthly_trend.csv`            | Pre-aggregated (1 row / month)                                |
| `state_delay_review.csv`       | Pre-aggregated (1 row / customer_state)                        |
| `payment_summary.csv`          | Pre-aggregated (1 row / payment_type)                          |
| `at_risk_customers.csv`        | Pre-aggregated (1 row / at-risk customer)                       |

Load all as **Import** mode (dataset is small; no need for DirectQuery).

## 2. Model relationships

In the Model view, create these relationships (all Many-to-One, single
direction unless noted):

- `fact_orders[customer_unique_id]` → `dim_customer[customer_unique_id]`
- `fact_order_items[order_id]` → `fact_orders[order_id]`
- `fact_order_items[product_id]` → `dim_product[product_id]`
- `fact_order_items[seller_id]` → `dim_seller[seller_id]`
- `fact_payments[order_id]` → `fact_orders[order_id]`
- `fact_reviews[order_id]` → `fact_orders[order_id]`
- `fact_orders[order_purchase_date]` → `dim_date[date_key]`
- `customer_rfm[customer_unique_id]` → `dim_customer[customer_unique_id]`

Mark `dim_date` as a **Date Table** (Table tools → Mark as date table →
`date_key` column) so time-intelligence DAX functions work correctly.

Note: `fact_order_items` and `fact_payments` both relate many-to-one back to
`fact_orders` (not directly to `dim_customer`) — this keeps the model a
proper star with `fact_orders` as the central "order" grain fact and the
line-item/payment/review facts as satellites, avoiding a fan-trap when
visuals mix item-level and payment-level measures.

## 3. DAX measures

Create a dedicated measures table (New Table → name it `_Measures`, any
single column, hide it) and add these:

```dax
Total Orders = COUNTROWS(fact_orders)

Total Revenue = SUM(fact_order_items[price]) + SUM(fact_order_items[freight_value])

Total Customers = DISTINCTCOUNT(dim_customer[customer_unique_id])

Avg Order Value = DIVIDE([Total Revenue], [Total Orders])

Total Margin = SUM(fact_order_items[item_margin])

Margin Pct = DIVIDE([Total Margin], SUM(fact_order_items[price]))

Avg Review Score = AVERAGE(fact_reviews[review_score])

Avg Delivery Delay (days) = AVERAGE(fact_orders[delivery_delay_days])

-- Revenue concentration (reads from the pre-aggregated customer_rfm table,
-- since a true Pareto/Gini calc isn't expressible efficiently in native DAX
-- over unaggregated order-level data at this scale)
Top Segment Revenue Pct =
CALCULATE(
    DIVIDE(SUM(rfm_segment_summary[total_revenue]), [Total Revenue]),
    rfm_segment_summary[segment] = "Champions"
)

Pct Customers At Risk =
DIVIDE(COUNTROWS(at_risk_customers), [Total Customers])

At Risk Revenue = SUM(at_risk_customers[monetary])

-- Time intelligence (works because dim_date is marked as a date table)
Revenue MTD = TOTALMTD([Total Revenue], dim_date[date_key])

Revenue Prior Month = CALCULATE([Total Revenue], DATEADD(dim_date[date_key], -1, MONTH))

Revenue MoM Growth = DIVIDE([Total Revenue] - [Revenue Prior Month], [Revenue Prior Month])
```

## 4. Suggested pages / visuals

1. **Executive Overview** — KPI cards (Total Revenue, Total Orders, Total
   Customers, Avg Order Value, Avg Review Score); a line chart of monthly
   revenue (`monthly_trend.csv`) with the Nov/Dec seasonality visible; a
   table of RFM segments ranked by `pct_of_revenue` from
   `rfm_segment_summary.csv`.
2. **Customer & RFM** — scatter plot of `recency_days` (x) vs `monetary`
   (y) from `customer_rfm.csv`, colored by `segment`; a Pareto/cumulative
   revenue chart (bar + line combo) showing the top-decile revenue share;
   a table of `at_risk_customers.csv` sorted by `monetary` descending, to
   prioritize win-back outreach.
3. **Delivery & Reviews** — bar chart of `avg_review_score` and
   `avg_delivery_delay_days` by `customer_state` from
   `state_delay_review.csv`, to visually reproduce the confound finding in
   `reports/findings.md`; a scatter of delivery delay vs review score at
   the order level (from `fact_orders` + `fact_reviews`), colored by
   customer state.
4. **Category & Margin** — bar chart of `revenue` and `margin_pct` by
   `product_category_name` from `category_summary.csv`, sorted by revenue,
   to identify high-revenue/low-margin categories worth a pricing review.
5. **Cohorts** — a matrix visual of `cohort_retention.csv`
   (`cohort_month` on rows, `month_index` on columns, `retention_pct` as
   values, conditional-formatted) — the classic cohort-retention heatmap.

Use **drill-through**: right-click a state bar on the Delivery & Reviews
page → set up drill-through to a state detail page filtered by
`dim_customer[customer_state]`, showing that state's order-level delay and
review distribution.

## 5. Why the source data doesn't include a live Power BI file

Power BI Desktop is a Windows GUI application with no scriptable CLI for
building `.pbix` files from a coding agent, so this repo ships the exact
model spec + pre-aggregated CSVs instead of a binary `.pbix`. If you have
Power BI Desktop installed, following steps 1-4 above reproduces the
intended dashboard in well under 30 minutes. An equivalent live,
browser-viewable KPI dashboard covering the same metrics is also included
at `reports/dashboard.html` for an immediate visual preview.
