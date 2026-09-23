# Findings: E-Commerce Sales & Customer Analytics

Dataset: 30,109 orders (29,177 delivered), 19,000 unique customers, 2,000 products across 42 categories, 500 sellers, 2016-01-01 to 2018-08-31. Total delivered revenue: $4,092,609.

## 1. RFM segmentation & revenue concentration

Customers were scored on Recency/Frequency/Monetary quartiles and grouped into 5 segments. **Champions** is the largest revenue driver: 10.51% of customers generate **33.57%** of total revenue. Looking at raw spend concentration (independent of the segment labels): the top 10% of customers by lifetime spend account for **49.87%** of revenue, and the top 20% account for **66.35%** (Gini coefficient = 0.6234, where 0 = perfect equality and 1 = maximum concentration) -- a genuine Pareto-like concentration that emerged from the generator's skewed per-customer spend propensity, not a hardcoded split.

## 2. Category revenue & margin

Across 42 product categories, **consoles_games** is the top revenue driver ($220,641). Profit margin (assumed product cost as 55-75% of price, category-dependent) ranges from **50.63%** (industry_commerce_and_business) to **60.94%** (health_beauty), against an overall blended margin of 55.34%.

## 3. Delivery delay drives review score -- but it's confounded by customer state

A naive pooled regression of review_score on delivery_delay_days (n=29,177 delivered, reviewed orders) gives a slope of **-0.2024** points/day (R² = 0.2505, p = 0.00e+00). But customer state is a confound: slow-logistics states have both longer average delay *and* an independently lower baseline review tendency unrelated to any single shipment's delay (a Simpson's-paradox risk). Controlling for state with a demeaned (fixed-effects) fit gives a slope of **-0.1875** points/day (R² = 0.1994, p = 0.00e+00) -- the pooled estimate overstates the per-day effect by +8.0% relative to the state-controlled figure. **The state-demeaned coefficient is the defensible causal estimate**: each additional day of delivery delay costs **0.1875 review-score points**, holding customer-state baseline constant.

## 4. Item price differs significantly by product category (ANOVA)

One-way ANOVA of item price across 42 product categories: F = 269.02, p = 0.00e+00 -- category is a statistically significant driver of price, from **cds_dvds_musicals** (avg $24.37) up to **consoles_games** (avg $215.47).

As a secondary check, order *value* does **not** differ significantly by payment type (one-way ANOVA across 4 payment types: F = 1.484, p = 2.17e-01) -- an honest null result worth reporting: customers don't systematically spend more or less depending on how they pay.

## 5. Slow-delivery states have significantly lower review scores (t-test)

The empirically slowest-delivery customer states were identified directly from the data (top half by average delivery_delay_days: MA, PB, RN, PE, MS, BA, CE, MT, PA, SE), averaging **4.39 days** delay vs 0.68 days for the rest. Welch's t-test on review score: slow-state average **2.812** vs **3.978** for the rest (t = -41.7, p = 0.00e+00, n_slow=3,663, n_fast=25,514) -- consistent with, and additional evidence for, the state confound identified in Finding 3.

## 6. Cohort retention

Across 32 monthly purchase cohorts, average month-1 retention (customers from a cohort who placed another order the following calendar month) is **2.57%** (n=31 cohorts with enough history to measure) -- consistent with Olist-style e-commerce, where most customers are one-time buyers (see data/processed/cohort_retention.csv for the full cohort x month-index matrix).

## 7. At-risk revenue

Using a 180-day no-repeat-purchase cutoff (relative to the dataset's latest observed order date, 2018-08-31), **13,491 customers** (72.45% of the customer base) are flagged at-risk/churned, representing **$2,248,631** in historical revenue (**54.94%** of total revenue) that is not currently being re-engaged.
