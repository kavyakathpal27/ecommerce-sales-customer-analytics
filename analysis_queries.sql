-- Analysis queries for the e-commerce sales & customer analytics star schema
-- (data/processed/ecommerce.db). Uses CTEs and window functions throughout.
-- Verified against SQLite; portable to Postgres/T-SQL with trivial
-- date-function substitutions.

-- ---------------------------------------------------------------------------
-- 1) RFM base metrics per customer (Recency / Frequency / Monetary), with
--    quartile scoring via window function NTILE()
-- ---------------------------------------------------------------------------
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_date,
        o.order_value
    FROM fact_orders o
    JOIN dim_customer c ON c.customer_unique_id = o.customer_unique_id
    WHERE o.order_status = 'delivered'
),
rfm_base AS (
    SELECT
        customer_unique_id,
        MAX(order_purchase_date)                              AS last_order_date,
        COUNT(DISTINCT order_id)                               AS frequency,
        ROUND(SUM(order_value), 2)                              AS monetary,
        CAST(JULIANDAY((SELECT MAX(order_purchase_date) FROM customer_orders)) -
             JULIANDAY(MAX(order_purchase_date)) AS INTEGER)    AS recency_days
    FROM customer_orders
    GROUP BY customer_unique_id
)
SELECT
    customer_unique_id,
    recency_days,
    frequency,
    monetary,
    NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,   -- more recent = higher score
    NTILE(4) OVER (ORDER BY frequency ASC)     AS f_score,
    NTILE(4) OVER (ORDER BY monetary ASC)      AS m_score
FROM rfm_base
ORDER BY monetary DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- 2) Revenue concentration: cumulative revenue share by customer, ranked
--    richest-first (window functions: SUM() OVER + PERCENT_RANK()) --
--    basis for the Pareto / Gini analysis done in Python at full precision.
-- ---------------------------------------------------------------------------
WITH customer_revenue AS (
    SELECT c.customer_unique_id, ROUND(SUM(o.order_value), 2) AS total_revenue
    FROM fact_orders o
    JOIN dim_customer c ON c.customer_unique_id = o.customer_unique_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
ranked AS (
    SELECT
        customer_unique_id,
        total_revenue,
        PERCENT_RANK() OVER (ORDER BY total_revenue DESC) AS customer_rank_pct,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_revenue,
        SUM(total_revenue) OVER ()                          AS grand_total_revenue
    FROM customer_revenue
)
SELECT
    customer_unique_id,
    total_revenue,
    ROUND(100.0 * customer_rank_pct, 2)                 AS pct_of_customers_above,
    ROUND(100.0 * cum_revenue / grand_total_revenue, 2)  AS cum_pct_of_revenue
FROM ranked
ORDER BY total_revenue DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- 3) Product category revenue & margin, ranked (window function: RANK())
-- ---------------------------------------------------------------------------
WITH category_perf AS (
    SELECT
        p.product_category_name,
        COUNT(*)                                    AS items_sold,
        ROUND(SUM(oi.price), 2)                       AS revenue,
        ROUND(SUM(oi.item_margin), 2)                 AS total_margin,
        ROUND(100.0 * SUM(oi.item_margin) / NULLIF(SUM(oi.price), 0), 2) AS margin_pct
    FROM fact_order_items oi
    JOIN dim_product p ON p.product_id = oi.product_id
    GROUP BY p.product_category_name
)
SELECT
    product_category_name,
    items_sold,
    revenue,
    total_margin,
    margin_pct,
    RANK() OVER (ORDER BY revenue DESC)    AS revenue_rank,
    RANK() OVER (ORDER BY margin_pct DESC) AS margin_pct_rank
FROM category_perf
ORDER BY revenue DESC;

-- ---------------------------------------------------------------------------
-- 4) Delivery delay & review score by customer state (input to the
--    confound check performed in Python)
-- ---------------------------------------------------------------------------
SELECT
    c.customer_state,
    COUNT(*)                                          AS delivered_orders,
    ROUND(AVG(o.delivery_delay_days), 2)                AS avg_delivery_delay_days,
    ROUND(AVG(r.review_score), 2)                        AS avg_review_score
FROM fact_orders o
JOIN dim_customer c ON c.customer_unique_id = o.customer_unique_id
JOIN fact_reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY avg_delivery_delay_days DESC;

-- ---------------------------------------------------------------------------
-- 5) Payment type comparison: average order value & installment usage
--    (window function: RANK())
-- ---------------------------------------------------------------------------
WITH payment_summary AS (
    SELECT
        pay.payment_type,
        COUNT(DISTINCT pay.order_id)                    AS orders,
        ROUND(AVG(pay.payment_value), 2)                  AS avg_payment_value,
        ROUND(AVG(pay.payment_installments), 2)           AS avg_installments
    FROM fact_payments pay
    GROUP BY pay.payment_type
)
SELECT
    payment_type,
    orders,
    avg_payment_value,
    avg_installments,
    RANK() OVER (ORDER BY avg_payment_value DESC) AS value_rank
FROM payment_summary
ORDER BY avg_payment_value DESC;

-- ---------------------------------------------------------------------------
-- 6) Monthly cohort retention -- each customer's cohort is their first
--    order month; for every (cohort, activity) month pair, count distinct
--    active customers. Uses MIN() OVER as a window function (one pass over
--    the data) rather than a correlated self-join per customer, which the
--    hospital-analytics sibling project found takes 15+ minutes at this
--    scale even with indexes. Retention-rate % is computed in Python
--    (reports/findings.json / dashboard) from this query's output.
-- ---------------------------------------------------------------------------
WITH customer_month AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        strftime('%Y-%m', o.order_purchase_date) AS order_month
    FROM fact_orders o
    JOIN dim_customer c ON c.customer_unique_id = o.customer_unique_id
    WHERE o.order_status = 'delivered'
),
cohort_assignment AS (
    SELECT
        customer_unique_id,
        order_month,
        MIN(order_month) OVER (PARTITION BY customer_unique_id) AS cohort_month
    FROM customer_month
),
cohort_activity AS (
    SELECT
        cohort_month,
        order_month,
        -- month index = number of calendar months between cohort and activity month
        (CAST(SUBSTR(order_month, 1, 4) AS INTEGER) * 12 + CAST(SUBSTR(order_month, 6, 2) AS INTEGER))
        - (CAST(SUBSTR(cohort_month, 1, 4) AS INTEGER) * 12 + CAST(SUBSTR(cohort_month, 6, 2) AS INTEGER)) AS month_index,
        COUNT(DISTINCT customer_unique_id) AS active_customers
    FROM cohort_assignment
    GROUP BY cohort_month, order_month
),
cohort_size AS (
    SELECT cohort_month, active_customers AS cohort_customers
    FROM cohort_activity
    WHERE month_index = 0
)
SELECT
    ca.cohort_month,
    ca.month_index,
    ca.active_customers,
    cs.cohort_customers,
    ROUND(100.0 * ca.active_customers / cs.cohort_customers, 1) AS retention_pct
FROM cohort_activity ca
JOIN cohort_size cs ON cs.cohort_month = ca.cohort_month
ORDER BY ca.cohort_month, ca.month_index;

-- ---------------------------------------------------------------------------
-- 7) At-risk / churned customer revenue: customers whose most recent
--    delivered order predates the dataset's effective "current date" by
--    more than 180 days (window function: MAX() OVER, no self-join)
-- ---------------------------------------------------------------------------
WITH customer_last_order AS (
    SELECT
        c.customer_unique_id,
        MAX(o.order_purchase_date)                                        AS last_order_date,
        ROUND(SUM(o.order_value), 2)                                       AS lifetime_value,
        (SELECT MAX(order_purchase_date) FROM fact_orders WHERE order_status = 'delivered') AS dataset_end_date
    FROM fact_orders o
    JOIN dim_customer c ON c.customer_unique_id = o.customer_unique_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT
    customer_unique_id,
    last_order_date,
    lifetime_value,
    CAST(JULIANDAY(dataset_end_date) - JULIANDAY(last_order_date) AS INTEGER) AS days_since_last_order,
    CASE WHEN JULIANDAY(dataset_end_date) - JULIANDAY(last_order_date) > 180 THEN 1 ELSE 0 END AS is_at_risk
FROM customer_last_order
ORDER BY days_since_last_order DESC
LIMIT 20;
