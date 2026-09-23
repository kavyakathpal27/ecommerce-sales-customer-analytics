-- Star schema for e-commerce sales & customer analytics (Olist-shape).
-- Loaded automatically by src/clean_data.py into data/processed/ecommerce.db (SQLite).
-- Written in ANSI-compatible SQL; also runs unmodified on PostgreSQL.

CREATE TABLE dim_customer (
    customer_unique_id      TEXT PRIMARY KEY,
    customer_city            TEXT,
    customer_state           TEXT NOT NULL,
    customer_zip_code_prefix INTEGER
);

CREATE TABLE dim_product (
    product_id               TEXT PRIMARY KEY,
    product_category_name    TEXT NOT NULL,
    product_weight_g         REAL,
    product_length_cm        REAL,
    product_height_cm        REAL,
    product_width_cm         REAL,
    product_cost              REAL NOT NULL
);

CREATE TABLE dim_seller (
    seller_id                TEXT PRIMARY KEY,
    seller_city               TEXT,
    seller_state              TEXT NOT NULL,
    seller_zip_code_prefix    INTEGER
);

CREATE TABLE dim_date (
    date_key         TEXT PRIMARY KEY,
    year              INTEGER NOT NULL,
    month_number     INTEGER NOT NULL,
    month_name       TEXT NOT NULL,
    day_name          TEXT NOT NULL,
    week_of_year     INTEGER NOT NULL,
    is_weekend        BOOLEAN NOT NULL,
    quarter           INTEGER NOT NULL
);

-- Grain: 1 row per order. customer_id (order-scoped) collapses to
-- customer_unique_id (the real person) via the original Olist customers map.
CREATE TABLE fact_orders (
    order_id                        TEXT PRIMARY KEY,
    customer_unique_id              TEXT NOT NULL REFERENCES dim_customer(customer_unique_id),
    order_status                     TEXT NOT NULL,
    order_purchase_timestamp         TIMESTAMP NOT NULL,
    order_purchase_date              DATE NOT NULL,
    order_approved_at                TIMESTAMP,
    order_delivered_carrier_date     TIMESTAMP,
    order_delivered_customer_date    TIMESTAMP,
    order_estimated_delivery_date    TIMESTAMP NOT NULL,
    delivery_delay_days              REAL,
    order_value                      REAL NOT NULL
);

-- Grain: 1 row per order line item.
CREATE TABLE fact_order_items (
    order_id                TEXT NOT NULL REFERENCES fact_orders(order_id),
    order_item_id            INTEGER NOT NULL,
    product_id                TEXT NOT NULL REFERENCES dim_product(product_id),
    seller_id                 TEXT NOT NULL REFERENCES dim_seller(seller_id),
    price                      REAL NOT NULL,
    freight_value              REAL NOT NULL,
    item_cost                  REAL NOT NULL,
    item_margin                REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);

-- Grain: 1 row per payment installment/split.
CREATE TABLE fact_payments (
    order_id                 TEXT NOT NULL REFERENCES fact_orders(order_id),
    payment_sequential        INTEGER NOT NULL,
    payment_type               TEXT NOT NULL,
    payment_installments       INTEGER NOT NULL,
    payment_value               REAL NOT NULL,
    PRIMARY KEY (order_id, payment_sequential)
);

-- Grain: 1 row per review.
CREATE TABLE fact_reviews (
    review_id                  TEXT PRIMARY KEY,
    order_id                    TEXT NOT NULL REFERENCES fact_orders(order_id),
    review_score                 INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
    review_creation_date          TIMESTAMP NOT NULL,
    review_answer_timestamp       TIMESTAMP NOT NULL
);

CREATE INDEX idx_orders_customer ON fact_orders(customer_unique_id);
CREATE INDEX idx_orders_date ON fact_orders(order_purchase_date);
CREATE INDEX idx_items_product ON fact_order_items(product_id);
CREATE INDEX idx_items_seller ON fact_order_items(seller_id);
CREATE INDEX idx_items_order ON fact_order_items(order_id);
CREATE INDEX idx_payments_order ON fact_payments(order_id);
CREATE INDEX idx_reviews_order ON fact_reviews(order_id);
