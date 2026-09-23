"""
ETL: clean the raw e-commerce extract and load a star schema into SQLite.

Cleaning steps (with before/after counts logged to reports/cleaning_log.json):
  1. Drop exact duplicate order_id / customer_id rows introduced by the
     (simulated) re-extract.
  2. Parse mixed-format datetime strings (ISO + US MM/DD/YYYY) into a single
     dtype (format="mixed" -- see note in analysis.py about why
     pd.read_sql(parse_dates=...) must NOT be used for these columns later).
  3. Strip "R$" currency symbols/commas from price/freight_value and cast to
     float.
  4. Drop rows with impossible (negative) price or freight_value.
  5. Normalize inconsistent payment_type casing.
  6. Impute missing customer_state as "Unknown" rather than dropping rows.
  7. Impute missing product_category_name / seller_city as "unknown".
  8. Collapse order-scoped customer_id to the real customer_unique_id.
  9. Merge the synthetic product_cost_reference into dim_product and compute
     per-line item_margin = price - item_cost.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "ecommerce.db"

log = {"steps": []}


def record(step, detail):
    log["steps"].append({"step": step, "detail": detail})
    print(f"[{step}] {detail}")


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    # format="mixed" infers the format per-element, correctly handling a
    # column containing both ISO ("2016-01-05 07:23:00") and US
    # ("01/15/2016 07:23:00") style strings. Never rely on
    # pd.read_csv/read_sql's own datetime inference for these columns -- it
    # locks onto the first row's format and silently NaTs the rest.
    return pd.to_datetime(series, format="mixed", errors="coerce")


def clean_currency(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[Rr]\$", "", regex=True).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def main():
    customers_raw = pd.read_csv(RAW_DIR / "customers.csv")
    orders_raw = pd.read_csv(RAW_DIR / "orders.csv")
    items_raw = pd.read_csv(RAW_DIR / "order_items.csv")
    payments_raw = pd.read_csv(RAW_DIR / "order_payments.csv")
    reviews_raw = pd.read_csv(RAW_DIR / "order_reviews.csv")
    products_raw = pd.read_csv(RAW_DIR / "products.csv")
    sellers_raw = pd.read_csv(RAW_DIR / "sellers.csv")
    cost_ref = pd.read_csv(RAW_DIR / "product_cost_reference.csv")

    record("load_raw", f"customers={len(customers_raw):,} orders={len(orders_raw):,} "
                        f"order_items={len(items_raw):,} order_payments={len(payments_raw):,} "
                        f"order_reviews={len(reviews_raw):,} products={len(products_raw):,} "
                        f"sellers={len(sellers_raw):,}")

    # ------------------------------------------------------------------
    # 1) Duplicates
    # ------------------------------------------------------------------
    before = len(orders_raw)
    orders = orders_raw.drop_duplicates(subset=["order_id"], keep="first").copy()
    record("drop_duplicate_orders", f"removed {before - len(orders):,} duplicate order_id rows "
                                     f"({(before - len(orders)) / before:.2%} of raw orders extract)")

    before = len(customers_raw)
    customers = customers_raw.drop_duplicates(subset=["customer_id"], keep="first").copy()
    record("drop_duplicate_customers", f"removed {before - len(customers):,} duplicate customer_id rows "
                                        f"({(before - len(customers)) / before:.2%} of raw customers extract)")

    # ------------------------------------------------------------------
    # 2) Datetimes (orders + reviews + shipping_limit_date)
    # ------------------------------------------------------------------
    order_dt_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                      "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in order_dt_cols:
        orders[col] = parse_mixed_datetime(orders[col])
    unparsed_orders = orders[order_dt_cols].isna().sum().sum()
    record("parse_order_datetimes", f"parsed mixed ISO/US datetime formats across {len(order_dt_cols)} order columns "
                                     f"({int(unparsed_orders)} unparseable/blank values, expected for "
                                     f"non-delivered orders' NULL delivery dates)")

    review_dt_cols = ["review_creation_date", "review_answer_timestamp"]
    for col in review_dt_cols:
        reviews_raw[col] = parse_mixed_datetime(reviews_raw[col])
    record("parse_review_datetimes", f"parsed mixed ISO/US datetime formats across {len(review_dt_cols)} review columns")

    items_raw["shipping_limit_date"] = parse_mixed_datetime(items_raw["shipping_limit_date"])

    # ------------------------------------------------------------------
    # 3) Currency-formatted price / freight_value strings
    # ------------------------------------------------------------------
    before_missing = items_raw["price"].isna().sum() + items_raw["freight_value"].isna().sum()
    items_raw["price"] = clean_currency(items_raw["price"])
    items_raw["freight_value"] = clean_currency(items_raw["freight_value"])
    after_missing = items_raw["price"].isna().sum() + items_raw["freight_value"].isna().sum()
    record("clean_currency_strings", f"stripped 'R$' currency symbols/commas from price and freight_value "
                                      f"({after_missing - before_missing} coercion failures)")

    # ------------------------------------------------------------------
    # 4) Impossible negative price / freight_value
    # ------------------------------------------------------------------
    before = len(items_raw)
    items = items_raw[(items_raw["price"] >= 0) & (items_raw["freight_value"] >= 0)].copy()
    record("drop_invalid_item_values", f"removed {before - len(items):,} order_item rows with negative "
                                        f"price or freight_value ({(before - len(items)) / before:.2%} of raw order_items)")

    # ------------------------------------------------------------------
    # 5) Payment type casing
    # ------------------------------------------------------------------
    payments = payments_raw.copy()
    payments["payment_type"] = payments["payment_type"].astype(object).str.strip().str.lower()
    record("normalize_payment_type", "standardized payment_type casing to lowercase "
                                      f"(values: {sorted(payments['payment_type'].unique())})")

    # ------------------------------------------------------------------
    # 6) Missing customer_state
    # ------------------------------------------------------------------
    missing_state_before = customers["customer_state"].isna().sum()
    customers["customer_state"] = customers["customer_state"].astype(object).fillna("Unknown")
    record("impute_customer_state", f"imputed {missing_state_before:,} missing customer_state values "
                                     f"({missing_state_before / len(customers):.2%} of clean customers) as 'Unknown'")

    # ------------------------------------------------------------------
    # 7) Missing product_category_name / seller_city
    # ------------------------------------------------------------------
    products = products_raw.copy()
    missing_cat_before = products["product_category_name"].isna().sum()
    products["product_category_name"] = products["product_category_name"].astype(object).fillna("unknown")
    record("impute_product_category", f"imputed {missing_cat_before:,} missing product_category_name values "
                                       f"({missing_cat_before / len(products):.2%} of products) as 'unknown'")

    sellers = sellers_raw.copy()
    missing_city_before = sellers["seller_city"].isna().sum()
    sellers["seller_city"] = sellers["seller_city"].astype(object).fillna("unknown")
    record("impute_seller_city", f"imputed {missing_city_before:,} missing seller_city values "
                                  f"({missing_city_before / len(sellers):.2%} of sellers) as 'unknown'")

    # ------------------------------------------------------------------
    # 8) Collapse order-scoped customer_id -> customer_unique_id
    # ------------------------------------------------------------------
    dim_customer = (
        customers.groupby("customer_unique_id")
        .agg(
            customer_city=("customer_city", "first"),
            customer_state=("customer_state", "first"),
            customer_zip_code_prefix=("customer_zip_code_prefix", "first"),
        )
        .reset_index()
    )
    record("build_dim_customer", f"collapsed {len(customers):,} order-scoped customer_id rows into "
                                  f"{len(dim_customer):,} unique customer_unique_id rows")

    order_customer_map = customers[["customer_id", "customer_unique_id"]].drop_duplicates(subset=["customer_id"])
    orders = orders.merge(order_customer_map, on="customer_id", how="left")
    unmapped = orders["customer_unique_id"].isna().sum()
    orders = orders.dropna(subset=["customer_unique_id"]).copy()
    record("map_orders_to_customer", f"mapped orders to customer_unique_id ({unmapped:,} orders with no "
                                      f"matching customer_id dropped)")

    # ------------------------------------------------------------------
    # 9) Product cost / margin
    # ------------------------------------------------------------------
    products = products.merge(cost_ref, on="product_id", how="left")
    missing_cost = products["product_cost"].isna().sum()
    if missing_cost:
        products["product_cost"] = products["product_cost"].fillna(products["product_cost"].median())
    record("merge_product_cost", f"merged synthetic product_cost_reference into dim_product "
                                  f"({missing_cost:,} products missing a cost fallback filled with median)")

    items = items.merge(products[["product_id", "product_cost"]], on="product_id", how="left")
    items["item_cost"] = items["product_cost"].fillna(items["price"] * 0.65)
    items["item_margin"] = (items["price"] - items["item_cost"]).round(2)
    items = items.drop(columns=["product_cost"])
    record("compute_item_margin", "computed item_margin = price - item_cost per order line")

    # ------------------------------------------------------------------
    # Derived fields: delivery delay, order value
    # ------------------------------------------------------------------
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    orders["order_purchase_date"] = orders["order_purchase_timestamp"].dt.date.astype(str)

    item_totals = items["price"] + items["freight_value"]
    order_value = item_totals.groupby(items["order_id"]).sum().rename("order_value")
    orders = orders.merge(order_value, on="order_id", how="left")
    orders["order_value"] = orders["order_value"].fillna(0.0)

    n_orders_final = len(orders)
    record("summary", f"{len(orders_raw):,} raw order rows -> {n_orders_final:,} clean order rows "
                       f"({(len(orders_raw) - n_orders_final) / len(orders_raw):.2%} removed as duplicate/unmapped)")

    # ------------------------------------------------------------------
    # dim_date
    # ------------------------------------------------------------------
    min_date = orders["order_purchase_timestamp"].min().normalize()
    max_date = orders["order_estimated_delivery_date"].max().normalize()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    dim_date = pd.DataFrame({"date": all_dates})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y-%m-%d")
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["month_number"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.month_name()
    dim_date["day_name"] = dim_date["date"].dt.day_name()
    dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["is_weekend"] = dim_date["date"].dt.dayofweek.isin([5, 6])
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date = dim_date.drop(columns=["date"])

    # ------------------------------------------------------------------
    # Final table assembly
    # ------------------------------------------------------------------
    dim_product = products[[
        "product_id", "product_category_name", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm", "product_cost",
    ]]
    dim_seller = sellers[["seller_id", "seller_city", "seller_state", "seller_zip_code_prefix"]]

    fact_orders = orders[[
        "order_id", "customer_unique_id", "order_status", "order_purchase_timestamp", "order_purchase_date",
        "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "delivery_delay_days", "order_value",
    ]].reset_index(drop=True)

    valid_order_ids = set(fact_orders["order_id"])
    fact_order_items = items[items["order_id"].isin(valid_order_ids)][[
        "order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value", "item_cost", "item_margin",
    ]].reset_index(drop=True)

    fact_payments = payments[payments["order_id"].isin(valid_order_ids)][[
        "order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value",
    ]].reset_index(drop=True)

    reviews = reviews_raw[reviews_raw["order_id"].isin(valid_order_ids)].dropna(
        subset=["review_creation_date", "review_answer_timestamp"]
    )
    fact_reviews = reviews[[
        "review_id", "order_id", "review_score", "review_creation_date", "review_answer_timestamp",
    ]].reset_index(drop=True)
    record("finalize_reviews", f"{len(reviews_raw):,} raw review rows -> {len(fact_reviews):,} clean review rows "
                                f"linked to valid orders with parseable dates")

    # ------------------------------------------------------------------
    # Persist processed CSVs (Power BI / BI-tool ready star schema)
    # ------------------------------------------------------------------
    dim_customer.to_csv(PROCESSED_DIR / "dim_customer.csv", index=False)
    dim_product.to_csv(PROCESSED_DIR / "dim_product.csv", index=False)
    dim_seller.to_csv(PROCESSED_DIR / "dim_seller.csv", index=False)
    dim_date.to_csv(PROCESSED_DIR / "dim_date.csv", index=False)
    fact_orders.to_csv(PROCESSED_DIR / "fact_orders.csv", index=False)
    fact_order_items.to_csv(PROCESSED_DIR / "fact_order_items.csv", index=False)
    fact_payments.to_csv(PROCESSED_DIR / "fact_payments.csv", index=False)
    fact_reviews.to_csv(PROCESSED_DIR / "fact_reviews.csv", index=False)

    # ------------------------------------------------------------------
    # Load into SQLite
    # ------------------------------------------------------------------
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    dim_customer.to_sql("dim_customer", conn, index=False)
    dim_product.to_sql("dim_product", conn, index=False)
    dim_seller.to_sql("dim_seller", conn, index=False)
    dim_date.to_sql("dim_date", conn, index=False)
    fact_orders.to_sql("fact_orders", conn, index=False)
    fact_order_items.to_sql("fact_order_items", conn, index=False)
    fact_payments.to_sql("fact_payments", conn, index=False)
    fact_reviews.to_sql("fact_reviews", conn, index=False)

    conn.execute("CREATE INDEX idx_orders_customer ON fact_orders(customer_unique_id)")
    conn.execute("CREATE INDEX idx_orders_date ON fact_orders(order_purchase_date)")
    conn.execute("CREATE INDEX idx_items_product ON fact_order_items(product_id)")
    conn.execute("CREATE INDEX idx_items_seller ON fact_order_items(seller_id)")
    conn.execute("CREATE INDEX idx_items_order ON fact_order_items(order_id)")
    conn.execute("CREATE INDEX idx_payments_order ON fact_payments(order_id)")
    conn.execute("CREATE INDEX idx_reviews_order ON fact_reviews(order_id)")
    conn.commit()
    conn.close()
    record("load_sqlite", f"loaded star schema into {DB_PATH.relative_to(ROOT)}")

    with open(REPORTS_DIR / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\nDone. Cleaning log written to reports/cleaning_log.json")


if __name__ == "__main__":
    main()
