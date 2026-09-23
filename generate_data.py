"""
Synthetic Brazilian e-commerce (Olist-shape) data generator.

Generates ~2.5 years (Jan 2016 - Aug 2018, matching the real Olist dataset's
actual date range) of orders across customers/products/sellers with realistic
operational patterns baked in on purpose:

  - Skewed (lognormal) per-customer lifetime spend so a real RFM analysis
    finds genuine revenue concentration (Pareto-like), not a hand-picked %.
  - A genuine causal relationship between delivery delay and review score
    (later delivery -> lower score, plus noise), confounded on purpose by
    product category and customer state so src/analysis.py has a real
    Simpson's-paradox check to perform before reporting the regression.
  - Product cost as 55-75% of price, varying by category, to support a
    profit-margin-by-category analysis.
  - A recency-based churn/at-risk signal driven by each customer's own
    historical inter-purchase interval.

Intentional messiness is injected into the raw CSV output to mirror a real
extract that needs cleaning downstream (see src/clean_data.py).

Source note: this is a synthetically generated dataset modeled on the public
Olist Brazilian E-Commerce schema (https://www.kaggle.com/datasets/olistbr/
brazilian-ecommerce), not the real dataset itself -- that dataset requires an
authenticated Kaggle login this environment doesn't have. See README.md.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm as _norm

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2018-08-31")

N_CUSTOMERS = 19000          # unique customer_unique_id (every one places >=1 order)
N_PRODUCTS = 2000
N_SELLERS = 500

STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE",
          "CE", "PA", "ES", "MT", "MS", "MA", "RN", "PB", "AL", "SE"]
STATE_WEIGHTS = np.array([42, 13, 11, 5.5, 5, 3.6, 3.4, 2.2, 2.1, 1.8,
                           1.6, 1.5, 1.4, 1.1, 0.9, 0.9, 0.8, 0.6, 0.5, 0.4])
STATE_WEIGHTS = STATE_WEIGHTS / STATE_WEIGHTS.sum()

CITY_BY_STATE = {
    "SP": ["sao paulo", "campinas", "santos", "sorocaba"],
    "RJ": ["rio de janeiro", "niteroi", "duque de caxias"],
    "MG": ["belo horizonte", "uberlandia", "contagem"],
    "RS": ["porto alegre", "caxias do sul"],
    "PR": ["curitiba", "londrina"],
    "SC": ["florianopolis", "joinville"],
    "BA": ["salvador", "feira de santana"],
    "DF": ["brasilia"],
    "GO": ["goiania"],
    "PE": ["recife"],
    "CE": ["fortaleza"],
    "PA": ["belem"],
    "ES": ["vitoria"],
    "MT": ["cuiaba"],
    "MS": ["campo grande"],
    "MA": ["sao luis"],
    "RN": ["natal"],
    "PB": ["joao pessoa"],
    "AL": ["maceio"],
    "SE": ["aracaju"],
}

# States with systematically slower logistics (rural/distant from SP/RJ
# fulfillment hubs) -- this is the confound baked into the review-score
# regression: these states have both longer delivery delays AND, independent
# of delay, slightly lower baseline satisfaction (rural service friction),
# which src/analysis.py must control for.
SLOW_LOGISTICS_STATES = {"MA", "PA", "AL", "SE", "PB", "RN", "MT", "MS", "CE", "PE", "BA"}

# 40 product categories with price tier and cost-ratio band (varies by
# category so profit-margin-by-category has real signal). category also
# functions as part of the regression confound: some categories (heavy/
# fragile goods) are intrinsically slower to ship AND independently reviewed
# a bit more harshly regardless of delay.
CATEGORIES = [
    # name, price_lognorm_mean, price_lognorm_sigma, cost_ratio_lo, cost_ratio_hi, ship_delay_bias_days, review_bias
    ("bed_bath_table",       3.6, 0.55, 0.60, 0.72, 0.0, 0.00),
    ("health_beauty",        3.4, 0.60, 0.55, 0.68, 0.0, 0.05),
    ("sports_leisure",       3.7, 0.55, 0.58, 0.70, 0.0, 0.00),
    ("furniture_decor",      4.0, 0.60, 0.62, 0.75, 1.5, -0.05),
    ("computers_accessories",4.3, 0.65, 0.60, 0.72, 0.0, 0.05),
    ("housewares",           3.3, 0.50, 0.58, 0.70, 0.0, 0.00),
    ("watches_gifts",        4.2, 0.60, 0.55, 0.68, 0.0, 0.05),
    ("telephony",            3.9, 0.65, 0.60, 0.73, 0.0, 0.00),
    ("garden_tools",         3.5, 0.55, 0.60, 0.72, 0.5, 0.00),
    ("auto",                 4.1, 0.60, 0.62, 0.75, 0.5, -0.03),
    ("cool_stuff",           4.0, 0.55, 0.58, 0.70, 0.0, 0.05),
    ("baby",                 3.6, 0.55, 0.58, 0.70, 0.0, 0.05),
    ("toys",                 3.5, 0.55, 0.58, 0.70, 0.0, 0.05),
    ("perfumery",            4.0, 0.55, 0.55, 0.66, 0.0, 0.08),
    ("fashion_bags_accessories", 3.7, 0.55, 0.58, 0.70, 0.0, 0.03),
    ("stationery",           2.9, 0.50, 0.60, 0.72, 0.0, 0.00),
    ("electronics",          4.6, 0.65, 0.62, 0.75, 0.0, -0.02),
    ("consoles_games",       4.7, 0.55, 0.65, 0.75, 0.0, 0.00),
    ("construction_tools",   4.0, 0.60, 0.62, 0.74, 1.0, -0.05),
    ("home_appliances",      4.5, 0.60, 0.62, 0.75, 1.0, -0.03),
    ("furniture_living_room",4.2, 0.60, 0.62, 0.75, 1.8, -0.07),
    ("office_furniture",     4.3, 0.60, 0.62, 0.75, 1.5, -0.05),
    ("small_appliances",     3.8, 0.55, 0.60, 0.72, 0.5, 0.00),
    ("fashion_shoes",        3.9, 0.55, 0.58, 0.70, 0.0, 0.02),
    ("fashion_underwear_beach", 3.2, 0.50, 0.55, 0.68, 0.0, 0.02),
    ("pet_shop",             3.6, 0.55, 0.58, 0.70, 0.0, 0.02),
    ("market_place",         3.0, 0.55, 0.58, 0.72, 0.5, -0.02),
    ("books_general",        2.8, 0.45, 0.55, 0.68, 0.0, 0.06),
    ("books_technical",      3.4, 0.45, 0.55, 0.68, 0.0, 0.06),
    ("cds_dvds_musicals",    2.6, 0.45, 0.55, 0.68, 0.0, 0.03),
    ("music",                2.9, 0.45, 0.58, 0.70, 0.0, 0.03),
    ("costruction_tools_garden", 3.9, 0.55, 0.60, 0.73, 1.0, -0.04),
    ("air_conditioning",     4.6, 0.55, 0.62, 0.75, 1.5, -0.05),
    ("luggage_accessories",  3.8, 0.55, 0.58, 0.70, 0.0, 0.00),
    ("food_drink",           2.9, 0.55, 0.60, 0.72, 0.0, 0.00),
    ("industry_commerce_and_business", 4.2, 0.60, 0.62, 0.75, 0.5, -0.02),
    ("signaling_and_security", 3.9, 0.55, 0.60, 0.73, 0.5, -0.02),
    ("christmas_supplies",   3.1, 0.55, 0.58, 0.70, 0.0, 0.02),
    ("party_supplies",       2.9, 0.50, 0.58, 0.70, 0.0, 0.02),
    ("diapers_and_hygiene",  2.7, 0.45, 0.58, 0.70, 0.0, 0.03),
    ("art",                  3.3, 0.55, 0.55, 0.68, 0.0, 0.03),
]
cat_df = pd.DataFrame(CATEGORIES, columns=[
    "product_category_name", "price_mu", "price_sigma", "cost_lo", "cost_hi", "ship_delay_bias", "review_bias"
])

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
PAYMENT_WEIGHTS = [0.74, 0.19, 0.05, 0.02]

# ---------------------------------------------------------------------------
# 1) Dimension tables: sellers, products, customers
# ---------------------------------------------------------------------------
seller_states = rng.choice(STATES, size=N_SELLERS, p=STATE_WEIGHTS)
sellers = pd.DataFrame({
    "seller_id": [f"seller_{i:05d}" for i in range(N_SELLERS)],
    "seller_zip_code_prefix": rng.integers(1000, 99999, N_SELLERS),
    "seller_state": seller_states,
})
sellers["seller_city"] = [rng.choice(CITY_BY_STATE[s]) for s in sellers["seller_state"]]
sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]].to_csv(
    RAW_DIR / "sellers.csv", index=False
)

product_cat_idx = rng.integers(0, len(cat_df), N_PRODUCTS)
products = pd.DataFrame({
    "product_id": [f"prod_{i:05d}" for i in range(N_PRODUCTS)],
    "_cat_idx": product_cat_idx,
})
products = products.merge(cat_df, left_on="_cat_idx", right_index=True)
products["product_category_name"] = products["product_category_name"]
products["product_weight_g"] = np.clip(rng.gamma(2.0, 700, N_PRODUCTS), 50, 30000).round(0).astype(int)
products["product_length_cm"] = np.clip(rng.gamma(4, 8, N_PRODUCTS), 5, 120).round(1)
products["product_height_cm"] = np.clip(rng.gamma(3, 5, N_PRODUCTS), 2, 100).round(1)
products["product_width_cm"] = np.clip(rng.gamma(4, 7, N_PRODUCTS), 5, 100).round(1)
# Base list price per product (lognormal, category-dependent); order_items
# will apply small item-level jitter around this.
products["_base_price"] = np.round(
    np.exp(rng.normal(products["price_mu"], products["price_sigma"])), 2
).clip(9.9, 6000.0)
products["_cost_ratio"] = rng.uniform(products["cost_lo"], products["cost_hi"])
products["product_cost"] = (products["_base_price"] * products["_cost_ratio"]).round(2)

products_out = products[[
    "product_id", "product_category_name", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
]]
products_out.to_csv(RAW_DIR / "products.csv", index=False)
# Keep an internal (not-raw) product economics table for order-item generation
products_econ = products[["product_id", "product_category_name", "_base_price", "product_cost", "ship_delay_bias", "review_bias"]]

# Real Olist has no cost data; a synthetic cost reference (55-75% of price,
# category-dependent -- see CATEGORIES cost_lo/cost_hi bands above) is written
# as its own raw file so the margin-by-category analysis is traceable to a
# genuine (if assumed) generation rule rather than invented at analysis time.
products[["product_id", "product_cost"]].to_csv(RAW_DIR / "product_cost_reference.csv", index=False)

customer_states = rng.choice(STATES, size=N_CUSTOMERS, p=STATE_WEIGHTS)
customers = pd.DataFrame({
    "customer_unique_id": [f"cust_{i:06d}" for i in range(N_CUSTOMERS)],
    "customer_state": customer_states,
})
customers["customer_city"] = [rng.choice(CITY_BY_STATE[s]) for s in customers["customer_state"]]
customers["customer_zip_code_prefix"] = rng.integers(1000, 99999, N_CUSTOMERS)

# Skewed per-customer "propensity to spend" (drives repeat-order likelihood
# AND order value) so revenue concentration emerges naturally from RFM
# analysis rather than being hardcoded.
customers["_spend_propensity"] = rng.lognormal(mean=0.0, sigma=1.15, size=N_CUSTOMERS)

print(f"Generated dimensions: {len(customers):,} customers, {len(products_out):,} products, {len(sellers):,} sellers.")

# ---------------------------------------------------------------------------
# 2) Orders: every customer places >=1 order (mirrors real Olist, where the
#    large majority of customers are one-time buyers); high-spend-propensity
#    customers are additionally more likely to place repeat orders (Poisson
#    extra-order count scaled by propensity) AND pay more per order (a price
#    multiplier applied when generating order_items below). Both channels
#    combine into a genuine, discoverable Pareto-like revenue concentration
#    without hardcoding the resulting percentages.
# ---------------------------------------------------------------------------
propensity = customers["_spend_propensity"].to_numpy()
extra_order_lambda = 0.58 * (propensity / propensity.mean())
extra_orders = rng.poisson(extra_order_lambda)
n_orders_per_customer = 1 + extra_orders  # every customer gets exactly 1 base order

customer_idx_for_orders = np.repeat(np.arange(N_CUSTOMERS), n_orders_per_customer)
N_ORDERS_TARGET = len(customer_idx_for_orders)
rng.shuffle(customer_idx_for_orders)

# Per-order price multiplier derived from the ordering customer's spend
# propensity (dampened with a power <1 so it nudges order value up without
# producing absurd outlier prices) -- this is what makes Monetary (RFM) and
# category revenue concentration a genuine, non-hardcoded finding.
order_price_multiplier = np.clip(
    (propensity[customer_idx_for_orders] ** 0.45) * rng.normal(1.0, 0.12, N_ORDERS_TARGET), 0.4, 4.5
)

print(f"Order count target from customer repeat-purchase model: {N_ORDERS_TARGET:,} orders "
      f"across {N_CUSTOMERS:,} customers ({(n_orders_per_customer > 1).sum():,} repeat buyers, "
      f"{(n_orders_per_customer > 1).mean():.1%} of customers).")

total_days = (END_DATE - START_DATE).days
# Mild month-over-month growth trend (realistic platform growth over 2.5 yrs)
# plus a Black Friday / holiday seasonality bump in November and December.
day_offsets = np.arange(total_days + 1)
month_of_day = (START_DATE + pd.to_timedelta(day_offsets, unit="D")).month
growth_weight = 1.0 + 0.55 * (day_offsets / total_days)
season_weight = np.where(np.isin(month_of_day, [11]), 1.9,
                  np.where(np.isin(month_of_day, [12]), 1.35,
                  np.where(np.isin(month_of_day, [1, 2]), 0.85, 1.0)))
day_weight = growth_weight * season_weight
day_weight = day_weight / day_weight.sum()
order_day_offsets = rng.choice(day_offsets, size=N_ORDERS_TARGET, p=day_weight)

HOUR_WEIGHTS = np.array([
    0.005,0.003,0.002,0.002,0.002,0.004,0.010,0.020,
    0.035,0.048,0.058,0.062,0.060,0.058,0.058,0.060,
    0.062,0.065,0.070,0.075,0.072,0.060,0.040,0.020,
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

order_ids = [f"order_{i:06d}" for i in range(N_ORDERS_TARGET)]
customer_ids_for_orders = [f"custid_{customer_idx_for_orders[i]:06d}_{i:06d}" for i in range(N_ORDERS_TARGET)]
# customer_id (per-order surrogate) vs customer_unique_id (real person) --
# mirrors real Olist schema where customer_id is order-scoped.

purchase_hours = rng.choice(24, size=N_ORDERS_TARGET, p=HOUR_WEIGHTS)
purchase_minutes = rng.integers(0, 60, N_ORDERS_TARGET)
purchase_ts = (START_DATE
               + pd.to_timedelta(order_day_offsets, unit="D")
               + pd.to_timedelta(purchase_hours, unit="h")
               + pd.to_timedelta(purchase_minutes, unit="m"))

orders = pd.DataFrame({
    "order_id": order_ids,
    "_customer_idx": customer_idx_for_orders,
    "customer_id": customer_ids_for_orders,
    "order_purchase_timestamp": purchase_ts,
    "_price_multiplier": order_price_multiplier,
})
orders = orders.merge(
    customers[["customer_state"]].reset_index().rename(columns={"index": "_customer_idx"}),
    on="_customer_idx", how="left",
)

# Approval: usually within hours
orders["order_approved_at"] = orders["order_purchase_timestamp"] + pd.to_timedelta(
    np.clip(rng.gamma(2.0, 3.0, N_ORDERS_TARGET), 0.1, 72), unit="h"
)

# Carrier handoff: 1-3 days after approval
orders["order_delivered_carrier_date"] = orders["order_approved_at"] + pd.to_timedelta(
    np.clip(rng.gamma(2.2, 0.9, N_ORDERS_TARGET), 0.2, 10), unit="D"
)

# Estimated delivery date: promised at purchase time, ~ carrier + logistics
# baseline days depending on customer state (distance from SP/RJ hub).
is_slow_state = orders["customer_state"].isin(SLOW_LOGISTICS_STATES).to_numpy()
base_transit_days = np.where(is_slow_state, rng.normal(18, 3, N_ORDERS_TARGET), rng.normal(11, 2.5, N_ORDERS_TARGET))
base_transit_days = np.clip(base_transit_days, 4, 45)
orders["order_estimated_delivery_date"] = orders["order_purchase_timestamp"] + pd.to_timedelta(
    np.round(base_transit_days), unit="D"
)

# Actual transit time: real carrier performance. Slow-logistics states carry
# both a longer ACTUAL transit AND a slightly longer estimate (so a naive
# comparison can look deceptively "on time" for those states) -- the real
# operational driver of delay is carrier variability, injected here
# independent of the promised estimate, plus a category-level shipping bias.
carrier_noise = rng.gamma(2.0, 2.2, N_ORDERS_TARGET) - 4.4  # mean ~0, right-skewed tail (late-shipment risk)
state_carrier_penalty = np.where(is_slow_state, rng.gamma(2.0, 2.0, N_ORDERS_TARGET), 0.0)
actual_transit_days = base_transit_days + carrier_noise + state_carrier_penalty
actual_transit_days = np.clip(actual_transit_days, 1, 60)

orders["order_delivered_customer_date"] = orders["order_delivered_carrier_date"] + pd.to_timedelta(
    np.clip(actual_transit_days - (orders["order_delivered_carrier_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400, 0.3, 55),
    unit="D"
)

# Order status: mostly delivered; small share canceled/unavailable/shipped-in-flight
# (in-flight only possible for orders placed very close to END_DATE)
days_to_end = (END_DATE - orders["order_purchase_timestamp"]).dt.days
status = np.full(N_ORDERS_TARGET, "delivered", dtype=object)
cancel_mask = rng.random(N_ORDERS_TARGET) < 0.015
status[cancel_mask] = "canceled"
unavail_mask = (~cancel_mask) & (rng.random(N_ORDERS_TARGET) < 0.006)
status[unavail_mask] = "unavailable"
in_flight_eligible = (days_to_end < 20) & (~cancel_mask) & (~unavail_mask)
in_flight_mask = in_flight_eligible & (rng.random(N_ORDERS_TARGET) < 0.35)
status[in_flight_mask] = np.where(rng.random(in_flight_mask.sum()) < 0.5, "shipped", "processing")
orders["order_status"] = status

# For non-delivered orders, null out delivery-dependent dates appropriately
not_delivered = orders["order_status"] != "delivered"
orders.loc[orders["order_status"].isin(["canceled", "unavailable"]), "order_delivered_carrier_date"] = pd.NaT
orders.loc[orders["order_status"].isin(["canceled", "unavailable"]), "order_delivered_customer_date"] = pd.NaT
orders.loc[orders["order_status"] == "processing", "order_delivered_carrier_date"] = pd.NaT
orders.loc[orders["order_status"] == "processing", "order_delivered_customer_date"] = pd.NaT
orders.loc[orders["order_status"] == "shipped", "order_delivered_customer_date"] = pd.NaT

print(f"Generated {len(orders):,} orders, status breakdown:\n{orders['order_status'].value_counts()}")

# ---------------------------------------------------------------------------
# 3) Order items (1-3 items per order, weighted toward 1), payments, reviews
# ---------------------------------------------------------------------------
n_items_per_order = rng.choice([1, 2, 3, 4], size=N_ORDERS_TARGET, p=[0.72, 0.18, 0.07, 0.03])
total_items = int(n_items_per_order.sum())

products_econ_arr = products_econ.reset_index(drop=True)
seller_ids_arr = sellers["seller_id"].to_numpy()

# Vectorized: repeat each order's id/purchase-timestamp by its item count,
# then draw all product/seller picks and price/freight noise in one shot
# (a per-order Python loop calling pandas .sample() ~30K times is far too
# slow at this scale -- this does the same random assignment vectorized).
item_order_ids = np.repeat(orders["order_id"].to_numpy(), n_items_per_order)
item_order_item_id = np.concatenate([np.arange(1, n + 1) for n in n_items_per_order])
item_purchase_ts = np.repeat(orders["order_purchase_timestamp"].to_numpy(), n_items_per_order)
item_price_multiplier = np.repeat(orders["_price_multiplier"].to_numpy(), n_items_per_order)

prod_pick_idx = rng.integers(0, N_PRODUCTS, total_items)
picked = products_econ_arr.iloc[prod_pick_idx].reset_index(drop=True)

item_seller_id = rng.choice(seller_ids_arr, size=total_items)
price = np.clip(
    picked["_base_price"].to_numpy() * item_price_multiplier * rng.normal(1.0, 0.06, total_items), 4.9, None
)
item_cost = np.round(picked["product_cost"].to_numpy() * rng.normal(1.0, 0.03, total_items), 2)
freight = np.clip(price * rng.uniform(0.04, 0.18, total_items) + rng.normal(8, 3, total_items), 6.5, None)
ship_limit_days = rng.integers(2, 12, total_items)
item_shipping_limit = pd.to_datetime(item_purchase_ts) + pd.to_timedelta(ship_limit_days, unit="D")

order_items = pd.DataFrame({
    "order_id": item_order_ids,
    "order_item_id": item_order_item_id,
    "product_id": picked["product_id"].to_numpy(),
    "seller_id": item_seller_id,
    "shipping_limit_date": item_shipping_limit,
    "price": np.round(price, 2),
    "freight_value": np.round(freight, 2),
    "_cost": item_cost,
    "_category": picked["product_category_name"].to_numpy(),
    "_ship_bias": picked["ship_delay_bias"].to_numpy(),
    "_review_bias": picked["review_bias"].to_numpy(),
})
print(f"Generated {len(order_items):,} order items.")

# Per-order aggregate category delay/review bias (average across items) --
# used below to inject the category confound into delivery delay & review.
order_bias = order_items.groupby("order_id").agg(
    _ship_bias=("_ship_bias", "mean"), _review_bias=("_review_bias", "mean"),
    _dominant_category=("_category", lambda s: s.mode().iloc[0]),
).reset_index()
orders = orders.merge(order_bias, on="order_id", how="left")
orders["_ship_bias"] = orders["_ship_bias"].fillna(0.0)
orders["_review_bias"] = orders["_review_bias"].fillna(0.0)

# Re-apply category shipping bias to delivered orders' actual delivery date
# (heavier/bulkier categories ship a bit slower, independent of state).
delivered_mask = orders["order_status"] == "delivered"
orders.loc[delivered_mask, "order_delivered_customer_date"] = (
    orders.loc[delivered_mask, "order_delivered_customer_date"]
    + pd.to_timedelta(orders.loc[delivered_mask, "_ship_bias"], unit="D")
)

# Payments: 1 payment row per order typically, occasionally split (voucher + card)
# Vectorized order-value lookup via groupby+sum (a per-order filter over the
# full order_items table would be O(n_orders x n_items) and effectively hangs
# at this scale).
item_totals = order_items["price"] + order_items["freight_value"]
order_value_map = item_totals.groupby(order_items["order_id"]).sum()
order_values = order_value_map.reindex(orders["order_id"]).to_numpy()

n = N_ORDERS_TARGET
ptypes = rng.choice(PAYMENT_TYPES, size=n, p=PAYMENT_WEIGHTS)
split_mask = rng.random(n) < 0.06
installments_cc = rng.choice([1, 2, 3, 6, 10, 12], size=n, p=[0.35, 0.15, 0.15, 0.15, 0.12, 0.08])
installments_split_cc = rng.choice([1, 2, 3, 6, 10], size=n)
voucher_frac = rng.uniform(0.1, 0.3, n)

payment_rows = []
for i in range(n):
    oid = orders["order_id"].iat[i]
    order_value = order_values[i]
    ptype = ptypes[i]
    if split_mask[i]:
        voucher_amt = round(order_value * voucher_frac[i], 2)
        remainder = round(order_value - voucher_amt, 2)
        installments = int(installments_split_cc[i]) if ptype == "credit_card" else 1
        payment_rows.append((oid, 1, "voucher", 1, voucher_amt))
        payment_rows.append((oid, 2, ptype, installments, remainder))
    else:
        installments = int(installments_cc[i]) if ptype == "credit_card" else 1
        payment_rows.append((oid, 1, ptype, installments, round(order_value, 2)))

order_payments = pd.DataFrame(payment_rows, columns=[
    "order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"
])
print(f"Generated {len(order_payments):,} payment rows.")

# Reviews: only for orders that reached at least 'shipped'; review_score
# causally depends on delivery_delay_days = delivered - estimated, plus the
# category/state confound biases and noise.
REVIEW_TITLES = ["", "Great product", "Not what I expected", "Fast delivery", "Poor quality",
                  "Would buy again", "Late delivery", "As described", "Damaged item", "Excellent"]
REVIEW_MESSAGES = [
    "", "Recommend it.", "Arrived late, not happy.", "Exactly as pictured, works well.",
    "Poor packaging, item was scratched.", "Delivery was much faster than expected!",
    "Product broke after a week.", "Good value for the price.", "Wrong item color sent.",
    "Very satisfied with this purchase.",
]

review_eligible = orders[orders["order_status"].isin(["delivered", "shipped"])].copy()
delay_days = (review_eligible["order_delivered_customer_date"] - review_eligible["order_estimated_delivery_date"]).dt.total_seconds() / 86400
delay_fill = pd.Series(rng.normal(2, 5, len(review_eligible)), index=delay_days.index)
delay_days = delay_days.fillna(delay_fill)  # shipped-not-yet-delivered: proxy noise, low weight

# Headline causal relationship: later delivery -> lower score, modeled as an
# ordinal/latent-variable process (not a simple linear-then-round score) so
# the resulting 1-5 distribution is realistically skewed (many 5s, a genuine
# tail of 1s) rather than clustered in the middle. A latent satisfaction
# score L is computed per review and bucketed into 1-5 using thresholds
# calibrated (via the standard normal quantile function) to produce a
# realistic score mix at zero delay; L is pushed down by delivery delay
# (the headline effect) and by the category/state confound (which correlates
# with delay but has its own independent nudge on review score -- the
# confound analysis.py must catch and control for).
CUM_PROPS_AT_ZERO_DELAY = [0.08, 0.13, 0.24, 0.46]  # P(score<=1..4) targets at delay=0, no bias
THRESHOLDS = _norm.ppf(CUM_PROPS_AT_ZERO_DELAY)

state_bias = review_eligible["customer_state"].isin(SLOW_LOGISTICS_STATES).astype(float) * -0.35
latent = (
    -0.16 * delay_days.clip(lower=-12, upper=30).to_numpy()
    + 2.2 * review_eligible["_review_bias"].to_numpy()
    + state_bias.to_numpy()
    + rng.normal(0, 1.0, len(review_eligible))
)
review_score = (np.digitize(latent, THRESHOLDS) + 1).astype(int)

review_creation = review_eligible["order_delivered_customer_date"].fillna(
    review_eligible["order_estimated_delivery_date"]
) + pd.to_timedelta(np.clip(rng.gamma(2, 1.5, len(review_eligible)), 0.2, 20), unit="D")
review_answer = review_creation + pd.to_timedelta(np.clip(rng.gamma(2, 0.8, len(review_eligible)), 0.1, 10), unit="D")

title_idx = np.where(review_score <= 2, rng.integers(4, 9, len(review_eligible)),
             np.where(review_score >= 4, rng.integers(0, 4, len(review_eligible)),
                       rng.integers(0, 10, len(review_eligible))))
msg_idx = title_idx  # paired titles/messages by same tone bucket for coherence

order_reviews = pd.DataFrame({
    "review_id": [f"rev_{i:06d}" for i in range(len(review_eligible))],
    "order_id": review_eligible["order_id"].to_numpy(),
    "review_score": review_score,
    "review_comment_title": [REVIEW_TITLES[i] for i in title_idx],
    "review_comment_message": [REVIEW_MESSAGES[i] for i in msg_idx],
    "review_creation_date": review_creation.to_numpy(),
    "review_answer_timestamp": review_answer.to_numpy(),
})
print(f"Generated {len(order_reviews):,} reviews. Mean score: {order_reviews['review_score'].mean():.2f}")

# ---------------------------------------------------------------------------
# 4) customers.csv (order-scoped customer_id -> customer_unique_id mapping,
#    matching the real Olist schema)
# ---------------------------------------------------------------------------
customers_full = customers.reset_index().rename(columns={"index": "_customer_idx"})
order_customer_map = orders[["customer_id", "_customer_idx"]].merge(customers_full, on="_customer_idx", how="left")
customers_out = order_customer_map[[
    "customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"
]].drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
customers_out.to_csv(RAW_DIR / "customers.csv", index=False)

orders_out = orders[[
    "order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date",
]].copy()

# ---------------------------------------------------------------------------
# 5) Inject realistic messiness into the RAW extract (cleaned downstream)
# ---------------------------------------------------------------------------
messy_orders = orders_out.copy()
messy_items = order_items[["order_id", "order_item_id", "product_id", "seller_id",
                            "shipping_limit_date", "price", "freight_value"]].copy()
messy_payments = order_payments.copy()
messy_reviews = order_reviews.copy()
messy_customers = customers_out.copy()

# --- customers: ~3% missing customer_state ---
mask = rng.random(len(messy_customers)) < 0.03
messy_customers.loc[mask, "customer_state"] = np.nan

# --- customers: ~0.4% duplicate rows (re-extract artifact) ---
dup_idx = rng.choice(messy_customers.index, size=max(1, int(len(messy_customers) * 0.004)), replace=False)
messy_customers = pd.concat([messy_customers, messy_customers.loc[dup_idx]], ignore_index=True)

# --- orders: ~0.5% duplicate order_id rows ---
dup_idx = rng.choice(messy_orders.index, size=int(len(messy_orders) * 0.005), replace=False)
messy_orders = pd.concat([messy_orders, messy_orders.loc[dup_idx]], ignore_index=True)

# --- orders: mixed ISO/US datetime string formats across timestamp columns ---
def mixed_format(ts):
    if pd.isna(ts):
        return ""
    if rng.random() < 0.3:
        return ts.strftime("%m/%d/%Y %H:%M:%S")
    return ts.isoformat(sep=" ")

for col in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date"]:
    messy_orders[col] = messy_orders[col].apply(mixed_format)

# --- order_items: ~2.5% price / ~2.5% freight_value stored as "R$123.45" strings ---
messy_items["price"] = messy_items["price"].astype(object)
messy_items["freight_value"] = messy_items["freight_value"].astype(object)
mask = rng.random(len(messy_items)) < 0.025
messy_items.loc[mask, "price"] = messy_items.loc[mask, "price"].apply(lambda x: f"R$ {x:,.2f}")
mask2 = rng.random(len(messy_items)) < 0.025
messy_items.loc[mask2, "freight_value"] = messy_items.loc[mask2, "freight_value"].apply(lambda x: f"R$ {x:,.2f}")

# --- order_items: a handful of impossible negative price/freight values ---
mask = rng.random(len(messy_items)) < 0.004
neg_targets = messy_items.loc[mask, "price"].apply(lambda x: isinstance(x, (int, float)))
neg_idx = messy_items.loc[mask].loc[neg_targets].index
messy_items.loc[neg_idx, "price"] = -messy_items.loc[neg_idx, "price"].astype(float)
mask3 = rng.random(len(messy_items)) < 0.003
neg_targets3 = messy_items.loc[mask3, "freight_value"].apply(lambda x: isinstance(x, (int, float)))
neg_idx3 = messy_items.loc[mask3].loc[neg_targets3].index
messy_items.loc[neg_idx3, "freight_value"] = -messy_items.loc[neg_idx3, "freight_value"].astype(float)

# --- order_items: shipping_limit_date mixed format too ---
messy_items["shipping_limit_date"] = messy_items["shipping_limit_date"].apply(mixed_format)

# --- payments: inconsistent payment_type casing on ~4% of rows ---
messy_payments["payment_type"] = messy_payments["payment_type"].astype(object)
mask = rng.random(len(messy_payments)) < 0.04
messy_payments.loc[mask, "payment_type"] = messy_payments.loc[mask, "payment_type"].apply(
    lambda s: s.title() if rng.random() < 0.5 else s.upper()
)

# --- reviews: mixed datetime formats + ~2% missing comment fields already blank; add some NaN score edge ---
for col in ["review_creation_date", "review_answer_timestamp"]:
    messy_reviews[col] = messy_reviews[col].apply(mixed_format)
messy_reviews["review_comment_title"] = messy_reviews["review_comment_title"].replace("", np.nan)
messy_reviews["review_comment_message"] = messy_reviews["review_comment_message"].replace("", np.nan)

# --- products: ~2% missing product_category_name ---
messy_products = products_out.copy()
messy_products["product_category_name"] = messy_products["product_category_name"].astype(object)
mask = rng.random(len(messy_products)) < 0.02
messy_products.loc[mask, "product_category_name"] = np.nan

# --- sellers: ~1.5% missing seller_city ---
messy_sellers = sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]].copy()
messy_sellers["seller_city"] = messy_sellers["seller_city"].astype(object)
mask = rng.random(len(messy_sellers)) < 0.015
messy_sellers.loc[mask, "seller_city"] = np.nan

# Shuffle row order (re-extract artifact) and write
messy_orders = messy_orders.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
messy_items = messy_items.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
messy_customers = messy_customers.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

messy_orders.to_csv(RAW_DIR / "orders.csv", index=False)
messy_items.to_csv(RAW_DIR / "order_items.csv", index=False)
messy_payments.to_csv(RAW_DIR / "order_payments.csv", index=False)
messy_reviews.to_csv(RAW_DIR / "order_reviews.csv", index=False)
messy_customers.to_csv(RAW_DIR / "customers.csv", index=False)
messy_products.to_csv(RAW_DIR / "products.csv", index=False)
messy_sellers.to_csv(RAW_DIR / "sellers.csv", index=False)

print(f"\nRaw extract written to {RAW_DIR}:")
print(f"  customers.csv     {len(messy_customers):,} rows")
print(f"  orders.csv        {len(messy_orders):,} rows")
print(f"  order_items.csv   {len(messy_items):,} rows")
print(f"  order_payments.csv{len(messy_payments):,} rows")
print(f"  order_reviews.csv {len(messy_reviews):,} rows")
print(f"  products.csv      {len(messy_products):,} rows")
print(f"  sellers.csv       {len(messy_sellers):,} rows")
