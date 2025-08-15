"""
Streamlit EDA on Synthetic Data (Pro)
- Domain context presets (Retail, E‑commerce, Ride-hailing, Hospital).
- Controls for trend, seasonality, noise, and an external index.
- Advanced filters (numeric ranges, text search, top‑N) and derived features.
- Feature selection + regression (Linear/RandomForest) with metrics.
- Multi-chart subplots (Bar/Line/Pie) and CSV exports.
Deploy with: `streamlit run app.py`
"""
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Synthetic EDA + Modeling (Pro)", page_icon="🧪", layout="wide")

# -------------------------- Helpers --------------------------
def kpi_delta(curr: float, prev: float) -> float:
    if prev == 0 or np.isnan(prev):
        return 0.0
    return 100.0 * (curr - prev) / abs(prev)

@st.cache_data
def make_external_index(n: int, seed: int, strength: float) -> np.ndarray:
    rng = np.random.default_rng(seed + 111)
    base = np.cumsum(rng.normal(0, 0.3, n))  # random walk
    seasonal = np.sin(np.linspace(0, 6 * math.pi, n))  # multi-cycle seasonality
    idx = (base + seasonal) * strength
    return (idx - np.min(idx)) / (np.ptp(idx) + 1e-9)  # scale 0-1

@st.cache_data
def make_data(n_rows: int, seed: int, trend: float, season_amp: float, noise_sd: float, ext_strength: float, context: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Context presets
    contexts = {
        "Retail Coffee Chain": dict(categories=["Coffee","Tea","Pastry","Sandwich"], regions=["North","South","East","West"], value_name="revenue", units_name="orders"),
        "E‑commerce": dict(categories=["Electronics","Fashion","Home","Sports"], regions=["NA","EU","APAC","LATAM"], value_name="sales", units_name="orders"),
        "Ride‑hailing": dict(categories=["Economy","Premium","Pool","XL"], regions=["CityA","CityB","CityC","CityD"], value_name="fare_amount", units_name="trips"),
        "Hospital": dict(categories=["ER","Surgery","Pediatrics","Cardiology"], regions=["Campus1","Campus2","Campus3","Campus4"], value_name="charges", units_name="visits"),
    }
    meta = contexts.get(context, list(contexts.values())[0])

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=365, freq="D")
    sample_dates = rng.choice(dates, size=n_rows, replace=True)
    categories = np.array(meta["categories"])
    regions = np.array(meta["regions"])

    # Base latent signal with trend + seasonality
    t_idx = rng.choice(np.arange(365), size=n_rows, replace=True)
    trend_sig = trend * (t_idx / 365.0)
    season_sig = season_amp * np.sin(2 * np.pi * (t_idx / 7.0)) + 0.5 * season_amp * np.sin(2 * np.pi * (t_idx / 30.0))
    ext_idx = make_external_index(365, seed, ext_strength)
    ext_sig = ext_idx[t_idx]

    base = 100 + 50 * rng.gamma(2.0, 1.0, n_rows) + 40 * trend_sig + 30 * season_sig + 60 * ext_sig
    noise = rng.normal(0, noise_sd, n_rows)
    value = np.round(base + noise, 2)

    # Units tied loosely to value
    units = np.clip(np.round(10 + 0.05 * base + rng.normal(0, 5, n_rows)), 0, None)

    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "date": sample_dates,
        "category": rng.choice(categories, size=n_rows, replace=True),
        "region": rng.choice(regions, size=n_rows, replace=True),
        "value": np.clip(value, 0.0, None),
        "units": units.astype(int)
    })
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["context_value_name"] = meta["value_name"]
    df["context_units_name"] = meta["units_name"]
    return df

def group_dataframe(df: pd.DataFrame, group_by: list, agg: str) -> pd.DataFrame:
    numeric_cols = ["value", "units", "value_per_unit"]
    present = [c for c in numeric_cols if c in df.columns]
    if not group_by:
        return df[present].agg(agg).to_frame().T.reset_index(drop=True)
    return df.groupby(group_by, dropna=False)[present].agg(agg).reset_index()

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["value_per_unit"] = df["value"] / df["units"].replace(0, np.nan)
    df["value_per_unit"] = df["value_per_unit"].fillna(0.0)
    # Rolling features per region
    df["rolling_7_value"] = df.groupby("region")["value"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["lag_1_value"] = df.groupby("region")["value"].shift(1).fillna(df["value"].median())
    return df

def safe_eval_custom(df: pd.DataFrame, expr: str) -> pd.Series:
    # Allow only basic arithmetic operations and selected columns
    allowed_cols = {"value","units","value_per_unit","rolling_7_value","lag_1_value","month","dayofweek"}
    tokens_ok = all([tok in allowed_cols or tok.replace("_","").isalpha() == False for tok in expr.replace("("," ").replace(")"," ").replace("+"," ").replace("-"," ").replace("*"," ").replace("/"," ").split() if tok])
    if not tokens_ok:
        raise ValueError("Expression contains disallowed tokens. Use columns like value, units, value_per_unit, rolling_7_value, lag_1_value, month, dayofweek.")
    return pd.Series(eval(expr, {"__builtins__": {}}, dict(df)), index=df.index)

# -------------------------- Sidebar --------------------------
st.sidebar.header("🏷️ Context & Data Controls")
context = st.sidebar.selectbox("Domain context", ["Retail Coffee Chain","E‑commerce","Ride‑hailing","Hospital"])
n_rows = st.sidebar.slider("Rows", 500, 120_000, 10_000, step=500)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

st.sidebar.markdown("**Generation parameters**")
trend = st.sidebar.slider("Trend", 0.0, 2.0, 0.6, 0.05)
season_amp = st.sidebar.slider("Seasonality amplitude", 0.0, 2.0, 0.8, 0.05)
noise_sd = st.sidebar.slider("Noise (std dev)", 1.0, 60.0, 12.0, 1.0)
ext_strength = st.sidebar.slider("External index strength", 0.0, 2.0, 0.7, 0.05)

df = make_data(n_rows, seed, trend, season_amp, noise_sd, ext_strength, context)
df = add_engineered_features(df)

st.sidebar.subheader("Filters")
# Date
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = map(pd.to_datetime, date_range)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Text search
query = st.sidebar.text_input("Search category/region")
if query:
    q = query.strip().lower()
    df = df[df["category"].str.lower().str.contains(q) | df["region"].str.lower().str.contains(q)]

# Numeric filters
val_min, val_max = float(df["value"].min()), float(df["value"].max())
units_min, units_max = int(df["units"].min()), int(df["units"].max())
v_range = st.sidebar.slider("Filter value range", val_min, val_max, (val_min, val_max))
u_range = st.sidebar.slider("Filter units range", units_min, units_max, (units_min, units_max))
df = df[(df["value"].between(*v_range)) & (df["units"].between(*u_range))]

# Top‑N by value per category
topn = st.sidebar.slider("Top‑N per category by value", 0, 50, 0, 1)
if topn > 0:
    df = df.sort_values(["category","value"], ascending=[True, False]).groupby("category").head(topn)

# Custom feature
st.sidebar.subheader("Custom feature")
with st.sidebar.expander("Define a custom feature (optional)"):
    st.write("Use arithmetic on columns: value, units, value_per_unit, rolling_7_value, lag_1_value, month, dayofweek")
    cf_expr = st.text_input("Expression, e.g., value/ (1+units) + 0.1*rolling_7_value", value="")
    cf_name = st.text_input("Feature name", value="custom_feat")
    if cf_expr:
        try:
            df[cf_name] = safe_eval_custom(df, cf_expr)
            st.success(f"Added feature: {cf_name}")
        except Exception as e:
            st.error(str(e))

# -------------------------- Header & Context --------------------------
titles = {
    "Retail Coffee Chain": ("☕ Retail Coffee Chain – Daily Performance", "Orders and revenue simulated across regions."),
    "E‑commerce": ("🛒 E‑commerce – Sales Dashboard", "Orders and sales across global regions."),
    "Ride‑hailing": ("🚕 Ride-hailing – Trip Analytics", "Trips and fares across cities."),
    "Hospital": ("🏥 Hospital – Service Volumes & Charges", "Visits and charges across departments/campuses."),
}
title, subtitle = titles[context]
st.title(title)
st.caption(subtitle)

# -------------------------- KPIs --------------------------
# Compute prior 30-day period for growth
df_sorted = df.sort_values("date")
cutoff = df_sorted["date"].max() - pd.Timedelta(days=30)
recent = df_sorted[df_sorted["date"] > cutoff]
prev = df_sorted[(df_sorted["date"] <= cutoff) & (df_sorted["date"] > cutoff - pd.Timedelta(days=30))]

total_value = df["value"].sum()
total_units = df["units"].sum()
aov = (df["value"].sum() / max(1, df["units"].sum()))
daily_avg_value = df.groupby(df["date"].dt.date)["value"].sum().mean()

recent_val = recent["value"].sum()
prev_val = prev["value"].sum()
delta_val = kpi_delta(recent_val, prev_val)

top_cat = df.groupby("category")["value"].sum().sort_values(ascending=False).index[:1].tolist()
top_reg = df.groupby("region")["value"].sum().sort_values(ascending=False).index[:1].tolist()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total value", f"{total_value:,.0f}")
k2.metric("Total " + df["context_units_name"].iloc[0], f"{int(total_units):,}")
k3.metric("AOV (value/unit)", f"{aov:,.2f}")
k4.metric("Daily avg value", f"{daily_avg_value:,.0f}")
k5.metric("30d value growth", f"{delta_val:,.1f}%")
k6.metric("Top cat / region", f"{(top_cat[0] if top_cat else '-') } / {(top_reg[0] if top_reg else '-') }")

with st.expander("🔎 Preview data", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# -------------------------- Grouping & Aggregation --------------------------
st.subheader("🧮 Aggregation & Grouping")
time_grain = st.selectbox("Time grain for date", ["None","Day","Week","Month","Quarter"])
agg = st.selectbox("Aggregation", ["sum","mean","median","min","max"])
group_by_dims = st.multiselect("Group by", ["category","region"], default=["category"])

df_plot = df.copy()
if time_grain != "None":
    mapping = {
        "Day": df_plot["date"].dt.to_period("D").dt.start_time,
        "Week": df_plot["date"].dt.to_period("W").dt.start_time,
        "Month": df_plot["date"].dt.to_period("M").dt.start_time,
        "Quarter": df_plot["date"].dt.to_period("Q").dt.start_time,
    }
    df_plot["time"] = mapping[time_grain]
    group_cols = (["time"] + group_by_dims) if group_by_dims else ["time"]
else:
    group_cols = group_by_dims

gdf = group_dataframe(df_plot, group_cols, agg)

with st.expander("📥 Export grouped data (CSV)", expanded=False):
    buf = io.StringIO(); gdf.to_csv(buf, index=False)
    st.download_button("Download grouped.csv", data=buf.getvalue(), file_name="grouped.csv", mime="text/csv")

# -------------------------- Subplots Visualization --------------------------
st.subheader("📊 Multi-Chart Subplots")
sub_col1, sub_col2 = st.columns([2,1])

with sub_col1:
    st.markdown("**Select charts to include**")
    use_bar = st.checkbox("Bar", value=True)
    use_line = st.checkbox("Line", value=True)
    use_pie = st.checkbox("Pie", value=False)

    numeric_col = st.selectbox("Numeric metric", ["value","units","value_per_unit"], index=0)
    possible_x = []
    if "time" in gdf.columns: possible_x.append("time")
    possible_x += [c for c in ["category","region"] if c in gdf.columns]
    if not possible_x:
        possible_x = ["index"]; gdf = gdf.reset_index()
    x_col = st.selectbox("X axis", options=possible_x, index=0)

with sub_col2:
    st.info("Tip: Use generation sliders (left) to tweak trend/seasonality/noise and simulate scenarios.")

selected = [use_bar, use_line, use_pie]
n = sum(selected); rows = 1 if n <= 2 else 2; cols = n if n <= 2 else 2
fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)],
                    subplot_titles=[t for t, flag in zip(["Bar","Line","Pie"], [use_bar, use_line, use_pie]) if flag])

r = c = 1
def place_trace(trace):
    global r, c
    fig.add_trace(trace, row=r, col=c)
    c += 1
    if c > cols: c = 1; r = min(r + 1, rows)

if use_bar:
    xv = gdf.index if x_col == "index" else gdf[x_col]
    place_trace(go.Bar(x=xv, y=gdf[numeric_col], name="Bar"))
if use_line:
    gdf_sorted = gdf.sort_values(by=x_col) if x_col in gdf.columns else gdf
    xv = gdf_sorted[x_col] if x_col in gdf_sorted.columns else gdf_sorted.index
    place_trace(go.Scatter(x=xv, y=gdf_sorted[numeric_col], mode="lines+markers", name="Line"))
if use_pie:
    pie_dim_candidates = [c for c in ["category","region"] if c in gdf.columns]
    pie_dim = st.selectbox("Pie slice by", options=pie_dim_candidates or ["category"], index=0)
    pie_df = group_dataframe(df_plot, [pie_dim], agg)
    place_trace(go.Pie(labels=pie_df[pie_dim], values=pie_df[numeric_col], name="Pie", hole=0.3))

fig.update_layout(height=520 if rows == 1 else 820, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------- Feature Selection & Modeling --------------------------
st.subheader("🧠 Feature Selection & Regression Modeling")
all_features = ["units","month","dayofweek","category","region","value_per_unit","rolling_7_value","lag_1_value"]
default_feats = ["units","category","region","rolling_7_value"]
selected_features = st.multiselect("Select features (X)", options=all_features, default=default_feats)
target = st.selectbox("Target (y)", options=["value","units","value_per_unit"], index=0)

col_fs1, col_fs2, col_fs3 = st.columns([1,1,1])
with col_fs1:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with col_fs2:
    use_selectk = st.checkbox("Use SelectKBest (mutual_info_regression)", value=True)
with col_fs3:
    k_best = st.slider("k (if SelectKBest)", 1, max(1, len(selected_features)), min(4, len(selected_features) if selected_features else 1), 1)

model_type = st.radio("Model", ["LinearRegression","RandomForestRegressor"], horizontal=True)
rf_n_estimators = st.slider("RF n_estimators", 50, 600, 250, 50, disabled=(model_type!="RandomForestRegressor"))
rf_max_depth = st.slider("RF max_depth (0=None)", 0, 40, 0, 1, disabled=(model_type!="RandomForestRegressor"))

X = df[selected_features].copy()
y = df[target].astype(float)

categorical_cols = [c for c in selected_features if c in ["category","region"]]
numeric_cols = [c for c in selected_features if c not in categorical_cols]

preprocess = ColumnTransformer([
    ("num", "passthrough", numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

estimator = LinearRegression() if model_type == "LinearRegression" else RandomForestRegressor(
    n_estimators=rf_n_estimators,
    max_depth=(None if rf_max_depth == 0 else rf_max_depth),
    random_state=seed, n_jobs=-1
)

if use_selectk and selected_features:
    fs = SelectKBest(score_func=mutual_info_regression, k=min(k_best, len(selected_features)))
    model = Pipeline([("preprocess", preprocess), ("select", fs), ("model", estimator)])
else:
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

train_btn = st.button("Train model")
pred_df = None
if train_btn:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:,.3f}")
    m2.metric("R²", f"{r2:,.3f}")
    m3.metric("Test size", f"{len(y_test):,} rows")

    pred_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred).round(3)
    })
    with st.expander("🔎 Predictions (test set)", expanded=False):
        st.dataframe(pred_df.head(200), use_container_width=True)

    buf_pred = io.StringIO(); pred_df.to_csv(buf_pred, index=False)
    st.download_button("Download predictions.csv", data=buf_pred.getvalue(), file_name="predictions.csv", mime="text/csv")

# -------------------------- Exports --------------------------
st.subheader("📤 Export Results")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Filtered raw data**")
    buf_raw = io.StringIO(); df.to_csv(buf_raw, index=False)
    st.download_button("Download filtered_data.csv", data=buf_raw.getvalue(), file_name="filtered_data.csv", mime="text/csv")

with c2:
    st.markdown("**Grouped data**")
    buf_grouped = io.StringIO(); gdf.to_csv(buf_grouped, index=False)
    st.download_button("Download grouped_data.csv", data=buf_grouped.getvalue(), file_name="grouped_data.csv", mime="text/csv")

with c3:
    st.markdown("**Engineered features snapshot**")
    feats_cols = ["value","units","value_per_unit","rolling_7_value","lag_1_value","month","dayofweek","category","region","date"]
    snap = df[feats_cols].head(500).copy()
    buf_feats = io.StringIO(); snap.to_csv(buf_feats, index=False)
    st.download_button("Download features_sample.csv", data=buf_feats.getvalue(), file_name="features_sample.csv", mime="text/csv")

st.markdown("---")
st.caption("Deployment ready. App file: `app.py`. Requirements: `requirements.txt`.")
