"""
Streamlit EDA on Synthetic Data
- Generates synthetic data inside this file (no external datasets).
- Lets you explore with filters and visualize bar, line, and pie charts.
- Ready for Streamlit Cloud / GitHub deployment: `streamlit run app.py`
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Synthetic EDA", page_icon="📊", layout="wide")

@st.cache_data
def make_data(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Synthetic categorical variables
    categories = np.array(["A", "B", "C", "D"])
    regions = np.array(["North", "South", "East", "West"])

    # Dates over the past 12 months
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=365, freq="D")
    sample_dates = rng.choice(dates, size=n_rows, replace=True)

    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "date": sample_dates,
        "category": rng.choice(categories, size=n_rows, replace=True, p=[0.25, 0.25, 0.25, 0.25]),
        "region": rng.choice(regions, size=n_rows, replace=True),
        # Positive skewed values (e.g., sales) + noise
        "value": np.round(rng.gamma(shape=2.0, scale=50.0, size=n_rows) + rng.normal(0, 10, n_rows), 2),
        # A second numeric feature correlated with value
        "units": np.clip(np.round(rng.normal(20, 7, n_rows) + 0.05 * rng.standard_normal(n_rows) *  df_val_corr_helper(n_rows, seed), 0), 0, None)
    })
    # Ensure non-negative value
    df["value"] = df["value"].clip(lower=0.0)
    return df

def df_val_corr_helper(n_rows: int, seed: int) -> np.ndarray:
    # helper to add slight correlation signal
    rng = np.random.default_rng(seed + 7)
    return rng.normal(0, 100, n_rows)

def group_dataframe(df: pd.DataFrame, group_by: list, agg: str) -> pd.DataFrame:
    numeric_cols = ["value", "units"]
    if not group_by:
        return df[numeric_cols].agg(agg).to_frame().T.reset_index(drop=True)
    grouped = df.groupby(group_by, dropna=False)[numeric_cols].agg(agg).reset_index()
    return grouped

# Sidebar controls
st.sidebar.header("⚙️ Controls")
n_rows = st.sidebar.slider("Rows", 200, 50_000, 2_500, step=100)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
df = make_data(n_rows, seed)

st.sidebar.subheader("Filters")
# Date range filter
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = map(pd.to_datetime, date_range)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

cat_filter = st.sidebar.multiselect("Category", sorted(df["category"].unique().tolist()))
if cat_filter:
    df = df[df["category"].isin(cat_filter)]

region_filter = st.sidebar.multiselect("Region", sorted(df["region"].unique().tolist()))
if region_filter:
    df = df[df["region"].isin(region_filter)]

st.title("📊 EDA on Synthetic Data")
st.caption("Generate 🧪 synthetic data, filter it, and visualize with bar, line, or pie charts.")

# Basic info
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Categories", df["category"].nunique())
c3.metric("Regions", df["region"].nunique())
c4.metric("Date span (days)", (df["date"].max() - df["date"].min()).days + 1)

with st.expander("🔎 Preview data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

with st.expander("📈 Summary stats", expanded=False):
    st.write(df[["value","units"]].describe())

# Time grain & grouping
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

st.subheader("📊 Charts")
chart_type = st.radio("Chart type", ["Bar","Line","Pie"], horizontal=True)

if chart_type in ("Bar", "Line"):
    x_col = st.selectbox("X axis", options=[c for c in (["time"] if "time" in gdf.columns else [])] + group_cols + ["category","region"], index=0 if "time" in gdf.columns else 0)
    y_col = st.selectbox("Y axis (numeric)", options=["value","units"], index=0)
    color_col = st.selectbox("Color (optional)", options=["None"] + [c for c in ["category","region"] if c in gdf.columns], index=0)
    color = None if color_col == "None" else color_col

    if chart_type == "Bar":
        fig = px.bar(gdf, x=x_col, y=y_col, color=color, barmode="group")
    else:
        fig = px.line(gdf.sort_values(by=x_col), x=x_col, y=y_col, color=color, markers=True)

    st.plotly_chart(fig, use_container_width=True)

else:  # Pie
    # For pie, require a single grouping dimension
    pie_dim = st.selectbox("Pie slice by", options=[c for c in ["category","region"] if c in gdf.columns] or ["category"])
    value_col = st.selectbox("Value", options=["value","units"], index=0)
    pie_df = group_dataframe(df_plot, [pie_dim], agg)
    fig = px.pie(pie_df, names=pie_dim, values=value_col, hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Tip: Adjust rows/seed in the sidebar to regenerate the dataset. Caching keeps things snappy.")
