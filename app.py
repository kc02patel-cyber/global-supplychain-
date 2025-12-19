import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

# ======================================================
# APP CONFIG
# ======================================================
st.set_page_config(
    layout="wide",
    page_title="Global Supply Chain Decision Intelligence Platform"
)

# ======================================================
# DATA LOADING
# ======================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["date"])
    return df

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_data(uploaded) if uploaded else load_data("global_supply_chain.csv")

# ======================================================
# GLOBAL FILTERS
# ======================================================
st.sidebar.header("Global Filters")

regions = st.sidebar.multiselect(
    "Region", df["region"].unique(), default=df["region"].unique()
)

products = st.sidebar.multiselect(
    "Product Category",
    df["product_category"].unique(),
    default=df["product_category"].unique()
)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Base Case", "Demand Surge", "Port Delay", "Fuel Price Spike"]
)

df = df[(df["region"].isin(regions)) & (df["product_category"].isin(products))]

# ======================================================
# SCENARIO ENGINE
# ======================================================
df_sim = df.copy()

if scenario == "Demand Surge":
    df_sim["monthly_demand"] *= 1.3
elif scenario == "Port Delay":
    df_sim["lead_time_days"] += 10
    df_sim["delay_probability"] += 0.15
elif scenario == "Fuel Price Spike":
    df_sim["fuel_cost"] *= 1.35
    df_sim["logistics_cost"] *= 1.25

# ======================================================
# TABS = MULTI-PAGE ENTERPRISE APP
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🧩 Clustering & Segmentation",
    "⚠️ Risk Classification",
    "📈 Forecasting & Planning"
])

# ======================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ======================================================
with tab1:
    st.subheader("Executive KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Lead Time", f"{df_sim.lead_time_days.mean():.1f} days")
    c2.metric("Avg Logistics Cost", f"${df_sim.logistics_cost.mean():,.0f}")
    c3.metric("Service Level", f"{df_sim.service_level.mean():.2f}")
    c4.metric("Delay Probability", f"{df_sim.delay_probability.mean():.2f}")

    st.subheader("Cost & Performance Trends")

    fig = px.line(
        df_sim.groupby("date")[["logistics_cost", "fuel_cost"]].mean().reset_index(),
        x="date",
        y=["logistics_cost", "fuel_cost"],
        title="Logistics vs Fuel Cost Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This view summarizes operational health under the selected scenario. "
        "All downstream analytics respond to these assumptions."
    )

# ======================================================
# TAB 2 — K-MEANS + PCA CLUSTERING (3D)
# ======================================================
with tab2:
    st.subheader("Supply Chain Segmentation (K-Means + PCA)")

    features = df_sim[
        [
            "logistics_cost",
            "lead_time_days",
            "delay_probability",
            "demand_volatility",
            "monthly_demand"
        ]
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Elbow Method
    distortions = []
    K = range(2, 9)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    elbow_fig = px.line(
        x=list(K),
        y=distortions,
        labels={"x": "K", "y": "Inertia"},
        title="Elbow Method for Optimal K"
    )
    st.plotly_chart(elbow_fig, use_container_width=True)

    k = st.slider("Select Number of Clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca, columns=["PC1", "PC2", "PC3"]
    )
    pca_df["Cluster"] = clusters
    pca_df["Region"] = df_sim["region"].values

    fig_3d = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        symbol="Region",
        title="3D PCA-Based Supply Chain Clusters"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(
        """
        **Interpretation:**  
        - Clusters represent structurally different supply chain profiles  
        - Separation indicates resilience vs fragility trade-offs  
        - PCA ensures interpretability while preserving variance
        """
    )

# ======================================================
# TAB 3 — CLASSIFICATION (RF + GBM)
# ======================================================
with tab3:
    st.subheader("Delay Risk Classification")

    df_model = df_sim.copy()
    df_model["high_delay_risk"] = (df_model["delay_probability"] > 0.6).astype(int)

    X = df_model[
        [
            "logistics_cost",
            "lead_time_days",
            "fuel_cost",
            "monthly_demand",
            "demand_volatility"
        ]
    ]
    y = df_model["high_delay_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model_choice = st.selectbox(
        "Select Classification Model",
        ["Random Forest", "Gradient Boosting"]
    )

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.success(
        "Model explains which operational conditions materially increase delay risk."
    )

# ======================================================
# TAB 4 — FORECASTING & SCENARIO PLANNING
# ======================================================
with tab4:
    st.subheader("Cost Forecasting Engine")

    forecast_metric = st.selectbox(
        "Forecast Metric",
        ["logistics_cost", "fuel_cost", "lead_time_days"]
    )

    df_ts = (
        df_sim.groupby("date")[forecast_metric]
        .mean()
        .reset_index()
    )

    df_ts["t"] = np.arange(len(df_ts))

    model = LinearRegression()
    model.fit(df_ts[["t"]], df_ts[forecast_metric])

    horizon = st.slider("Forecast Horizon (Months)", 3, 18, 6)

    future_t = np.arange(len(df_ts), len(df_ts) + horizon)
    forecast = model.predict(future_t.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "date": pd.date_range(
            df_ts["date"].max(), periods=horizon + 1, freq="MS"
        )[1:],
        "forecast": forecast
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ts["date"], y=df_ts[forecast_metric],
        name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        name="Forecast"
    ))

    fig.update_layout(title=f"{forecast_metric} Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Forecast adapts to scenario assumptions and enables forward-looking decisions."
    )

# ======================================================
# FOOTER
# ======================================================
st.caption(
    "Global Supply Chain Decision Intelligence Platform | Portfolio-Grade | Enterprise Ready"
)
