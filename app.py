import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Customer Persona Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# Sidebar – Data Source
# =====================================================
st.sidebar.title("📂 Data Source")
df = None

if os.path.exists("data/processed_data.parquet"):
    df = pd.read_parquet("data/processed_data.parquet")
    st.sidebar.success("Loaded local Parquet file")
elif os.path.exists("data/processed_data.csv"):
    df = pd.read_csv("data/processed_data.csv")
    st.sidebar.success("Loaded local CSV file")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer Dataset (CSV or Parquet)",
        type=["csv", "parquet"]
    )
    if uploaded_file is None:
        st.info("⬅️ Please upload a dataset to start")
        st.stop()
    if uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully")

# =====================================================
# Feature Engineering – Safe & Robust
# =====================================================
def ensure_column(df, col_name, default_value=0):
    if col_name not in df.columns:
        df[col_name] = default_value
    return df[col_name]

# Ensure columns exist
campaign = ensure_column(df, "campaign", 0)
previous = ensure_column(df, "previous", 0)
duration = ensure_column(df, "duration", 0)

# Avoid zero division
duration_norm = duration.astype(float) / max(duration.max(), 1e-6)

# Engagement Score
df["engagement_score"] = campaign.astype(float) + previous.astype(float) + duration_norm

# Persistence Score
df["persistence_score"] = previous.astype(float) + 1

# Financial Exposure
housing = ensure_column(df, "housing", "no")
loan = ensure_column(df, "loan", "no")
df["financial_exposure"] = ((housing == "yes").astype(int) + (loan == "yes").astype(int))

# Persona Assignment (Business Rules)
eng_q75 = df["engagement_score"].quantile(0.75)
eng_med = df["engagement_score"].median()

def assign_persona(row):
    if row["engagement_score"] >= eng_q75:
        return "Highly Engaged Loyalist"
    elif row["financial_exposure"] >= 2:
        return "Financially Stressed Repeater"
    elif row["engagement_score"] < eng_med:
        return "Curious Safe Explorer"
    else:
        return "Moderate Potential"

df["persona"] = df.apply(assign_persona, axis=1)

# =====================================================
# Persona Colors
# =====================================================
PERSONA_COLORS = {
    "Highly Engaged Loyalist": "#1f77b4",
    "Moderate Potential": "#2ca02c",
    "Curious Safe Explorer": "#ff7f0e",
    "Financially Stressed Repeater": "#d62728"
}

# =====================================================
# Sidebar Filters
# =====================================================
st.sidebar.title("🎯 Filters")
persona_filter = st.sidebar.multiselect(
    "Select Personas",
    options=sorted(df["persona"].unique()),
    default=sorted(df["persona"].unique())
)

# Safe slider
eng_min = float(df["engagement_score"].min())
eng_max = float(df["engagement_score"].max())
if eng_min == eng_max:
    engagement_range = (eng_min, eng_max)
else:
    engagement_range = st.sidebar.slider(
        "Engagement Score Range",
        min_value=eng_min,
        max_value=eng_max,
        value=(eng_min, eng_max)
    )

filtered_df = df[
    (df["persona"].isin(persona_filter)) &
    (df["engagement_score"].between(*engagement_range))
]

# =====================================================
# Header
# =====================================================
st.title("Customer Persona Intelligence Dashboard")
st.markdown(
    "Business-driven personas that transform raw customer data into **actionable marketing and risk insights**."
)

# =====================================================
# KPI Section
# =====================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(filtered_df):,}")
c2.metric("Active Personas", filtered_df["persona"].nunique())
c3.metric(
    "High Engagement %",
    f"{(filtered_df['engagement_score'] > filtered_df['engagement_score'].median()).mean()*100:.1f}%"
)
c4.metric(
    "At-Risk %",
    f"{(filtered_df['financial_exposure'] > 1).mean()*100:.1f}%"
)

st.divider()

# =====================================================
# Persona Distribution
# =====================================================
st.subheader("Persona Distribution")
dist = filtered_df["persona"].value_counts(normalize=True).reset_index()
dist.columns = ["persona", "share"]
fig_pie = px.pie(
    dist,
    names="persona",
    values="share",
    hole=0.55,
    color="persona",
    color_discrete_map=PERSONA_COLORS
)
fig_pie.update_traces(textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================
# Persona Deep Dive
# =====================================================
st.subheader("Persona Deep Dive")
selected = st.selectbox(
    "Select Persona",
    filtered_df["persona"].unique()
)
pdf = filtered_df[filtered_df["persona"] == selected]
m1, m2, m3 = st.columns(3)
m1.metric("Avg Engagement", f"{pdf['engagement_score'].mean():.2f}")
m2.metric("Avg Persistence", f"{pdf['persistence_score'].mean():.2f}")
m3.metric("Avg Financial Exposure", f"{pdf['financial_exposure'].mean():.2f}")

fig_radar = go.Figure(go.Scatterpolar(
    r=[
        pdf["engagement_score"].mean(),
        pdf["persistence_score"].mean(),
        pdf["financial_exposure"].mean()
    ],
    theta=["Engagement", "Persistence", "Financial Exposure"],
    fill="toself",
    line_color=PERSONA_COLORS[selected]
))
fig_radar.update_layout(showlegend=False)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# =====================================================
# Risk vs Engagement
# =====================================================
st.subheader("Risk vs Engagement")
fig_scatter = px.scatter(
    filtered_df,
    x="engagement_score",
    y="financial_exposure",
    color="persona",
    color_discrete_map=PERSONA_COLORS,
    hover_data=["persistence_score"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================
# Footer Insight
# =====================================================
st.success(
    "This dashboard automatically adapts raw datasets into business personas, "
    "enabling smarter targeting, risk control, and strategic decision-making."
)
