# ============================================
# 🚀 PRODUCTION STREAMLIT APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(layout="wide")

# ============================================
# 🎨 SIMPLE STYLE
# ============================================
st.markdown("""
<style>
.main {background-color: #0f172a;}
h1, h2, h3 {color: #e2e8f0;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Social Media Intelligence Dashboard")

# ============================================
# 📦 SAFE LOADERS
# ============================================

@st.cache_data
def load_youtube():
    path = "youtube_features.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # SAFE columns
    df.rename(columns={
        "views": "total_views",
        "likes": "total_likes",
        "comments": "total_comments"
    }, inplace=True)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["publish_hour"] = df["published_at"].dt.hour
    else:
        df["publish_hour"] = 0

    df["like_ratio"] = df["total_likes"] / (df["total_views"] + 1)
    df["comment_ratio"] = df["total_comments"] / (df["total_views"] + 1)

    return df


@st.cache_data
def load_instagram():
    path = "instagram_features.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # 🔥 FIX KEYERROR published_at
    possible_cols = ["published_at", "timestamp", "created_time", "date"]

    found = None
    for col in possible_cols:
        if col in df.columns:
            found = col
            break

    if found:
        df["post_hour"] = pd.to_datetime(df[found], errors="coerce").dt.hour
    else:
        df["post_hour"] = 0

    return df


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


yt = load_youtube()
ig = load_instagram()
model = load_model()

# ============================================
# 📌 SIDEBAR
# ============================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Welcome", "YouTube Hub", "Instagram Hub", "Prediction"]
)

# ============================================
# 🏠 WELCOME
# ============================================

if page == "Welcome":
    st.header("👋 Welcome")

    st.markdown("""
    This dashboard provides:

    - 🎥 YouTube analytics (views, engagement, trends)
    - 📸 Instagram behavior insights
    - 🤖 Prediction models
    """)

# ============================================
# 🎥 YOUTUBE HUB
# ============================================

elif page == "YouTube Hub":
    st.header("🎥 YouTube Strategy Hub")

    if yt.empty:
        st.warning("No YouTube data found")
        st.stop()

    # ===== METRICS =====
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Videos", len(yt))
    col2.metric("Total Views", int(yt["total_views"].sum()))
    col3.metric("Avg Engagement", round(yt["engagement_rate"].mean(), 4))

    # ===== DISTRIBUTION =====
    st.subheader("Views Distribution")

    fig, ax = plt.subplots()
    sns.histplot(yt["total_views"], ax=ax)
    st.pyplot(fig)

    # ===== CORRELATION =====
    st.subheader("Correlation")

    fig, ax = plt.subplots()
    corr = yt[["total_views","total_likes","total_comments"]].corr()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # ===== TIME ANALYSIS =====
    st.subheader("Views by Hour")

    hourly = yt.groupby("publish_hour")["total_views"].mean()

    fig, ax = plt.subplots()
    hourly.plot(ax=ax)
    st.pyplot(fig)

    # ===== TOP VIDEOS =====
    st.subheader("Top Videos")

    top = yt.sort_values("total_views", ascending=False).head(10)
    st.dataframe(top[["title","channel","total_views"]])

# ============================================
# 📸 INSTAGRAM HUB
# ============================================

elif page == "Instagram Hub":
    st.header("📸 Instagram Engagement Hub")

    if ig.empty:
        st.warning("No Instagram data found")
        st.stop()

    col1, col2, col3 = st.columns(3)

    col1.metric("Posts", len(ig))
    col2.metric("Avg Viral Score", round(ig["viral_score"].mean(), 4))
    col3.metric("Avg Engagement", round(ig["engagement_per_impression"].mean(), 4))

    # ===== BEHAVIOR =====
    st.subheader("Like vs Comment")

    fig, ax = plt.subplots()
    ax.scatter(ig["like_ratio"], ig["comment_ratio"], alpha=0.3)
    st.pyplot(fig)

    # ===== CATEGORY =====
    if "content_category" in ig.columns:
        st.subheader("Category Performance")

        cat = ig.groupby("content_category")["viral_score"].mean()

        fig, ax = plt.subplots()
        cat.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ===== TIME =====
    st.subheader("Viral Score by Hour")

    hour = ig.groupby("post_hour")["viral_score"].mean()

    fig, ax = plt.subplots()
    hour.plot(ax=ax)
    st.pyplot(fig)

# ============================================
# 🤖 PREDICTION
# ============================================

elif page == "Prediction":
    st.header("🤖 Prediction")

    if model is None:
        st.error("Model not found")
        st.stop()

    col1, col2, col3 = st.columns(3)

    day = col1.slider("Day", 1, 31, 15)
    month = col2.slider("Month", 1, 12, 3)
    hour = col3.slider("Hour", 0, 23, 20)

    if st.button("Predict Views"):

        df_input = pd.DataFrame({
            "day": [day],
            "month": [month],
            "hour": [hour]
        })

        log_pred = model.predict(df_input)[0]
        views = int(np.expm1(log_pred))

        st.success(f"📈 Estimated Views: {views:,}")