import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Patch

st.set_page_config(page_title="AI Focus Analyzer", layout="wide")
st.title(" AI-Powered Focus Analyzer")
st.markdown("Analyze your daily activities and visualize your productivity patterns!")

# -----------------------------
# 1. CSV Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your activity CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # -----------------------------
    # 2. Category-Based Productivity Weights
    # -----------------------------
    category_weights = {"Work": 1.0, "Neutral": 0.5, "Distraction": 0.1}
    data["productivity_score"] = data["category"].map(category_weights)
    data["productivity_score"].fillna(0.1, inplace=True)

    # -----------------------------
    # 3. Sidebar Filters
    # -----------------------------
    categories = st.sidebar.multiselect(
        "Select categories to include",
        options=data["category"].unique(),
        default=data["category"].unique()
    )
    filtered_data = data[data["category"].isin(categories)]

    # -----------------------------
    # 4. Focus Score Calculation
    # -----------------------------
    filtered_data["weighted_time"] = filtered_data["time_minutes"] * filtered_data["productivity_score"]
    focus_score = (filtered_data["weighted_time"].sum() / filtered_data["time_minutes"].sum()) * 100
    focus_score = round(focus_score, 2)

    num_activities = filtered_data.shape[0]
    switch_penalty = min(num_activities * 2, 15)
    adjusted_focus_score = max(focus_score - switch_penalty, 0)
    adjusted_focus_score = round(adjusted_focus_score, 2)

    # -----------------------------
    # 5. Focus Status Logic
    # -----------------------------
    if adjusted_focus_score >= 75:
        status = "Deep Focus 🟢"
    elif adjusted_focus_score >= 50:
        status = "Distracted 🟡"
    else:
        status = "Burnout Risk 🔴"

    # -----------------------------
    # 6. Summary Metrics
    # -----------------------------
    total_time = filtered_data["time_minutes"].sum()
    high_focus_time = filtered_data[filtered_data["productivity_score"] >= 0.7]["time_minutes"].sum()
    distraction_time = filtered_data[filtered_data["productivity_score"] < 0.4]["time_minutes"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Focus Score", f"{adjusted_focus_score}/100", delta=f"-{switch_penalty}% context penalty")
    col2.metric("Total Time", f"{total_time} min")
    col3.metric("High-Focus Time", f"{high_focus_time} min")
    col4.metric("Low-Focus Time", f"{distraction_time} min")

    # Progress bar
    st.progress(int(adjusted_focus_score))

    # -----------------------------
    # 7. Color-Coded Bar Chart
    # -----------------------------
    def get_activity_color(score):
        if score >= 0.7: return "green"
        elif score >= 0.4: return "orange"
        else: return "red"

    colors = filtered_data["productivity_score"].apply(get_activity_color)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(filtered_data["activity"], filtered_data["time_minutes"], color=colors)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Time Spent (minutes)")
    ax.set_title("Daily Activity Distribution (Focus-Aware)")
    ax.tick_params(axis='x', rotation=30)
    legend_elements = [
        Patch(facecolor='green', label='High Focus'),
        Patch(facecolor='orange', label='Neutral'),
        Patch(facecolor='red', label='Low Focus')
    ]
    ax.legend(handles=legend_elements)
    st.pyplot(fig)

    # -----------------------------
    # 8. ML Clustering (Optional Visualization)
    # -----------------------------
    X = filtered_data[["time_minutes", "productivity_score"]]
    if len(filtered_data) >= 3:  # KMeans requires at least 3 samples for 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        filtered_data["cluster"] = kmeans.fit_predict(X)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for cluster in filtered_data["cluster"].unique():
            cluster_data = filtered_data[filtered_data["cluster"] == cluster]
            ax2.scatter(cluster_data["time_minutes"], cluster_data["productivity_score"], s=100, label=f"Cluster {cluster}")
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Productivity Score")
        ax2.set_title("Activity Clusters (Behavior Patterns)")
        ax2.legend()
        st.pyplot(fig2)
