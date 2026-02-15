import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Risk Predictor", layout="wide")

st.title("🎓 Student Dropout Risk Prediction System")

# Load trained model
model = joblib.load("models/xgb_model.pkl")

st.sidebar.header("Upload Student Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with student features",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Load training feature list
    feature_cols = joblib.load("models/feature_columns.pk1")

# Keep only training features (ignore extra columns automatically)
    X = data[feature_cols].copy()
    # ---------------- Prediction ----------------
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    data["Predicted Risk"] = preds
    data["Risk Probability"] = probs
    data["Risk Label"] = data["Predicted Risk"].map({0: "Safe", 1: "At Risk"})

    st.subheader("Prediction Results")
    st.dataframe(data)

    # ---------------- Risk Distribution ----------------
    st.subheader("Risk Distribution")

    fig1, ax1 = plt.subplots()
    data["Risk Label"].value_counts().plot(kind="bar", ax=ax1, color=["green", "red"])
    ax1.set_xlabel("Risk Category")
    ax1.set_ylabel("Number of Students")
    st.pyplot(fig1)

    # ---------------- Risk Probability Histogram ----------------
    st.subheader("Risk Probability Distribution")

    fig2, ax2 = plt.subplots()
    sns.histplot(data["Risk Probability"], bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("Risk Probability")
    st.pyplot(fig2)

    # ---------------- Average Risk Metric ----------------
    st.metric("Average Risk Probability", round(data["Risk Probability"].mean(), 3))

    # ---------------- Feature Importance ----------------
    st.subheader("Top Features Influencing Risk")

    importances = model.feature_importances_
    feature_names = X.columns

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig3, ax3 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax3)
    st.pyplot(fig3)

    # ---------------- Cluster vs Risk (if cluster present) ----------------
    if "cluster_label" in data.columns:
        st.subheader("Cluster vs Risk Distribution")

        cluster_risk = pd.crosstab(data["cluster_label"], data["Risk Label"])

        fig4, ax4 = plt.subplots()
        cluster_risk.plot(kind="bar", stacked=True, ax=ax4)
        ax4.set_ylabel("Number of Students")
        st.pyplot(fig4)

else:
    st.info("Upload a CSV file to start prediction")