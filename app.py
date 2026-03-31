import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

# ==================== HELPERS ====================

def get_risk_level(prob):
    if prob < 0.4:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def get_card_color(prob):
    if prob < 0.4:
        return "#16a34a"
    elif prob < 0.7:
        return "#eab308"
    else:
        return "#dc2626"

# ==================== PAGE ====================

st.set_page_config(page_title="Student Risk Predictor", layout="wide")

st.title("🎓 Student Dropout Risk Prediction System")
st.markdown("AI-powered insights to identify at-risk students")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("📁 Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    threshold = st.slider("Risk Threshold", 0.3, 0.9, 0.5, 0.05)

    st.divider()

    show_shap = st.checkbox("Enable SHAP Explainability", value=True)

    sample_size = st.slider("SHAP Sample Size", 50, 300, 100)

# ==================== MAIN ====================

if uploaded_file is not None:
    try:
        # -------- LOAD --------
        data = pd.read_csv(uploaded_file)

        model = joblib.load("models/xgb_model.pkl")
        feature_cols = joblib.load("models/feature_columns.pk1")  # change if needed

        # -------- VALIDATE --------
        missing = set(feature_cols) - set(data.columns)
        if missing:
            st.error(f"Missing features: {missing}")
            st.stop()

        X = data[feature_cols].copy()

        # -------- PREDICT --------
        probs = model.predict_proba(X)[:, 1]
        preds = (probs > threshold).astype(int)

        data["Risk Probability"] = probs
        data["Risk Level"] = [get_risk_level(p) for p in probs]

        # -------- METRICS --------
        st.subheader("📊 Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(data))
        col2.metric("At Risk", int((preds == 1).sum()))
        col3.metric("Avg Risk", round(probs.mean(), 3))

        st.write(f"🔧 Current Threshold: {threshold}")

        st.divider()

        # -------- DISTRIBUTION --------
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            pd.Series(preds).value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots()
            ax.hist(probs, bins=20)
            st.pyplot(fig)
            plt.close()

        st.divider()

        # -------- FEATURE IMPORTANCE --------
        st.subheader("📈 Feature Importance")

        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ==================== SHAP ====================
        if show_shap:
            st.subheader("🔍 SHAP Explainability")

            size = min(sample_size, len(X))
            idx = np.random.choice(len(X), size, replace=False)
            X_sample = X.iloc[idx]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            col1, col2 = st.columns(2)

            # ---- SUMMARY ----
            with col1:
                st.markdown("**Summary Plot**")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.close()

            # ---- DEPENDENCE ----
            with col2:
                st.markdown("**Dependence Plot**")
                feature = st.selectbox("Select Feature", X_sample.columns)

                fig, ax = plt.subplots()
                shap.dependence_plot(feature, shap_values, X_sample, ax=ax, show=False)
                st.pyplot(fig)
                plt.close()

            st.divider()

            # ---- INDIVIDUAL ----
            st.markdown("### 📌 Individual Student Analysis")

            i = st.slider("Select Student", 0, len(X_sample) - 1, 0)

            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

            # ---- CARD ----
            prob = probs[idx][i]
            color = get_card_color(prob)

            st.markdown(f"""
            <div style="
                background-color: {color};
                padding: 15px;
                border-radius: 10px;
                color: white;
                font-weight: bold;
            ">
                Student ID: {idx[i]} <br>
                Risk Probability: {prob:.3f}
            </div>
            """, unsafe_allow_html=True)

            # ---- WATERFALL ----
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[i],
                    base_values=expected_value,
                    data=X_sample.iloc[i],
                    feature_names=X_sample.columns
                ),
                show=False
            )
            st.pyplot(fig)
            plt.close()

            # ---- WHY ----
            st.markdown("### 📌 Why this student is at risk")

            top_features = np.argsort(np.abs(shap_values[i]))[-3:][::-1]

            for f in top_features:
                name = X_sample.columns[f]
                value = X_sample.iloc[i][name]
                impact = shap_values[i][f]

                if impact > 0:
                    st.write(f"🔴 {name} ({value:.2f}) is increasing risk")
                else:
                    st.write(f"🟢 {name} ({value:.2f}) is reducing risk")

            # ---- INTERVENTIONS ----
            st.markdown("### 🛠 Suggested Interventions")

            for f in top_features:
                name = X_sample.columns[f]
                impact = shap_values[i][f]

                if impact > 0:
                    if "content" in name or "homepage" in name:
                        st.write("📘 Increase engagement with learning materials")
                    elif "quiz" in name:
                        st.write("📝 Provide more quiz practice")
                    elif "forum" in name:
                        st.write("💬 Encourage discussion participation")
                    else:
                        st.write(f"⚠️ Improve performance in {name}")

            # ---- STATUS ----
            if prob > 0.7:
                st.warning("⚠️ High intervention required")
            elif prob > 0.4:
                st.info("📊 Monitor student closely")
            else:
                st.success("✅ Student performing well")

        st.divider()

        # -------- RESULTS --------
        st.subheader("📋 Results")
        st.dataframe(data.head(20))

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV file to start")