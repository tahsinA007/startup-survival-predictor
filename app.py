import joblib
import numpy as np
import pandas as pd
import streamlit as st
from model_utils import ClippedLinearRegression  # noqa: F401 -- needed for joblib unpickling

st.set_page_config(page_title="Startup Survival Time Predictor", page_icon="📈", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")

bundle = load_bundle()
pipeline = bundle["pipeline"]
categories = bundle["categories"]
ranges = bundle["numeric_ranges"]

st.title("📈 Startup Survival Time Predictor")
st.caption(
    "Estimates how many months a startup's current cash + burn + growth profile "
    "could sustain it, based on a linear regression model."
)
st.info(
    "⚠️ **Trained on a synthetic dataset for demonstration purposes.** "
    "This is a portfolio ML project, not financial or investment advice — "
    "do not use it to make real funding or operating decisions.",
    icon="ℹ️",
)

with st.expander("Model details"):
    st.write(f"- Held-out test R²: **{bundle['test_r2']:.3f}**")
    st.write(f"- Held-out test RMSE: **{bundle['test_rmse']:.1f} months**")
    st.write(f"- 5-fold cross-validated R²: **{bundle['cv_r2_mean']:.3f}**")
    st.write("- Model: Linear Regression on a log-scaled + one-hot pipeline, predictions floored at 1 month")

st.subheader("Enter your startup's numbers")

col1, col2 = st.columns(2)
inputs = {}

with col1:
    inputs["Funding_Stage"] = st.selectbox("Funding Stage", categories["Funding_Stage"])
    inputs["Industry_Sector"] = st.selectbox("Industry Sector", categories["Industry_Sector"])
    inputs["Business_Model"] = st.selectbox("Business Model", categories["Business_Model"])
    inputs["Cash_Available($)"] = st.number_input(
        "Cash Available ($k)", min_value=0.0,
        value=round(ranges["Cash_Available($)"][2], 1), step=1.0,
    )
    inputs["Monthly_Burn_Rate($/month)"] = st.number_input(
        "Monthly Burn Rate ($k/month)", min_value=0.01,
        value=round(ranges["Monthly_Burn_Rate($/month)"][2], 1), step=0.5,
    )
    inputs["Monthly_Revenue($/month)"] = st.number_input(
        "Monthly Revenue ($k/month)", min_value=0.0,
        value=round(ranges["Monthly_Revenue($/month)"][2], 1), step=0.5,
    )
    inputs["CAC($/customer)"] = st.number_input(
        "CAC ($/customer)", min_value=0.01,
        value=round(ranges["CAC($/customer)"][2], 1), step=1.0,
    )

with col2:
    inputs["LTV($/customer)"] = st.number_input(
        "LTV ($/customer)", min_value=0.0,
        value=round(ranges["LTV($/customer)"][2], 1), step=1.0,
    )
    inputs["Revenue_Growth_Rate(%)"] = st.slider("Revenue Growth Rate (%/month)", -30.0, 50.0, float(round(ranges["Revenue_Growth_Rate(%)"][2], 1)))
    inputs["Customer_Churn_Rate(%)"] = st.slider("Customer Churn Rate (%)", 0.0, 30.0, float(round(ranges["Customer_Churn_Rate(%)"][2], 1)))
    inputs["Customer_Growth_Rate(%)"] = st.slider("Customer Growth Rate (%/month)", -20.0, 40.0, float(round(ranges["Customer_Growth_Rate(%)"][2], 1)))
    inputs["Gross_Profit_Margin(%)"] = st.slider("Gross Profit Margin (%)", 0.0, 100.0, float(round(ranges["Gross_Profit_Margin(%)"][2], 1)))
    inputs["Employee_Attrition(%)"] = st.slider("Employee Attrition (%)", 0.0, 60.0, float(round(ranges["Employee_Attrition(%)"][2], 1)))

if st.button("Predict Survival Time", type="primary", use_container_width=True):
    row = pd.DataFrame([inputs])[bundle["feature_columns"]]
    pred = pipeline.predict(row)[0]
    years = pred / 12

    st.markdown("### Result")
    m1, m2 = st.columns(2)
    m1.metric("Predicted Survival Time", f"{pred:.1f} months")
    m2.metric("≈", f"{years:.1f} years")

    if pred < 12:
        st.warning("Under a year of projected runway at this profile — burn rate vs. cash/revenue is the tightest lever.")
    elif pred < 36:
        st.info("Moderate runway — revenue growth and churn are usually the biggest levers from here.")
    else:
        st.success("Strong projected runway on this profile.")

st.divider()
st.caption("Built with scikit-learn + Streamlit · Linear Regression pipeline (OneHot + log-scaling) · [source dataset via Kaggle]")