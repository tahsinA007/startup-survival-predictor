**Problem Statement**

Startups often face uncertainty regarding how long they can continue operating due to factors such as available cash, monthly expenses, revenue growth, customer retention, profitability, funding stage, and employee attrition.
The objective of this project is to build a machine learning regression model that predicts the expected Startup Survival Time (in months) based on financial, customer, employee, and business-related features.
The project compares multiple regression algorithms, including Multiple Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression, to identify the model that provides the most accurate predictions.

[`notebook/startup-survival-time-prediction1.ipynb`](notebook/startup-survival-time-prediction1.ipynb).

**🛠️ Tools & Technologies**
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Development:** Kaggle Notebook
- **Version Control:** Git, GitHub

## Files 
- `train.py` — retrains the pipeline from the original CSV and saves `model_bundle.joblib`
- `model_utils.py` — the custom `ClippedLinearRegression` estimator (must be importable for joblib to load the pipeline)
- `model_bundle.joblib` — the trained, ready-to-serve pipeline
- `app.py` — the Streamlit app
- `requirements.txt` — dependencies for Streamlit Community Cloud

🔗 **Live app:** [https://startup-survival-predictor.streamlit.app](https://startup-longevity-predictor.streamlit.app/)

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a public (or private, if you have Cloud access) GitHub repo.
2. Go to https://share.streamlit.io → "New app" → pick the repo, branch, and `app.py` as the entry point.
3. Deploy. You'll get a URL like `https://<name>.streamlit.app` — that's your CV/LinkedIn link.

## Run locally first (recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```
