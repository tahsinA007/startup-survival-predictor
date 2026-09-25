# Startup Survival Time Predictor

A deployable regression app estimating startup runway (in months) from
financial, customer, and business inputs. Built on the original Kaggle
notebook's pipeline, with three fixes applied for deployment:

1. **No leakage**: imputation now happens inside the sklearn Pipeline
   (fit on train folds only), instead of on the full dataset pre-split.
2. **No invalid predictions**: replaced the notebook's `max(pred, 1)`
   post-hoc patch with a proper `ClippedLinearRegression` estimator that
   floors predictions at 1 month. (A log1p target transform was tested
   first but made R² *worse* — this dataset behaves linearly, not
   log-linearly — so clipping was the correct fix, not a workaround.)
3. **Defensible metrics**: 5-fold CV reported alongside the single
   train/test split (mean R² = 0.894, std = 0.005).

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
