import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from model_utils import ClippedLinearRegression  # noqa: F401 -- needed for joblib unpickling

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="StartupLens | Startup Survival Time Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# PATHS / MODEL LOADING
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model_bundle.joblib"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {BUNDLE_PATH}")
    return joblib.load(BUNDLE_PATH)


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(
        "StartupLens could not load the trained model bundle. "
        "Make sure `model_bundle.joblib` and `model_utils.py` are in the same folder as `app.py`."
    )
    st.caption(f"Details: {exc}")
    st.stop()

pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
categories = bundle["categories"]
ranges = bundle["numeric_ranges"]

TEST_R2 = float(bundle.get("test_r2", 0.0))
TEST_RMSE = float(bundle.get("test_rmse", 0.0))
CV_R2 = float(bundle.get("cv_r2_mean", 0.0))

# =============================================================================
# SESSION STATE
# =============================================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "theme" not in st.session_state:
    st.session_state.theme = "Violet"

if "mode" not in st.session_state:
    st.session_state.mode = "dark"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =============================================================================
# THEME CONFIG
# =============================================================================
THEMES = {
    "Violet": {
        "accent": "#8B5CF6",
        "accent2": "#6366F1",
        "accent_soft": "#DDD6FE",
    },
    "Ocean": {
        "accent": "#0EA5E9",
        "accent2": "#2563EB",
        "accent_soft": "#BAE6FD",
    },
    "Emerald": {
        "accent": "#10B981",
        "accent2": "#059669",
        "accent_soft": "#A7F3D0",
    },
}

if st.session_state.mode == "dark":
    COLORS = {
        "bg": "#07111F",
        "bg2": "#0B1528",
        "surface": "#101C31",
        "surface2": "#16243D",
        "border": "#2A3A56",
        "text": "#F8FAFC",
        "muted": "#AAB7CC",
        "button_text": "#000000",
    }
else:
    COLORS = {
        "bg": "#F5F7FB",
        "bg2": "#EEF2F8",
        "surface": "#FFFFFF",
        "surface2": "#F7F8FC",
        "border": "#D7DDE8",
        "text": "#111827",
        "muted": "#5B6474",
        "button_text": "#000000",
    }

COLORS.update(THEMES[st.session_state.theme])

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown(
    f"""
    <style>
    /* ================================================================
       APP BACKGROUND / TYPOGRAPHY
       ================================================================ */
    .stApp {{
        background:
            radial-gradient(900px 500px at 10% -10%, {COLORS['accent']}20, transparent 60%),
            radial-gradient(900px 500px at 95% 5%, {COLORS['accent2']}18, transparent 58%),
            {COLORS['bg']};
        color: {COLORS['text']};
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text']} !important;
    }}

    p, label, span {{
        color: {COLORS['text']};
    }}

    .muted {{
        color: {COLORS['muted']} !important;
    }}

    /* ================================================================
       CARDS
       ================================================================ */
    .hero-card,
    .card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }}

    .hero-card {{
        background:
            linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface2']} 100%);
        overflow: hidden;
    }}

    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 18px;
        height: 100%;
    }}

    .metric-label {{
        color: {COLORS['muted']};
        font-size: 12px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: .06em;
        font-weight: 700;
    }}

    .metric-value {{
        color: {COLORS['text']};
        font-size: 28px;
        line-height: 1.1;
        font-weight: 900;
    }}

    .section-kicker {{
        color: {COLORS['accent2']} !important;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .info-strip {{
        border: 1px solid {COLORS['border']};
        background: {COLORS['surface2']};
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 10px;
    }}

    .result-number {{
        color: {COLORS['accent2']} !important;
        font-size: 54px;
        font-weight: 950;
        line-height: 1;
        margin: 8px 0;
    }}

    .result-sub {{
        color: {COLORS['muted']} !important;
        font-size: 15px;
    }}

    .feature-icon {{
        font-size: 26px;
        margin-bottom: 8px;
    }}

    .small-title {{
        color: {COLORS['text']} !important;
        font-weight: 900;
        font-size: 15px;
        margin-bottom: 4px;
    }}

    .small-text {{
        color: {COLORS['muted']} !important;
        font-size: 13px;
        line-height: 1.55;
    }}

    /* ================================================================
       INPUTS
       ================================================================ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: {COLORS['surface']} !important;
        border-color: {COLORS['border']} !important;
    }}

    input, textarea {{
        color: {COLORS['text']} !important;
    }}

    /* ================================================================
       CRITICAL BUTTON FIX

       1) Label/icon = deep bold black in EVERY theme.
       2) The actual native Streamlit button receives the click even when
          the user clicks directly on text/icon inside the button.
       ================================================================ */
    div[data-testid="stButton"] > button,
    div.stButton > button {{
        width: 100% !important;
        min-height: 48px !important;
        border-radius: 13px !important;
        border: 1px solid rgba(0,0,0,.10) !important;
        background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%) !important;
        color: #000000 !important;
        font-weight: 950 !important;
        font-size: 14px !important;
        text-shadow: none !important;
        cursor: pointer !important;
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease !important;
    }}

    /* Streamlit wraps button text/icons in child elements. These children
       MUST NOT capture the pointer, otherwise only the button edge can be
       clicked. Let the real <button> receive the click everywhere. */
    div[data-testid="stButton"] > button *,
    div.stButton > button *,
    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div.stButton > button p,
    div.stButton > button span {{
        color: #000000 !important;
        font-weight: 950 !important;
        text-shadow: none !important;
        pointer-events: none !important;
        user-select: none !important;
    }}

    div[data-testid="stButton"] > button:hover,
    div.stButton > button:hover {{
        filter: brightness(1.05) saturate(1.08) !important;
        box-shadow: 0 8px 22px {COLORS['accent']}35 !important;
        transform: translateY(-1px) !important;
        color: #000000 !important;
    }}

    div[data-testid="stButton"] > button:active,
    div.stButton > button:active {{
        transform: translateY(0) !important;
        color: #000000 !important;
    }}

    div[data-testid="stButton"] > button:focus,
    div.stButton > button:focus {{
        outline: 3px solid {COLORS['accent']}55 !important;
        outline-offset: 2px !important;
        color: #000000 !important;
    }}

    /* Secondary / theme buttons stay black-text while still using the
       selected accent gradient. */
    .theme-button div[data-testid="stButton"] > button,
    .theme-button div.stButton > button {{
        min-height: 42px !important;
        font-size: 13px !important;
    }}

    /* ================================================================
       NAV BUTTONS
       ================================================================ */
    /* All three navigation buttons intentionally share the exact same
       dimensions and vertical alignment. */
    div[data-testid="stButton"] > button,
    div.stButton > button {{
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    /* ================================================================
       RESPONSIVE SPACING
       ================================================================ */
    @media (max-width: 900px) {{
        .hero-card, .card {{
            padding: 18px;
            border-radius: 16px;
        }}

        .result-number {{
            font-size: 42px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================
def go_to(page_name: str):
    st.session_state.page = page_name


def predict_startup(inputs: dict):
    row = pd.DataFrame([inputs])[feature_columns]
    pred = float(pipeline.predict(row)[0])
    pred = max(pred, 1.0)
    return pred


def format_money(value: float) -> str:
    return f"${value:,.1f}k"


def feature_card(icon: str, title: str, body: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="feature-icon">{icon}</div>
            <div class="small-title">{title}</div>
            <div class="small-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    # Brand
    brand_col, nav1, nav2, nav3, spacer, mode_col = st.columns(
        [2.4, 1.0, 1.0, 1.0, 2.6, 1.3]
    )

    with brand_col:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding-top:7px">
                <div style="
                    width:38px;height:38px;border-radius:11px;
                    display:flex;align-items:center;justify-content:center;
                    background:linear-gradient(135deg,{COLORS['accent']},{COLORS['accent2']});
                    font-size:21px;
                ">🚀</div>
                <div>
                    <div style="font-size:18px;font-weight:950;color:{COLORS['text']}">StartupLens</div>
                    <div style="font-size:11px;color:{COLORS['muted']}">Survival Time Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_specs = [
        (nav1, "Home"),
        (nav2, "Predictor"),
        (nav3, "Insights"),
    ]

    for col, page_name in nav_specs:
        with col:
            clicked = st.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
            )
            if clicked:
                go_to(page_name)
                st.rerun()

    with mode_col:
        mode_label = "☀️ Light" if st.session_state.mode == "dark" else "🌙 Dark"
        if st.button(
            mode_label,
            key="theme_mode_toggle",
            use_container_width=True,
        ):
            st.session_state.mode = (
                "light" if st.session_state.mode == "dark" else "dark"
            )
            st.rerun()

    st.markdown(
        f"<div style='height:1px;background:{COLORS['border']};margin:14px 0 22px'></div>",
        unsafe_allow_html=True,
    )


def render_theme_picker():
    st.markdown("<div class='section-kicker'>Appearance</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    for col, theme_name in zip((t1, t2, t3), THEMES.keys()):
        with col:
            label = f"{theme_name} ✓" if st.session_state.theme == theme_name else theme_name
            if st.button(
                label,
                key=f"theme_{theme_name}",
                use_container_width=True,
            ):
                st.session_state.theme = theme_name
                st.rerun()


# =============================================================================
# TOPBAR
# =============================================================================
render_topbar()

# Theme controls stay at the top of the application on every page.
render_theme_picker()

st.markdown(
    f"<div style='height:1px;background:{COLORS['border']};margin:10px 0 24px'></div>",
    unsafe_allow_html=True,
)

# =============================================================================
# PAGE: HOME
# =============================================================================
if st.session_state.page == "Home":
    st.markdown(
        "<div class='section-kicker'>STARTUP INTELLIGENCE</div>",
        unsafe_allow_html=True,
    )

    # Full-width hero keeps the Home page aligned cleanly while removing the
    # static "Estimated Survival Time" output panel.
    st.markdown(
        f"""
        <div class='hero-card'>
            <div style='font-size:12px;color:{COLORS['accent2']};font-weight:900;letter-spacing:.12em'>
                DATA-DRIVEN STARTUP ANALYSIS
            </div>
            <h1 style='font-size:48px;line-height:1.05;margin:12px 0 12px;font-weight:950'>
                Startup Survival<br>Time Predictor
            </h1>
            <p style='font-size:17px;line-height:1.7;color:{COLORS['muted']};max-width:850px;margin-bottom:0'>
                Turn your startup's financial, customer, and business metrics into an estimated
                survival time using a trained Linear Regression pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚀 Start Prediction →",
        key="home_start_prediction",
        type="primary",
        use_container_width=True,
    ):
        go_to("Predictor")
        st.rerun()

    st.markdown(
        """
        <div class='info-strip'>
            <span style='font-weight:900'>Portfolio ML Project</span>
            <span style='opacity:.7'> · </span>
            <span class='muted'>Built with Python, Pandas, NumPy, Scikit-learn and Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why Use StartupLens?")
    f1, f2, f3, f4 = st.columns(4, gap="medium")

    with f1:
        feature_card("📊", "Data Driven", "Uses startup financial, customer, and operational inputs to estimate survival time.")
    with f2:
        feature_card("🧠", "Reliable Pipeline", "Preprocessing and the trained regression model are packaged into one deployable pipeline.")
    with f3:
        feature_card("⚡", "Fast & Simple", "Enter the startup profile and get an estimated result in seconds.")
    with f4:
        feature_card("🎯", "Actionable Insights", "Use the result as a portfolio-level analytical signal and explore model information.")

    st.markdown("### How It Works")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:28px;font-weight:950;color:{COLORS['accent2']}'>01</div>
                <div class='small-title'>Enter Startup Details</div>
                <div class='small-text'>Provide funding stage, industry, business model, cash, burn, revenue, customer and workforce metrics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:28px;font-weight:950;color:{COLORS['accent2']}'>02</div>
                <div class='small-title'>Run the Model</div>
                <div class='small-text'>The trained pipeline applies the same preprocessing and Linear Regression model used during training.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:28px;font-weight:950;color:{COLORS['accent2']}'>03</div>
                <div class='small-title'>Explore the Outcome</div>
                <div class='small-text'>View the estimated survival time and then review model performance and interpretation metrics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: PREDICTOR
# =============================================================================
elif st.session_state.page == "Predictor":
    st.markdown(
        "<div class='section-kicker'>PREDICTION</div>",
        unsafe_allow_html=True,
    )
    st.title("Predict Startup Survival Time")
    st.caption(
        "Enter your startup details below to get an estimated survival time in months."
    )

    form_card = st.container()
    with form_card:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:20px;font-weight:950;color:{COLORS['text']}'>🏢 Startup Details</div>
                <div class='small-text' style='margin-top:4px'>All values are passed through the trained preprocessing and regression pipeline.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns(2, gap="large")
    inputs = {}

    # ----------------------------------------------------------------------
    # LEFT COLUMN
    # ----------------------------------------------------------------------
    with left:
        inputs["Funding_Stage"] = st.selectbox(
            "Funding Stage",
            categories["Funding_Stage"],
            index=(categories["Funding_Stage"].index("Series A") if "Series A" in categories["Funding_Stage"] else 0),
        )

        inputs["Industry_Sector"] = st.selectbox(
            "Industry Sector",
            categories["Industry_Sector"],
            index=(categories["Industry_Sector"].index("FinTech") if "FinTech" in categories["Industry_Sector"] else 0),
        )

        inputs["Business_Model"] = st.selectbox(
            "Business Model",
            categories["Business_Model"],
            index=(categories["Business_Model"].index("B2B SaaS") if "B2B SaaS" in categories["Business_Model"] else 0),
        )

        inputs["Cash_Available($)"] = st.number_input(
            "Cash Available (Thousand US Dollars)",
            min_value=0.0,
            value=float(round(ranges["Cash_Available($)"][2], 1)),
            step=1.0,
        )

        inputs["Monthly_Burn_Rate($/month)"] = st.number_input(
            "Monthly Burn Rate (Thousand US Dollars per Month)",
            min_value=0.01,
            value=float(round(ranges["Monthly_Burn_Rate($/month)"][2], 1)),
            step=0.5,
        )

        inputs["Monthly_Revenue($/month)"] = st.number_input(
            "Monthly Revenue (Thousand US Dollars per Month)",
            min_value=0.0,
            value=float(round(ranges["Monthly_Revenue($/month)"][2], 1)),
            step=0.5,
        )

        inputs["CAC($/customer)"] = st.number_input(
            "Customer Acquisition Cost (US Dollars per Customer)",
            min_value=0.01,
            value=float(round(ranges["CAC($/customer)"][2], 1)),
            step=1.0,
        )

    # ----------------------------------------------------------------------
    # RIGHT COLUMN
    # ----------------------------------------------------------------------
    with right:
        inputs["LTV($/customer)"] = st.number_input(
            "Lifetime Value (US Dollars per Customer)",
            min_value=0.0,
            value=float(round(ranges["LTV($/customer)"][2], 1)),
            step=1.0,
        )

        inputs["Revenue_Growth_Rate(%)"] = st.slider(
            "Revenue Growth Rate (Percentage per Month)",
            -30.0,
            50.0,
            float(round(ranges["Revenue_Growth_Rate(%)"][2], 1)),
        )

        inputs["Customer_Churn_Rate(%)"] = st.slider(
            "Customer Churn Rate (Percentage)",
            0.0,
            30.0,
            float(round(ranges["Customer_Churn_Rate(%)"][2], 1)),
        )

        inputs["Customer_Growth_Rate(%)"] = st.slider(
            "Customer Growth Rate (Percentage per Month)",
            -20.0,
            40.0,
            float(round(ranges["Customer_Growth_Rate(%)"][2], 1)),
        )

        inputs["Gross_Profit_Margin(%)"] = st.slider(
            "Gross Profit Margin (Percentage)",
            0.0,
            100.0,
            float(round(ranges["Gross_Profit_Margin(%)"][2], 1)),
        )

        inputs["Employee_Attrition(%)"] = st.slider(
            "Employee Attrition (Percentage)",
            0.0,
            60.0,
            float(round(ranges["Employee_Attrition(%)"][2], 1)),
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button(
        "🚀 Predict Survival Time →",
        key="predict_survival_time",
        type="primary",
        use_container_width=True,
    ):
        prediction = predict_startup(inputs)
        st.session_state.prediction = {
            "value": prediction,
            "inputs": dict(inputs),
        }
        st.rerun()

    st.markdown(
        """
        <div class='info-strip'>
            ℹ️ <b>Model note:</b> The trained pipeline includes preprocessing such as imputation,
            one-hot encoding, logarithmic transformation of selected monetary variables, scaling,
            and the deployed clipped Linear Regression estimator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # LATEST RESULT
    # ----------------------------------------------------------------------
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction["value"]
        years = pred / 12

        st.markdown("### Latest Prediction")
        r1, r2 = st.columns([1.2, 1], gap="large")

        with r1:
            st.markdown(
                f"""
                <div class='hero-card'>
                    <div class='section-kicker'>PREDICTION RESULT</div>
                    <div style='font-size:14px;color:{COLORS['muted']}'>Predicted Startup Survival Time</div>
                    <div class='result-number'>{pred:.1f} months</div>
                    <div class='result-sub'>Approximately {years:.1f} years</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class='card'>
                    <div class='section-kicker'>MODEL INFORMATION</div>
                    <div class='small-title'>Linear Regression</div>
                    <div style='height:10px'></div>
                    <div class='info-strip'>Held-out Test R²: <b>{TEST_R2:.3f}</b></div>
                    <div class='info-strip'>Held-out Test RMSE: <b>{TEST_RMSE:.2f} months</b></div>
                    <div class='info-strip'>5-Fold Cross-Validated R²: <b>{CV_R2:.3f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class='card'>
                <div class='section-kicker'>INTERPRETATION</div>
                <div class='small-text'>
                    The current input profile produces an estimated survival time of
                    <b style='color:{COLORS['text']}'>{pred:.1f} months</b>.
                    This is a machine-learning estimate based on the training dataset and should be interpreted
                    as a portfolio project output rather than a financial forecast or investment recommendation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "📊 View Detailed Insights →",
            key="view_insights_from_predictor",
            use_container_width=True,
        ):
            go_to("Insights")
            st.rerun()

# =============================================================================
# PAGE: INSIGHTS
# =============================================================================
else:
    st.markdown(
        "<div class='section-kicker'>MODEL INSIGHTS</div>",
        unsafe_allow_html=True,
    )
    st.title("Model & Prediction Insights")
    st.caption("Review the latest model output and the evaluation figures from the deployed pipeline.")

    if st.session_state.prediction is None:
        st.info("No prediction has been generated in this session yet. Open Predictor and run the model once.")
        if st.button("🚀 Go to Predictor →", key="insights_go_predictor", use_container_width=True):
            go_to("Predictor")
            st.rerun()
    else:
        pred = st.session_state.prediction["value"]
        years = pred / 12
        saved_inputs = st.session_state.prediction["inputs"]

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        metric_data = [
            (m1, "Predicted Survival Time", f"{pred:.1f} mo"),
            (m2, "Approximate Years", f"{years:.1f}"),
            (m3, "Test R²", f"{TEST_R2:.3f}"),
            (m4, "Cross-Validated R²", f"{CV_R2:.3f}"),
        ]
        for col, label, value in metric_data:
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value'>{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Latest Outcome")
        o1, o2 = st.columns(2, gap="large")

        with o1:
            st.markdown(
                f"""
                <div class='hero-card'>
                    <div class='section-kicker'>STARTUPLENS OUTCOME</div>
                    <div class='result-number'>{pred:.1f}</div>
                    <div class='result-sub'>estimated survival time in months</div>
                    <div class='info-strip'>Equivalent to approximately <b>{years:.1f} years</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with o2:
            st.markdown(
                f"""
                <div class='card'>
                    <div class='section-kicker'>MODEL PERFORMANCE</div>
                    <div class='small-title'>Linear Regression Evaluation</div>
                    <div class='info-strip'>Held-out Test R² <b>{TEST_R2:.3f}</b></div>
                    <div class='info-strip'>Held-out Test RMSE <b>{TEST_RMSE:.2f} months</b></div>
                    <div class='info-strip'>5-Fold Cross-Validated R² <b>{CV_R2:.3f}</b></div>
                    <div class='small-text' style='margin-top:10px'>
                        R² describes the proportion of target variance explained by the model on the evaluated data,
                        while RMSE expresses average prediction error magnitude in months.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Input Profile Used for the Latest Prediction")
        profile_df = pd.DataFrame(
            {
                "Feature": [
                    "Funding Stage",
                    "Industry Sector",
                    "Business Model",
                    "Cash Available (Thousand US Dollars)",
                    "Monthly Burn Rate (Thousand US Dollars per Month)",
                    "Monthly Revenue (Thousand US Dollars per Month)",
                    "Customer Acquisition Cost (US Dollars per Customer)",
                    "Lifetime Value (US Dollars per Customer)",
                    "Revenue Growth Rate (Percentage per Month)",
                    "Customer Churn Rate (Percentage)",
                    "Customer Growth Rate (Percentage per Month)",
                    "Gross Profit Margin (Percentage)",
                    "Employee Attrition (Percentage)",
                ],
                "Value": [
                    saved_inputs["Funding_Stage"],
                    saved_inputs["Industry_Sector"],
                    saved_inputs["Business_Model"],
                    f"{saved_inputs['Cash_Available($)']:.1f}",
                    f"{saved_inputs['Monthly_Burn_Rate($/month)']:.1f}",
                    f"{saved_inputs['Monthly_Revenue($/month)']:.1f}",
                    f"{saved_inputs['CAC($/customer)']:.1f}",
                    f"{saved_inputs['LTV($/customer)']:.1f}",
                    f"{saved_inputs['Revenue_Growth_Rate(%)']:.1f}",
                    f"{saved_inputs['Customer_Churn_Rate(%)']:.1f}",
                    f"{saved_inputs['Customer_Growth_Rate(%)']:.1f}",
                    f"{saved_inputs['Gross_Profit_Margin(%)']:.1f}",
                    f"{saved_inputs['Employee_Attrition(%)']:.1f}",
                ],
            }
        )
        st.dataframe(profile_df, hide_index=True, use_container_width=True)

        st.markdown("### Technical Notes")
        n1, n2, n3 = st.columns(3, gap="medium")

        with n1:
            feature_card(
                "🔧",
                "Preprocessing",
                "Imputation, one-hot encoding, logarithmic transformation for selected monetary variables, and standard scaling are handled inside the deployed pipeline.",
            )
        with n2:
            feature_card(
                "📐",
                "Evaluation",
                "The app exposes held-out Test R², Test RMSE, and five-fold cross-validated R² from the trained model bundle.",
            )
        with n3:
            feature_card(
                "🛡️",
                "Prediction Floor",
                "The deployed estimator clips predictions to a minimum of 1 month, preventing invalid non-positive survival-time outputs.",
            )

        st.markdown("### Continue")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "🔄 Make New Prediction",
                key="new_prediction_from_insights",
                use_container_width=True,
            ):
                go_to("Predictor")
                st.rerun()
        with c2:
            if st.button(
                "🏠 Back to Home",
                key="home_from_insights",
                use_container_width=True,
            ):
                go_to("Home")
                st.rerun() 


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f"""
    <div style='text-align:center;padding:30px 0 10px;color:{COLORS['muted']};font-size:12px'>
        StartupLens · Startup Survival Time Prediction · Linear Regression · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
