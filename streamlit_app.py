"""
Churn Observatory - Streamlit Frontend
A bespoke, editorial B2B SaaS interface for Customer Churn Prediction.
Strictly adheres to an editorial, white-dominant aesthetic with burgundy accents.
"""
import os
import textwrap
import requests
from typing import Dict, Any, Tuple, Optional
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Churn Observatory · Customer Intelligence",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Configuration & API Connection
# -----------------------------------------------------------------------------
def get_api_base_url() -> str:
    """
    Resolves the backend API base URL in order of precedence:
    1. Streamlit secrets: st.secrets["API_BASE_URL"]
    2. Environment variable: API_BASE_URL
    3. Default fallback requested: http://127.0.0.1:8081
    """
    try:
        if hasattr(st, "secrets") and "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass

    if "API_BASE_URL" in os.environ:
        return os.environ["API_BASE_URL"].rstrip("/")
    return "http://127.0.0.1:8081"


API_BASE_URL = get_api_base_url()


def probe_api_health(base_url: str) -> Tuple[bool, str]:
    """
    Pings the /health endpoint with a strict 1.5s timeout.
    Returns (is_online, message).
    """
    try:
        resp = requests.get(f"{base_url}/health", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("model_loaded", False):
                return True, "MODEL ONLINE"
            return False, "MODEL DEGRADED"
        return False, f"STATUS {resp.status_code}"
    except requests.exceptions.RequestException:
        # Fallback probe to 8000 if 8081 is unreachable and no explicit secret set
        if base_url.endswith("8081") and not (hasattr(st, "secrets") and "API_BASE_URL" in st.secrets):
            try:
                alt_resp = requests.get("http://127.0.0.1:8000/health", timeout=1.0)
                if alt_resp.status_code == 200 and alt_resp.json().get("model_loaded"):
                    return True, "ONLINE (PORT 8000)"
            except Exception:
                pass
        return False, "MODEL OFFLINE"


# Probe health once per run
is_model_online, health_status_text = probe_api_health(API_BASE_URL)

# -----------------------------------------------------------------------------
# Custom Editorial Styling (Strict Palette & Typography)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

    /* Global reset & typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #252323;
        background-color: #FFFFFF;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: #FFFFFF;
        max-width: 1280px;
        margin: 0 auto;
        padding: 0 1.5rem 3rem 1.5rem;
    }

    /* Hide standard Streamlit header, footer, and menu */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }

    /* Top Navigation bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 0 1rem 0;
        border-bottom: 1px solid #EAE6E3;
        margin-bottom: 2.5rem;
    }

    .nav-brand {
        display: flex;
        align-items: baseline;
        gap: 0.85rem;
    }

    .brand-wordmark {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        color: #252323;
        text-transform: uppercase;
    }

    .brand-descriptor {
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        color: #77716F;
        text-transform: uppercase;
    }

    .nav-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.2rem 0.55rem;
        border-radius: 2px;
        border: 1px solid #EAE6E3;
        background: #FAF9F7;
        color: #252323;
    }

    .status-dot-online {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2A5A3B;
    }

    .status-dot-offline {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #8B2635;
    }

    .nav-monogram {
        font-size: 0.65rem;
        font-weight: 700;
        color: #762F43;
        background: #F8F1F3;
        border: 1px solid #EAE6E3;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 2px;
        letter-spacing: 0.05em;
    }

    /* Hero section */
    .hero-container {
        padding: 0.5rem 0 2rem 0;
        border-bottom: 1px solid #EAE6E3;
        margin-bottom: 2.5rem;
    }

    .eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #762F43;
        margin-bottom: 0.6rem;
    }

    .hero-headline {
        font-family: 'Newsreader', serif;
        font-size: 2.85rem;
        line-height: 1.15;
        font-weight: 400;
        color: #252323;
        margin: 0 0 0.85rem 0;
        letter-spacing: -0.02em;
    }

    .hero-subtext {
        font-size: 0.98rem;
        line-height: 1.6;
        color: #77716F;
        max-width: 680px;
        margin: 0;
    }

    /* Section Headings */
    .section-eyebrow {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #762F43;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .section-eyebrow::after {
        content: "";
        flex: 1;
        height: 1px;
        background: #EAE6E3;
    }

    .form-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #252323;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }

    .form-desc {
        font-size: 0.85rem;
        color: #77716F;
        margin-bottom: 1.5rem;
    }

    .field-group-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #252323;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #EAE6E3;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }

    .field-group-num {
        color: #762F43;
        font-weight: 700;
    }

    /* Streamlit widget modifications */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.75rem;
    }

    /* Labels */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        color: #252323 !important;
        text-transform: none !important;
        margin-bottom: 0.25rem !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border-color: #EAE6E3 !important;
        border-radius: 3px !important;
        color: #252323 !important;
        font-size: 0.85rem !important;
        transition: border-color 0.15s ease !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #762F43 !important;
    }

    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: #762F43 !important;
        box-shadow: 0 0 0 1px #762F43 !important;
    }

    /* Primary submit button */
    div.stButton > button {
        background-color: #762F43 !important;
        color: #FFFFFF !important;
        border: 1px solid #762F43 !important;
        border-radius: 3px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.6rem !important;
        margin-top: 1.25rem !important;
        width: 100% !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: none !important;
    }

    div.stButton > button:hover {
        background-color: #602437 !important;
        border-color: #602437 !important;
        color: #FFFFFF !important;
    }

    div.stButton > button:active {
        background-color: #4A1928 !important;
        border-color: #4A1928 !important;
    }

    /* Secondary reset button */
    .reset-btn div.stButton > button {
        background-color: #FAF9F7 !important;
        color: #77716F !important;
        border: 1px solid #EAE6E3 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.45rem 0.9rem !important;
        margin-top: 0.5rem !important;
        width: auto !important;
    }

    .reset-btn div.stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #252323 !important;
        border-color: #77716F !important;
    }

    /* Prediction Workspace Container */
    .workspace-box {
        background: #FAF9F7;
        border: 1px solid #EAE6E3;
        border-radius: 4px;
        padding: 2rem;
        position: sticky;
        top: 2rem;
    }

    .workspace-empty-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #252323;
        margin-top: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .workspace-empty-desc {
        font-size: 0.82rem;
        color: #77716F;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    .empty-graphic-guide {
        border: 1px dashed #EAE6E3;
        background: #FFFFFF;
        border-radius: 3px;
        padding: 1.5rem 1rem;
        text-align: center;
        font-size: 0.72rem;
        color: #77716F;
        letter-spacing: 0.04em;
    }

    /* Prediction Result Styles */
    .result-badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        border-radius: 2px;
        margin-bottom: 0.9rem;
    }

    .badge-churn {
        background-color: #FDF2F4;
        border: 1px solid #F5CDD3;
        color: #8B2635;
    }

    .badge-no-churn {
        background-color: #F3F7F4;
        border: 1px solid #D4E2D8;
        color: #2A5A3B;
    }

    .result-outcome {
        font-family: 'Newsreader', serif;
        font-size: 2.4rem;
        font-weight: 400;
        line-height: 1.1;
        letter-spacing: -0.01em;
        color: #252323;
        margin-bottom: 1.25rem;
    }

    .probability-number {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #252323;
        line-height: 1;
        font-feature-settings: "tnum";
    }

    .prob-bar-container {
        width: 100%;
        height: 6px;
        background-color: #EAE6E3;
        border-radius: 2px;
        overflow: hidden;
        margin: 0.85rem 0 1.25rem 0;
    }

    .prob-bar-fill-churn {
        height: 100%;
        background-color: #762F43;
        transition: width 0.4s ease-in-out;
    }

    .prob-bar-fill-safe {
        height: 100%;
        background-color: #2A5A3B;
        transition: width 0.4s ease-in-out;
    }

    .result-interpretation {
        font-size: 0.86rem;
        line-height: 1.55;
        color: #252323;
        background: #FFFFFF;
        border: 1px solid #EAE6E3;
        border-radius: 3px;
        padding: 1rem;
        margin-bottom: 1.25rem;
    }

    .result-caption {
        font-size: 0.68rem;
        line-height: 1.45;
        color: #77716F;
        border-top: 1px solid #EAE6E3;
        padding-top: 0.85rem;
    }

    /* Footer */
    .app-footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #EAE6E3;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        color: #77716F;
        letter-spacing: 0.04em;
    }

    .footer-left {
        font-weight: 500;
    }

    .footer-right {
        font-feature-settings: "tnum";
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top Navigation
# -----------------------------------------------------------------------------
status_dot_class = "status-dot-online" if is_model_online else "status-dot-offline"

st.markdown(
    f"""
    <div class="top-nav">
        <div class="nav-brand">
            <span class="brand-wordmark">CHURN / OBSERVATORY</span>
            <span class="brand-descriptor">CUSTOMER INTELLIGENCE</span>
        </div>
        <div class="nav-status">
            <span class="status-pill">
                <span class="{status_dot_class}"></span>
                {health_status_text}
            </span>
            <div class="nav-monogram">ML</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Editorial Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="eyebrow">CUSTOMER RISK ANALYSIS</div>
        <h1 class="hero-headline">See churn before it happens.</h1>
        <p class="hero-subtext">
            Evaluate customer behaviour and estimate churn risk using your trained machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Backend URL fallback check / configuration alert
# -----------------------------------------------------------------------------
effective_api_url = API_BASE_URL
if not is_model_online:
    # Check if 8000 is active (common local uvicorn port)
    try:
        r8000 = requests.get("http://127.0.0.1:8000/health", timeout=0.8)
        if r8000.status_code == 200 and r8000.json().get("model_loaded"):
            effective_api_url = "http://127.0.0.1:8000"
            is_model_online = True
    except Exception:
        pass

# Initialize session state for prediction result
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# -----------------------------------------------------------------------------
# Main Workspace: Two-Column Asymmetric Layout (approx 60% / 40%)
# -----------------------------------------------------------------------------
col_form, col_pred = st.columns([1.52, 1.0], gap="large")

# =============================================================================
# LEFT COLUMN: Customer Profile Form
# =============================================================================
with col_form:
    st.markdown('<div class="section-eyebrow">INPUT PROFILE</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-title">Customer profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="form-desc">Enter the customer\'s current account details to estimate churn risk.</div>',
        unsafe_allow_html=True,
    )

    with st.form("customer_profile_form", clear_on_submit=False):

        # ---------------------------------------------------------------------
        # Group 01: Account & Plan
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="field-group-header"><span>01 — Account &amp; Plan</span>'
            '<span class="field-group-num">5 FIELDS</span></div>',
            unsafe_allow_html=True,
        )

        territory_options = [
            "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
            "AK", "AL", "AR", "AZ", "CO", "CT", "DC", "DE", "HI", "IA",
            "ID", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MN", "MO",
            "MS", "MT", "ND", "NE", "NH", "NJ", "NM", "NV", "OK", "OR",
            "RI", "SC", "SD", "TN", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
        ]

        g1_col1, g1_col2 = st.columns(2)
        with g1_col1:
            territory = st.selectbox(
                "Territory",
                options=territory_options,
                index=0,
                help="Customer billing region or US state code",
            )
            device_type = st.selectbox(
                "Device Type",
                options=["iPhone", "Galaxy", "-"],
                index=0,
                help="Primary mobile device associated with account",
            )
            plan_type = st.selectbox(
                "Plan Type",
                options=["Purple Plan", "Blue Plan", "Green Plan", "Red Plan"],
                index=0,
                help="Active subscription plan tier",
            )

        with g1_col2:
            promotion = st.selectbox(
                "Promotional Signup",
                options=["no", "yes"],
                index=0,
                help="Whether customer enrolled via a promotional discount",
            )
            has_renewed = st.selectbox(
                "Has Renewed",
                options=["yes", "no"],
                index=0,
                help="Whether the customer has renewed their subscription previously",
            )

        # ---------------------------------------------------------------------
        # Group 02: Usage & Spend
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="field-group-header"><span>02 — Usage &amp; Spend</span>'
            '<span class="field-group-num">4 FIELDS</span></div>',
            unsafe_allow_html=True,
        )

        g2_col1, g2_col2 = st.columns(2)
        with g2_col1:
            base_fee = st.number_input(
                "Base Monthly Fee ($)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=1.0,
                format="%.2f",
                help="Regular monthly subscription fee charged",
            )
            current_period_usage = st.number_input(
                "Current Period Usage (units)",
                min_value=0.0,
                max_value=2000.0,
                value=120.0,
                step=5.0,
                format="%.1f",
                help="Total usage units in the current billing cycle",
            )

        with g2_col2:
            additional_feature_spend = st.number_input(
                "Additional Feature Spend ($)",
                min_value=0.0,
                max_value=500.0,
                value=15.50,
                step=1.0,
                format="%.2f",
                help="Expenditure on add-on packages and supplemental features",
            )
            prior_period_usage = st.number_input(
                "Prior Period Usage (units)",
                min_value=0.0,
                max_value=2000.0,
                value=115.0,
                step=5.0,
                format="%.1f",
                help="Total usage units recorded in the preceding billing cycle",
            )

        # ---------------------------------------------------------------------
        # Group 03: Customer Experience
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="field-group-header"><span>03 — Customer Experience</span>'
            '<span class="field-group-num">6 FIELDS</span></div>',
            unsafe_allow_html=True,
        )

        g3_col1, g3_col2 = st.columns(2)
        with g3_col1:
            customer_tenure = st.number_input(
                "Customer Tenure (months)",
                min_value=0,
                max_value=360,
                value=24,
                step=1,
                help="Months account has remained continuously active",
            )
            service_rating = st.number_input(
                "Service Satisfaction Rating",
                min_value=0.0,
                max_value=15.0,
                value=4.0,
                step=0.5,
                format="%.1f",
                help="Feedback score recorded from customer service assessments",
            )
            number_of_penalties = st.number_input(
                "Number of Penalties",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="Count of penalty fees, payment defaults, or violations",
            )

        with g3_col2:
            service_tickets = st.number_input(
                "Service Support Tickets",
                min_value=0,
                max_value=50,
                value=1,
                step=1,
                help="Number of customer service or technical tickets logged",
            )

            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            start_month = st.selectbox(
                "Start Month",
                options=months,
                index=0,
                help="Calendar month subscription originated",
            )

            weeks = [f"W{i:02d}" for i in range(1, 54)]
            start_week = st.selectbox(
                "Start Week",
                options=weeks,
                index=1,
                help="Week index subscription originated",
            )

        # Submit Action
        submit_clicked = st.form_submit_button("Analyse customer risk")

    # Form Submission Handler
    if submit_clicked:
        payload = {
            "Territory": territory,
            "DeviceType": device_type,
            "Promotion": promotion,
            "HasRenewed": has_renewed,
            "PlanType": plan_type,
            "BaseFee": float(base_fee),
            "AdditionalFeatureSpend": float(additional_feature_spend),
            "NumberOfPenalties": int(number_of_penalties),
            "CurrentPeriodUsage": float(current_period_usage),
            "PriorPeriodUsage": float(prior_period_usage),
            "ServiceRating": float(service_rating),
            "ServiceTickets": int(service_tickets),
            "StartMonth": start_month,
            "StartWeek": start_week,
            "CustomerTenure": int(customer_tenure),
        }

        target_url = f"{effective_api_url}/predict"
        try:
            resp = requests.post(target_url, json=payload, timeout=4.0)
            if resp.status_code == 200:
                st.session_state.prediction_result = resp.json()
                st.session_state.last_error = None
            else:
                st.session_state.prediction_result = None
                try:
                    err_json = resp.json()
                    st.session_state.last_error = f"API Error ({resp.status_code}): {err_json.get('detail', resp.text)}"
                except Exception:
                    st.session_state.last_error = f"API Error ({resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            st.session_state.prediction_result = None
            st.session_state.last_error = (
                f"Connection failure: unable to reach prediction service at {effective_api_url}. "
                "Ensure FastAPI backend is running via `python -m uvicorn app.main:app --reload`."
            )
        except requests.exceptions.Timeout:
            st.session_state.prediction_result = None
            st.session_state.last_error = "Request timed out while waiting for inference response."
        except Exception as e:
            st.session_state.prediction_result = None
            st.session_state.last_error = f"Unexpected client error: {str(e)}"

# =============================================================================
# RIGHT COLUMN: Prediction Workspace
# =============================================================================
with col_pred:
    st.markdown('<div class="section-eyebrow">PREDICTION WORKSPACE</div>', unsafe_allow_html=True)

    result = st.session_state.prediction_result
    error_msg = st.session_state.last_error

    if error_msg:
        st.markdown(
            textwrap.dedent(f"""
                <div class="workspace-box">
                    <div class="result-badge badge-churn">SERVICE NOTICE</div>
                    <div class="workspace-empty-title">Inference Unavailable</div>
                    <div class="workspace-empty-desc">{error_msg}</div>
                    <div class="empty-graphic-guide">
                        Expected API endpoint:<br><code>{effective_api_url}/predict</code>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    elif result is None:
        # Empty State
        st.markdown(
            textwrap.dedent("""
                <div class="workspace-box">
                    <div class="result-badge" style="background: #FFFFFF; border: 1px solid #EAE6E3; color: #77716F;">
                        STANDBY
                    </div>
                    <div class="workspace-empty-title">Awaiting customer profile</div>
                    <div class="workspace-empty-desc">
                        Complete the profile and run the model to view the estimated churn outcome.
                    </div>
                    <div class="empty-graphic-guide">
                        15 features configured<br>
                        Decision Tree Pipeline Ready
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    else:
        # Active Prediction Outcome
        pred_label = result.get("prediction", "Unknown")
        churn_prob = float(result.get("churn_probability", 0.0))
        prob_pct = int(round(churn_prob * 100))
        is_churn = (pred_label == "Churn") or (result.get("predicted_class") == 1)

        badge_class = "badge-churn" if is_churn else "badge-no-churn"
        badge_text = "ELEVATED CHURN RISK" if is_churn else "HEALTHY RETENTION"
        bar_fill_class = "prob-bar-fill-churn" if is_churn else "prob-bar-fill-safe"

        if is_churn:
            interpretation_text = (
                f"The Decision Tree classifier estimated a <strong>{prob_pct}% probability of churn</strong>. "
                "Account attributes (such as usage fluctuation, penalties, or service inquiries) place this "
                "customer in an at-risk cluster."
            )
        else:
            interpretation_text = (
                f"The Decision Tree classifier estimated a <strong>{prob_pct}% probability of churn</strong>. "
                "Current tenure, spend, and satisfaction patterns indicate steady account continuity."
            )

        result_html = textwrap.dedent(f"""
            <div class="workspace-box">
                <div class="result-badge {badge_class}">{badge_text}</div>
                <div class="result-outcome">{pred_label}</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; color: #77716F; text-transform: uppercase;">
                        Model Estimate
                    </span>
                    <span class="probability-number">{prob_pct}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="{bar_fill_class}" style="width: {max(prob_pct, 2)}%;"></div>
                </div>
                <div class="result-interpretation">
                    {interpretation_text}
                </div>
                <div class="result-caption">
                    Model estimate based on trained Decision Tree pipeline; not a calibrated risk score or guarantee of future behaviour.
                </div>
            </div>
        """).strip()
        st.markdown(result_html, unsafe_allow_html=True)

        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("Run another analysis"):
            st.session_state.prediction_result = None
            st.session_state.last_error = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Compact Editorial Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-left">
            Customer Churn Prediction · Machine Learning
        </div>
        <div class="footer-right">
            SCIKIT-LEARN PIPELINE · FASTAPI BACKEND
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
