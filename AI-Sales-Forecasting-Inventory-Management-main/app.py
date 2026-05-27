import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, HRFlowable
)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory user store
# ─────────────────────────────────────────────────────────────────────────────
if "USERS" not in st.session_state:
    st.session_state.USERS = {
        "admin@mail.com": {"password": "admin123", "username": "Admin"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auth Page  — FIX: true vertical centre, no clipped heading
# ─────────────────────────────────────────────────────────────────────────────
def render_auth_page():
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    [data-testid="stSidebar"], header, footer,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { display: none !important; }

    html, body { height: 100% !important; margin: 0 !important; }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #0b0f1a !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* ── True vertical centering — no negative margins ── */
    [data-testid="stMain"] {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    [data-testid="stVerticalBlock"] > div { gap: 0 !important; padding: 0 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

    /* Card panel (middle column) */
    div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlock"] {
        background: #131929 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 16px !important;
        padding: 2.2rem 1.8rem 1.8rem !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6) !important;
    }

    div[data-testid="stTextInput"] > label { display: none !important; }
    div[data-testid="stTextInput"] > div > div > input {
        background: #0d1424 !important;
        border: 1px solid #1e2d45 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 0.85rem !important;
        font-family: 'Inter', sans-serif !important;
        height: 36px !important;
        transition: border-color 0.2s !important;
    }
    div[data-testid="stTextInput"] > div > div > input::placeholder { color: #2d3f5c !important; }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #5b5ff5 !important;
        box-shadow: 0 0 0 2px rgba(91,95,245,0.15) !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] { margin-bottom: 0 !important; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #5b5ff5 0%, #7c3aed 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1rem !important;
        height: 38px !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
        box-shadow: 0 4px 16px rgba(91,95,245,0.3) !important;
        transition: opacity 0.2s !important;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

    [data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.85rem !important; }

    .field-lbl {
        display: block;
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        margin: 0.6rem 0 0.25rem 0;
        font-family: 'Inter', sans-serif;
    }
    .card-head {
        font-size: 1.6rem; font-weight: 700; color: #f0f6fc;
        margin: 0 0 0.2rem 0; letter-spacing: -0.02em;
        font-family: 'Inter', sans-serif;
    }
    .card-sub {
        font-size: 0.87rem; color: #6b7a99;
        margin: 0 0 1rem 0; font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    USERS = st.session_state.USERS

    _, col, _ = st.columns([1.6, 2, 1.6])

    with col:
        if st.session_state.auth_mode == "login":
            st.markdown('<div class="card-head">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Sign in to Sales Forecasting Dashboard</div>', unsafe_allow_html=True)

            st.markdown('<span class="field-lbl">Email</span>', unsafe_allow_html=True)
            email = st.text_input("_e", key="li_email", placeholder="example@mail.com",
                                  label_visibility="collapsed")

            st.markdown('<span class="field-lbl">Password</span>', unsafe_allow_html=True)
            password = st.text_input("_p", key="li_pw", placeholder="Enter password",
                                     type="password", label_visibility="collapsed")

            if st.button("Login", key="li_btn", use_container_width=True):
                if not email or not password:
                    st.warning("Please fill in both fields.")
                elif email not in USERS:
                    st.error("No account found with that email.")
                elif USERS[email]["password"] != password:
                    st.error("Incorrect password.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.current_user  = USERS[email]["username"]
                    st.rerun()

            c1, c2 = st.columns([5, 2])
            c1.markdown("<span style='color:#4a5568;font-size:0.84rem;'>Don't have an account?</span>",
                        unsafe_allow_html=True)
            with c2:
                if st.button("Sign Up →", key="li_sw", use_container_width=True):
                    st.session_state.auth_mode = "signup"
                    st.rerun()

        else:
            st.markdown('<div class="card-head">Create Account</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Sign up and start forecasting in seconds.</div>',
                        unsafe_allow_html=True)

            st.markdown('<span class="field-lbl">Username</span>', unsafe_allow_html=True)
            username = st.text_input("_u", key="su_user", placeholder="Choose a username",
                                     label_visibility="collapsed")

            st.markdown('<span class="field-lbl">Email</span>', unsafe_allow_html=True)
            email = st.text_input("_e2", key="su_email", placeholder="example@mail.com",
                                  label_visibility="collapsed")

            st.markdown('<span class="field-lbl">Password</span>', unsafe_allow_html=True)
            password = st.text_input("_p2", key="su_pass", placeholder="Min. 6 characters",
                                     type="password", label_visibility="collapsed")

            st.markdown('<span class="field-lbl">Confirm Password</span>', unsafe_allow_html=True)
            confirm = st.text_input("_c", key="su_conf", placeholder="Repeat password",
                                    type="password", label_visibility="collapsed")

            if st.button("Create Account", key="su_btn", use_container_width=True):
                if not all([username, email, password, confirm]):
                    st.warning("Please fill in all fields.")
                elif len(password) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif email in USERS:
                    st.error("An account with this email already exists.")
                else:
                    st.session_state.USERS[email] = {"password": password, "username": username}
                    st.session_state.authenticated = True
                    st.session_state.current_user  = username
                    st.rerun()

            c1, c2 = st.columns([5, 2])
            c1.markdown("<span style='color:#4a5568;font-size:0.84rem;'>Already have an account?</span>",
                        unsafe_allow_html=True)
            with c2:
                if st.button("← Sign In", key="su_sw", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PDF Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(model_name, mae, rmse, mape, accuracy,
                 future, days, demand, stock, forecast_fig, eval_fig):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles   = getSampleStyleSheet()
    title_s  = ParagraphStyle("T2", parent=styles["Title"], fontSize=20, spaceAfter=4)
    head_s   = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#1f4e79"))
    normal_s = styles["Normal"]
    story    = []

    story.append(Paragraph("AI Sales Forecasting Report", title_s))
    story.append(Paragraph(f"Model: {model_name}  |  Forecast Horizon: {days} days", normal_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1f4e79")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Performance", head_s))
    perf_data = [
        ["Metric", "Value"],
        ["MAE",      f"{round(mae, 2):,}"],
        ["RMSE",     f"{round(rmse, 2):,}"],
        ["MAPE",     f"{round(mape, 2)}%"],
        ["Accuracy", f"{round(accuracy, 2)}%"],
    ]
    t = Table(perf_data, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("ALIGN",          (0,0),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#eaf1fb"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.grey),
        ("TOPPADDING",     (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Prediction vs Actual", head_s))
    buf2 = io.BytesIO()
    eval_fig.savefig(buf2, format="png", bbox_inches="tight", dpi=150)
    buf2.seek(0)
    story.append(RLImage(buf2, width=6*inch, height=2.8*inch))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Future Sales Forecast", head_s))
    buf3 = io.BytesIO()
    forecast_fig.savefig(buf3, format="png", bbox_inches="tight", dpi=150)
    buf3.seek(0)
    story.append(RLImage(buf3, width=6*inch, height=2.8*inch))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Daily Forecast Values", head_s))
    tdata = [["Day", "Forecasted Sales"]]
    for i, val in enumerate(future.flatten()):
        tdata.append([f"Day {i+1}", f"{int(val):,}"])
    ft = Table(tdata, colWidths=[2.5*inch, 2.5*inch])
    ft.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("ALIGN",          (0,0),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#eaf1fb"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.grey),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
    ]))
    story.append(ft)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Forecast Insights", head_s))
    story.append(Paragraph(f"Avg Forecasted Sales: {int(np.mean(future)):,}", normal_s))
    story.append(Paragraph(f"Max Forecasted Sales: {int(np.max(future)):,}", normal_s))
    story.append(Paragraph(f"Min Forecasted Sales: {int(np.min(future)):,}", normal_s))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Inventory Decision", head_s))
    story.append(Paragraph(f"Predicted Demand (Day 1): {demand:,}  |  Current Stock: {stock:,}", normal_s))
    story.append(Spacer(1, 6))
    if demand > stock:
        verdict = "ACTION REQUIRED: Demand exceeds stock. Consider restocking immediately."
        vc = colors.HexColor("#c00000")
    else:
        verdict = "Stock is sufficient to meet forecasted demand."
        vc = colors.HexColor("#375623")
    story.append(Paragraph(verdict,
        ParagraphStyle("vrd", parent=normal_s, textColor=vc, fontName="Helvetica-Bold")))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Model Helpers
# ─────────────────────────────────────────────────────────────────────────────
def calc_mape(actual, predicted):
    a, p = actual.flatten(), predicted.flatten()
    mask = a != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def run_lstm(sales_series, lookback):
    data        = sales_series.values.reshape(-1, 1)
    scaler      = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    split        = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    m = Sequential([LSTM(50, input_shape=(lookback, 1)), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(X_train, y_train, epochs=5, verbose=0)
    pred   = scaler.inverse_transform(m.predict(X_test, verbose=0))
    y_test = scaler.inverse_transform(y_test)
    return m, scaler, data_scaled, pred, y_test


def run_arima(sales_series):
    split      = int(len(sales_series) * 0.8)
    train_data = sales_series[:split]
    test_data  = sales_series[split:]
    fit        = ARIMA(train_data, order=(5, 1, 0)).fit()
    pred       = fit.forecast(len(test_data)).values.reshape(-1, 1)
    y_test     = test_data.values.reshape(-1, 1)
    return fit, pred, y_test


# ─────────────────────────────────────────────────────────────────────────────
# App Entry Point
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sales Forecasting", layout="wide")

if not st.session_state.get("authenticated", False):
    render_auth_page()
    st.stop()

# ── Inject CSS to pin user-info block to bottom of sidebar ───────────────────
st.markdown("""
<style>
/* Push sidebar content so user-info sits at the very bottom */
[data-testid="stSidebar"] > div:first-child {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    padding-bottom: 1.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
/* The last element in the sidebar vertical block = user info div → pin to bottom */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:last-child {
    margin-top: auto !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Sales Forecasting Dashboard")
st.info("Upload all dataset files to start analysis")

# ── Sidebar: Settings first, user-info last ──────────────────────────────────
st.sidebar.header("⚙️ Settings")
days = st.sidebar.slider("Forecast Days", 1, 30, 7)
st.sidebar.markdown("---")

st.sidebar.header("🧠 Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["LSTM", "ARIMA"],
                                    help="Manually pick a model to run.")
st.sidebar.markdown("---")

st.sidebar.header("🤖 Auto Model Selector")
auto_select = st.sidebar.toggle("Enable Auto Selection", value=False,
    help="Runs both LSTM and ARIMA, compares MAPE, picks the best automatically.")
if auto_select:
    st.sidebar.info("Both models will run and the best will be selected automatically.")
st.sidebar.markdown("---")

# ── Store-Level Forecast ─────────────────────────────────────────────
st.sidebar.header("🏪 Store-Level Forecast")

store_mode = st.sidebar.radio(
    "Forecast Mode",
    ["All Stores (Combined)", "Individual Store"],
    help="Combined = total of all stores. Individual = one store at a time."
)

selected_store = None

# ── Spacer pushes user block to bottom (CSS handles the rest) ────────────────
st.sidebar.markdown(
    "<div style='flex:1;'></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── User info + sign-out at very bottom ──────────────────────────────────────
st.sidebar.markdown(
    f"""<div style='padding:0.4rem 0 0.6rem 0;'>
    <span style='color:#64748b;font-size:0.75rem;text-transform:uppercase;
                 letter-spacing:0.08em;'>Signed in as</span><br>
    <span style='color:#e2e8f0;font-size:0.95rem;font-weight:600;'>
        {st.session_state.get('current_user','User')}
    </span>
    </div>""",
    unsafe_allow_html=True,
)
if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.current_user  = None
    st.rerun()

# ── File Uploaders ────────────────────────────────────────────────────────────
st.subheader("📂 Upload Dataset Files")
train_file    = st.file_uploader("Upload train.csv",    type=["csv"])
test_file     = st.file_uploader("Upload test.csv",     type=["csv"])
features_file = st.file_uploader("Upload features.csv", type=["csv"])
stores_file   = st.file_uploader("Upload stores.csv",   type=["csv"])

if not (train_file and test_file and features_file and stores_file):
    st.stop()

# ── Data Loading & Feature Engineering ───────────────────────────────────────
train    = pd.read_csv(train_file)
test     = pd.read_csv(test_file)
features = pd.read_csv(features_file)
stores   = pd.read_csv(stores_file)

df = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
df = df.merge(stores, on="Store", how="left")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

if "IsHoliday_feat" in df.columns:
    df["IsHoliday"] = df["IsHoliday"].fillna(df["IsHoliday_feat"])
    df.drop(columns=["IsHoliday_feat"], inplace=True)

available_stores = sorted(df["Store"].unique().tolist())

# Create placeholder exactly below radio button
store_placeholder = st.sidebar.empty()

if store_mode == "Individual Store":
    with store_placeholder:
        selected_store = st.selectbox(
            "Select Store",
            available_stores,
            format_func=lambda x: f"Store {x}",
            key="store_select"
        )

    df = df[df["Store"] == selected_store].copy()
    st.sidebar.success(f"Showing: Store {selected_store}")

else:
    selected_store = None
    st.sidebar.info(f"Showing: All {len(available_stores)} stores combined")

agg_dict = {"Weekly_Sales": "sum"}
for col, func in [("Temperature","mean"), ("Fuel_Price","mean"),
                   ("CPI","mean"), ("Unemployment","mean"), ("IsHoliday","max")]:
    if col in df.columns:
        agg_dict[col] = func

df = df.groupby("Date").agg(agg_dict).reset_index()
df.rename(columns={"Weekly_Sales": "Sales"}, inplace=True)

for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]:
    if col not in df.columns:
        df[col] = np.nan

df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
df["Month"]     = df["Date"].dt.month
df["Quarter"]   = df["Date"].dt.quarter
df["Year"]      = df["Date"].dt.year
df["DayOfYear"] = df["Date"].dt.dayofyear

n_rows = len(df)
df["Lag_1"] = df["Sales"].shift(1)
if n_rows > 20:
    df["Lag_4"] = df["Sales"].shift(4)
if n_rows > 100:
    df["Lag_52"] = df["Sales"].shift(52)

roll_long = min(12, max(2, n_rows // 5))
df["RollMean_4"]            = df["Sales"].shift(1).rolling(4).mean()
df["RollStd_4"]             = df["Sales"].shift(1).rolling(4).std()
df[f"RollMean_{roll_long}"] = df["Sales"].shift(1).rolling(roll_long).mean()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

LOOKBACK = min(10, max(2, len(df) - 21))

# ── Engineered Features Table ─────────────────────────────────────────────────
st.subheader("🛠 Engineered Temporal Features")
base_feat_cols = ["Date","Week","Month","Quarter","Lag_1","RollMean_4","RollStd_4",
                  "Temperature","Fuel_Price","CPI","Unemployment","IsHoliday"]
for opt in ["Lag_4", "Lag_52", f"RollMean_{roll_long}"]:
    if opt in df.columns:
        base_feat_cols.append(opt)
feat_cols = [c for c in base_feat_cols if c in df.columns]
st.dataframe(df[feat_cols].tail(10), use_container_width=True)

with st.expander("What are these features?"):
    st.markdown("""
| Feature | Description |
|---|---|
| **Week / Month / Quarter** | Calendar position — captures seasonality |
| **Lag_1** | Sales from 1 week ago — direct autoregressive signal |
| **Lag_4** | Sales from 4 weeks ago — monthly trend |
| **Lag_52** | Sales from 52 weeks ago — same week last year |
| **RollMean_4** | 4-week moving average — smoothed trend |
| **RollStd_4** | 4-week rolling std — volatility signal |
| **RollMean_12** | 12-week moving average — quarterly trend |
| **Temperature / Fuel / CPI / Unemployment** | External economic signals from features.csv |
| **IsHoliday** | Holiday flag — captures demand spikes |
""")

# ── Data Overview ─────────────────────────────────────────────────────────────
st.subheader("📊 Data Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train Records",    len(train))
c2.metric("Test Records",     len(test))
c3.metric("Avg Weekly Sales", f"{int(df['Sales'].mean()):,}")
if store_mode == "Individual Store":
    c4.metric("Store", f"Store {selected_store}")
else:
    c4.metric("Stores", f"{len(available_stores)} Combined")

sales = df["Sales"]

# ── Model Training ────────────────────────────────────────────────────────────
lstm_model       = None
lstm_scaler      = None
lstm_data_scaled = None
arima_fit        = None

if auto_select:
    with st.spinner("⏳ Auto Model Selection — running LSTM & ARIMA..."):
        lstm_model, lstm_scaler, lstm_data_scaled, lstm_pred, lstm_ytest = run_lstm(sales, LOOKBACK)
        arima_fit, arima_pred, arima_ytest = run_arima(sales)

    lstm_mape  = calc_mape(lstm_ytest,  lstm_pred)
    arima_mape = calc_mape(arima_ytest, arima_pred)
    lstm_acc   = max(0.0, 100 - lstm_mape)
    arima_acc  = max(0.0, 100 - arima_mape)

    st.subheader("🤖 Auto Model Selection")
    c1, c2, c3 = st.columns(3)
    c1.metric("ARIMA Accuracy", f"{round(arima_acc,2)}%",
              delta=f"MAPE: {round(arima_mape,2)}%", delta_color="inverse")
    c2.metric("LSTM Accuracy",  f"{round(lstm_acc,2)}%",
              delta=f"MAPE: {round(lstm_mape,2)}%",  delta_color="inverse")

    if lstm_acc >= arima_acc:
        active_model = "LSTM"
        pred, y_test = lstm_pred, lstm_ytest
        c3.success("✅ Using: **LSTM**")
        st.info(f"🏆 ARIMA: {round(arima_acc,2)}%  |  LSTM: {round(lstm_acc,2)}%  →  **Using LSTM**")
    else:
        active_model = "ARIMA"
        pred, y_test = arima_pred, arima_ytest
        c3.success("✅ Using: **ARIMA**")
        st.info(f"🏆 ARIMA: {round(arima_acc,2)}%  |  LSTM: {round(lstm_acc,2)}%  →  **Using ARIMA**")

    with st.expander("📊 See full model comparison"):
        comp_fig, comp_ax = plt.subplots(figsize=(10, 3))
        best_is_lstm = lstm_acc >= arima_acc
        comp_ax.bar(["ARIMA","LSTM"],
                    [arima_acc, lstm_acc],
                    color=["#2ecc71" if not best_is_lstm else "#3498db",
                           "#2ecc71" if best_is_lstm     else "#3498db"],
                    width=0.4)
        comp_ax.set_ylabel("Accuracy (%)")
        comp_ax.set_title("ARIMA vs LSTM — Accuracy Comparison")
        comp_ax.set_ylim(0, 110)
        for i, v in enumerate([arima_acc, lstm_acc]):
            comp_ax.text(i, v + 1, f"{round(v,2)}%", ha="center", fontweight="bold")
        comp_ax.grid(axis="y", alpha=0.3)
        st.pyplot(comp_fig)
        plt.close(comp_fig)
        st.dataframe(pd.DataFrame({
            "Model":        ["ARIMA", "LSTM"],
            "MAPE (%)":     [round(arima_mape,2), round(lstm_mape,2)],
            "Accuracy (%)": [round(arima_acc,2),  round(lstm_acc,2)],
        }), use_container_width=True, hide_index=True)

elif model_choice == "LSTM":
    active_model = "LSTM"
    with st.spinner("⏳ Training LSTM..."):
        lstm_model, lstm_scaler, lstm_data_scaled, pred, y_test = run_lstm(sales, LOOKBACK)
else:
    active_model = "ARIMA"
    with st.spinner("⏳ Running ARIMA..."):
        arima_fit, pred, y_test = run_arima(sales)

# ── Evaluation ────────────────────────────────────────────────────────────────
st.subheader("📈 Prediction vs Actual")
eval_fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(y_test, label="Actual",    color="steelblue", linewidth=1.5)
axes[0].plot(pred,   label="Predicted", color="tomato", linestyle="dashed", linewidth=1.5)
axes[0].set_title(f"Actual vs Predicted Sales  [{active_model}]")
axes[0].set_xlabel("Time Steps"); axes[0].set_ylabel("Sales")
axes[0].legend(); axes[0].grid(alpha=0.3)
residuals = y_test.flatten() - pred.flatten()
axes[1].bar(range(len(residuals)), residuals,
            color=["tomato" if r < 0 else "steelblue" for r in residuals], alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Residuals (Actual − Predicted)")
axes[1].set_xlabel("Time Steps"); axes[1].set_ylabel("Error"); axes[1].grid(alpha=0.3)
plt.tight_layout()
st.pyplot(eval_fig)

mae      = mean_absolute_error(y_test, pred)
rmse     = np.sqrt(mean_squared_error(y_test, pred))
mape     = calc_mape(y_test, pred)
accuracy = max(0.0, 100 - mape)

st.subheader("📉 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE",      f"{round(mae,  2):,}")
c2.metric("RMSE",     f"{round(rmse, 2):,}")
c3.metric("MAPE",     f"{round(mape, 2)}%")
c4.metric("Accuracy", f"{round(accuracy, 2)}%")

# ── Future Forecast ───────────────────────────────────────────────────────────
st.subheader("🔮 Future Forecast")
if active_model == "LSTM":
    last_seq = lstm_data_scaled[-LOOKBACK:].copy()
    future   = []
    for _ in range(days):
        inp      = last_seq.reshape(1, LOOKBACK, 1)
        next_val = lstm_model.predict(inp, verbose=0)
        future.append(next_val[0][0])
        last_seq = np.append(last_seq[1:], next_val).reshape(-1, 1)
    future = lstm_scaler.inverse_transform(np.array(future).reshape(-1, 1))
else:
    future = arima_fit.forecast(days).values.reshape(-1, 1)

forecast_df = pd.DataFrame(future, columns=["Forecasted Sales"],
                           index=[f"Day {i+1}" for i in range(days)])
st.line_chart(forecast_df)

st.subheader("📊 Forecast Insights")
c1, c2, c3 = st.columns(3)
c1.metric("Avg Forecasted Sales", f"{int(np.mean(future)):,}")
c2.metric("Max Forecasted Sales", f"{int(np.max(future)):,}")
c3.metric("Min Forecasted Sales", f"{int(np.min(future)):,}")

st.subheader("🤖 Inventory Decision")
demand = int(future[0][0])
stock  = int(df["Sales"].iloc[-4:].mean())
if demand > stock:
    st.error(f"⚠️ Demand ({demand:,}) exceeds stock ({stock:,}). Consider restocking.")
else:
    st.success(f"✅ Stock is sufficient. Demand ({demand:,}) ≤ Stock ({stock:,}).")

# ── PDF ───────────────────────────────────────────────────────────────────────
st.subheader("📥 Download Report")
forecast_fig, forecast_ax = plt.subplots(figsize=(8, 3))
forecast_ax.plot(range(1, days+1), future.flatten(), marker="o", color="steelblue", linewidth=2)
forecast_ax.set_xlabel("Day"); forecast_ax.set_ylabel("Sales")
forecast_ax.set_title(f"Future Sales Forecast [{active_model}]"); forecast_ax.grid(alpha=0.3)
plt.tight_layout()
pdf_bytes = generate_pdf(model_name=active_model, mae=mae, rmse=rmse, mape=mape,
                         accuracy=accuracy, future=future, days=days,
                         demand=demand, stock=stock,
                         forecast_fig=forecast_fig, eval_fig=eval_fig)
plt.close(forecast_fig)
st.download_button(label="⬇️ Download Forecast Report as PDF", data=pdf_bytes,
                   file_name="sales_forecast_report.pdf", mime="application/pdf")

st.success("🎉 Analysis Complete!")

# ── Store Comparison ──────────────────────────────────────────────────────────
if store_mode == "All Stores (Combined)":
    st.markdown("---")
    st.subheader("🏪 Store-Level Comparison")
    st.caption("Compare all stores side by side — switch to Individual Store in sidebar for a dedicated forecast.")

    store_stats = train.groupby("Store")["Weekly_Sales"].agg(
        Avg_Sales="mean", Total_Sales="sum", Weeks="count").reset_index()
    store_stats = store_stats.merge(stores, on="Store", how="left")
    store_stats["Avg_Sales"]   = store_stats["Avg_Sales"].astype(int)
    store_stats["Total_Sales"] = store_stats["Total_Sales"].astype(int)
    overall_avg = store_stats["Avg_Sales"].mean()
    store_stats["Status"] = store_stats["Avg_Sales"].apply(
        lambda x: "⚠️ Low Sales" if x < overall_avg * 0.85
        else ("✅ Normal" if x < overall_avg * 1.15 else "🔥 High Sales"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔝 Top 5 Stores by Avg Weekly Sales**")
        top5 = store_stats.nlargest(5,"Avg_Sales")[["Store","Avg_Sales","Type","Status"]]
        top5["Avg_Sales"] = top5["Avg_Sales"].apply(lambda x: f"{x:,}")
        st.dataframe(top5, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**📉 Bottom 5 Stores by Avg Weekly Sales**")
        bot5 = store_stats.nsmallest(5,"Avg_Sales")[["Store","Avg_Sales","Type","Status"]]
        bot5["Avg_Sales"] = bot5["Avg_Sales"].apply(lambda x: f"{x:,}")
        st.dataframe(bot5, use_container_width=True, hide_index=True)

    st.markdown("**📊 Avg Weekly Sales — All Stores**")
    fig_s, ax_s = plt.subplots(figsize=(14, 4))
    colors_bar = ["#e74c3c" if x < overall_avg*0.85
                  else ("#2ecc71" if x > overall_avg*1.15 else "#3498db")
                  for x in store_stats["Avg_Sales"]]
    ax_s.bar(store_stats["Store"], store_stats["Avg_Sales"], color=colors_bar)
    ax_s.axhline(overall_avg, color="orange", linestyle="--",
                 linewidth=1.5, label=f"Overall Avg: {int(overall_avg):,}")
    ax_s.set_xlabel("Store"); ax_s.set_ylabel("Avg Weekly Sales")
    ax_s.set_title("Store-by-Store Performance"); ax_s.legend(); ax_s.grid(axis="y", alpha=0.3)
    st.pyplot(fig_s); plt.close(fig_s)
    st.caption("🔴 Red = Low Sales (>15% below avg) | 🔵 Blue = Normal | 🟢 Green = High Sales (>15% above avg)")
    st.info("💡 Tip: Select **Individual Store** in the sidebar to run a dedicated forecast for any store above.")
    st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:13px;'>
Built with ❤️ using Streamlit | AI Sales Forecasting System
</div>
""", unsafe_allow_html=True)