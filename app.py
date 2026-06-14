"""
=============================================================================
app.py  –  NASA MDP Software Defect Prediction Dashboard  (v2 – Production)
=============================================================================
Six-tab production dashboard with:

  Tab 0 – Executive Overview         High-level KPIs + donut chart
  Tab 1 – Code Metrics Deep-Dive     Sidebar-filtered scatter + histogram
  Tab 2 – ML Model Insights          Confusion matrix + feature importance
                                     + 🔄 Trigger Retraining button
  Tab 3 – Interactive Risk Predictor Real-time defect probability form
  Tab 4 – Telemetry & API            Live DB table + manual log entry form
  Tab 5 – 🧠 AI Assistant            Gemini chat with 3 tool-calling agents

Navigation is controlled via st.session_state.active_tab so the LLM agent
can programmatically switch tabs by setting a state_action.

Run
---
    streamlit run app.py

New dependencies (see requirements.txt)
---------------------------------------
    fastapi, uvicorn, httpx, google-generativeai, tabulate
=============================================================================
"""

import os
import sys
import time
import warnings
import threading
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (MUST be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeSentinel AI | NASA MDP Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:      #080c14;
    --bg-card:      rgba(15,22,38,0.85);
    --bg-glass:     rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.08);
    --accent-blue:  #4f8ef7;
    --accent-teal:  #22d3c5;
    --accent-amber: #f59e0b;
    --accent-red:   #ef4444;
    --accent-green: #10b981;
    --text-primary: #e8edf5;
    --text-muted:   #6b7a96;
}

.stApp {
    background: linear-gradient(135deg,#0d1b3e 0%,#0f2952 50%,#071428 100%);
    font-family:'Inter',sans-serif;
    color:var(--text-primary);
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem 3rem;}

[data-testid="stSidebar"]{
    background:rgba(8,15,30,0.95)!important;
    border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{color:var(--text-primary)!important;}

/* ── Custom Navigation Bar ── */
.nav-bar{
    display:flex;gap:4px;
    background:rgba(255,255,255,0.03);
    padding:6px;border-radius:14px;
    border:1px solid var(--border);
    margin-bottom:1.5rem;
}
.nav-btn{
    flex:1;text-align:center;padding:8px 4px;
    border-radius:10px;font-size:0.78rem;font-weight:500;
    cursor:pointer;border:none;
    background:transparent;color:var(--text-muted);
    transition:all 0.2s ease;
}
.nav-btn.active{
    background:linear-gradient(135deg,var(--accent-blue),#3b6fd4)!important;
    color:white!important;
    box-shadow:0 4px 20px rgba(79,142,247,0.35);
}

/* ── Metric Cards ── */
[data-testid="stMetric"]{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:16px;padding:1.2rem 1.5rem;
    backdrop-filter:blur(20px);
    transition:transform 0.2s ease,box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover{
    transform:translateY(-3px);
    box-shadow:0 12px 40px rgba(79,142,247,0.15);
}
[data-testid="stMetricLabel"]{
    color:var(--text-muted)!important;font-size:0.78rem!important;
    font-weight:600!important;letter-spacing:0.06em!important;
    text-transform:uppercase!important;
}
[data-testid="stMetricValue"]{
    color:var(--text-primary)!important;font-size:2rem!important;
    font-weight:700!important;
}

.section-heading{
    font-size:1.4rem;font-weight:700;color:var(--text-primary);
    margin-bottom:1rem;padding-bottom:0.6rem;
    border-bottom:1px solid var(--border);
}
.hero-banner{
    background:linear-gradient(135deg,rgba(79,142,247,0.18) 0%,
        rgba(34,211,197,0.10) 50%,rgba(79,142,247,0.05) 100%);
    border:1px solid rgba(79,142,247,0.25);border-radius:20px;
    padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;
}
.hero-title{
    font-size:2.2rem;font-weight:800;
    background:linear-gradient(135deg,#ffffff 0%,#22d3c5 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin:0 0 0.4rem 0;
}
.hero-subtitle{color:var(--text-muted);font-size:0.95rem;margin:0;}

.risk-high{
    background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(239,68,68,0.05));
    border:1px solid rgba(239,68,68,0.4);border-radius:14px;
    padding:1.2rem 1.5rem;margin-top:1rem;
}
.risk-low{
    background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(16,185,129,0.05));
    border:1px solid rgba(16,185,129,0.4);border-radius:14px;
    padding:1.2rem 1.5rem;margin-top:1rem;
}
.risk-title{font-size:1.15rem;font-weight:700;margin-bottom:0.4rem;}
.risk-text{font-size:0.9rem;color:var(--text-muted);}

.stButton>button{
    background:linear-gradient(135deg,var(--accent-blue),#3b6fd4)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    padding:0.7rem 2rem!important;font-weight:600!important;
    font-size:1rem!important;letter-spacing:0.03em!important;
    transition:all 0.2s ease!important;
    box-shadow:0 4px 20px rgba(79,142,247,0.35)!important;width:100%;
}
.stButton>button:hover{
    transform:translateY(-2px)!important;
    box-shadow:0 8px 30px rgba(79,142,247,0.5)!important;
}

/* Chat bubbles */
.chat-user{
    background:rgba(79,142,247,0.12);border:1px solid rgba(79,142,247,0.2);
    border-radius:14px 14px 4px 14px;padding:0.8rem 1.2rem;
    margin:0.5rem 0;text-align:right;
}
.chat-ai{
    background:rgba(34,211,197,0.08);border:1px solid rgba(34,211,197,0.2);
    border-radius:14px 14px 14px 4px;padding:0.8rem 1.2rem;margin:0.5rem 0;
}
.tool-badge{
    display:inline-block;background:rgba(245,158,11,0.15);
    border:1px solid rgba(245,158,11,0.3);border-radius:20px;
    padding:0.2rem 0.7rem;font-size:0.75rem;color:#f59e0b;margin:0.2rem;
}

::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(79,142,247,0.3);border-radius:99px;}

/* Floating Action Button (FAB) */
.floating-chat-marker {
    display: none;
}
div[data-testid="element-container"]:has(.floating-chat-marker) + div[data-testid="element-container"] {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    z-index: 999999 !important;
}
div[data-testid="element-container"]:has(.floating-chat-marker) + div[data-testid="element-container"] button {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #4f8ef7, #22d3c5) !important;
    color: white !important;
    font-size: 26px !important;
    box-shadow: 0 6px 20px rgba(79, 142, 247, 0.45) !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="element-container"]:has(.floating-chat-marker) + div[data-testid="element-container"] button:hover {
    transform: scale(1.1) rotate(5deg) !important;
    box-shadow: 0 8px 30px rgba(34, 211, 197, 0.65) !important;
}

/* Floating Chat Panel Container */
.floating-chat-panel-marker {
    display: none;
}
div[data-testid="element-container"]:has(.floating-chat-panel-marker) + div[data-testid="element-container"] {
    position: fixed !important;
    bottom: 105px !important;
    right: 30px !important;
    width: 440px !important;
    max-width: 90vw !important;
    height: 570px !important;
    max-height: 80vh !important;
    background: rgba(10, 18, 36, 0.98) !important;
    border: 1px solid rgba(79, 142, 247, 0.25) !important;
    border-radius: 20px !important;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.6) !important;
    z-index: 999998 !important;
    padding: 1.2rem !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    backdrop-filter: blur(25px) !important;
    animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly dark theme
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter,sans-serif", color="#e8edf5", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)
COLOR_CLEAN  = "#22d3c5"
COLOR_DEFECT = "#ef4444"
COLOR_BLUE   = "#4f8ef7"
COLOR_AMBER  = "#f59e0b"

# ─────────────────────────────────────────────────────────────────────────────
# Navigation tabs
# ─────────────────────────────────────────────────────────────────────────────
NAV_TABS = [
    "📊 Overview",
    "🔬 Deep-Dive",
    "🤖 ML Insights",
    "⚡ Risk Predictor",
    "🗄️ Telemetry",
]

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "active_tab"    not in st.session_state: st.session_state.active_tab = 0
if "chat_history"  not in st.session_state: st.session_state.chat_history = []
if "agent"         not in st.session_state: st.session_state.agent = None
if "retrain_done"  not in st.session_state: st.session_state.retrain_done = False
if "retrain_metrics" not in st.session_state: st.session_state.retrain_metrics = None

# ─────────────────────────────────────────────────────────────────────────────
# Data & Model Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataframe(data_path: str) -> pd.DataFrame:
    from src.data_pipeline import run_pipeline
    _, _, _, _, _, df_full = run_pipeline(data_path)
    return df_full

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> dict:
    return joblib.load(model_path)

@st.cache_data(show_spinner=False, ttl=30)
def load_telemetry_df() -> pd.DataFrame:
    """Fetch the live DB contents (refreshes every 30 s)."""
    try:
        from src.database import get_dataframe
        return get_dataframe()
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Database Bootstrap (initialise + seed on first run)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _bootstrap_db(arff_path: str):
    """Run once per Streamlit process lifetime."""
    try:
        from src.database import init_db, seed_from_file
        init_db()
        if os.path.isfile(arff_path):
            seed_from_file(arff_path)
    except Exception as exc:
        pass   # Non-fatal – dashboard still works without DB

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🛰️</div>
        <div style='font-weight:700;font-size:1.1rem;color:#e8edf5;margin-top:0.5rem;'>
            CodeSentinel AI
        </div>
        <div style='color:#6b7a96;font-size:0.78rem;margin-top:0.2rem;'>
            NASA MDP Defect Platform
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08);margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")
    csv_path = st.text_input(
        "Dataset Path (.arff / .csv)",
        value="data/KC1.arff",
        help="Relative or absolute path to the NASA MDP ARFF or CSV file.",
    )
    model_path = st.text_input(
        "Model Artefact Path",
        value="models/defect_model.pkl",
    )

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.08);margin:1rem 0;'>
    <div style='color:#6b7a96;font-size:0.78rem;margin-bottom:0.5rem;'>📌 Tab 2 Filters</div>
    """, unsafe_allow_html=True)

    min_loc   = st.slider("Min. Lines of Code", 0, 2000, 0, 10)
    min_cyclo = st.slider("Min. Cyclomatic Complexity", 0, 100, 0, 1)

    # Gemini API key – use st.secrets if available, else use user's key.
    gemini_key = ""
    try:
        gemini_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        gemini_key = "AIzaSyCjzTMkcL9K23H2_RXp9XVHWYCOOGW4gpw"

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.08);margin:1rem 0;'>
    <div style='color:#6b7a96;font-size:0.75rem;text-align:center;'>
        Built with ❤️ using Streamlit + Plotly<br>
        NASA MDP · SMOTE · Random Forest · Gemini
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Data & Model
# ─────────────────────────────────────────────────────────────────────────────
DATA_LOADED  = False
MODEL_LOADED = False
df           = None
artefact     = None

_bootstrap_db(csv_path)

with st.spinner("🔄 Loading dataset …"):
    try:
        df = load_dataframe(csv_path)
        DATA_LOADED = True
    except FileNotFoundError as exc:
        st.error(f"**Dataset not found:** {exc}")
    except Exception as exc:
        st.error(f"**Error loading dataset:** {exc}")

with st.spinner("🔄 Loading model …"):
    try:
        artefact = load_model(model_path)
        MODEL_LOADED = True
    except FileNotFoundError:
        st.warning(
            f"⚠️ Model not found at `{model_path}`. "
            "Run `python src/train_model.py data/KC1.arff` first."
        )
    except Exception as exc:
        st.warning(f"⚠️ Could not load model: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🛰️ CodeSentinel AI – NASA Software Defect Platform</p>
    <p class="hero-subtitle">
        Real-time telemetry ingestion · Continuous model learning · LLM-powered codebase auditing
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────
nav_cols = st.columns(len(NAV_TABS))
for i, (col, label) in enumerate(zip(nav_cols, NAV_TABS)):
    is_active = st.session_state.active_tab == i
    if col.button(
        label,
        key=f"nav_{i}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.active_tab = i
        st.rerun()

st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.5rem 0 1.5rem;'>",
            unsafe_allow_html=True)

active = st.session_state.active_tab
if active >= len(NAV_TABS):
    active = 0
    st.session_state.active_tab = 0

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if active == 0:
    st.markdown('<p class="section-heading">📊 Executive Overview</p>', unsafe_allow_html=True)

    if not DATA_LOADED or df is None:
        st.info("📁 Provide a valid dataset path in the sidebar.")
    else:
        total_modules = len(df)
        n_defective   = int(df["defective"].sum())
        defect_rate   = 100.0 * n_defective / total_modules

        cyclo_col = next((c for c in ["cyclomatic_complexity","v(g)","vg"] if c in df.columns), None)
        loc_col   = next((c for c in ["loc_total","loc_executable","loc"] if c in df.columns), None)
        avg_cyclo = df[cyclo_col].mean() if cyclo_col else 0.0
        avg_loc   = df[loc_col].mean()   if loc_col   else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🗂️ Total Modules",     f"{total_modules:,}")
        c2.metric("🐛 Defective Modules", f"{n_defective:,}",
                  delta=f"{defect_rate:.1f}% of total", delta_color="inverse")
        c3.metric("📈 Global Defect Rate", f"{defect_rate:.2f}%")
        c4.metric(f"⚙️ Avg {cyclo_col or 'Complexity'}", f"{avg_cyclo:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        lc, rc = st.columns([1, 1])

        with lc:
            st.markdown("#### 🍩 Defect Distribution")
            fig_d = go.Figure(go.Pie(
                labels=["Clean","Defective"],
                values=[total_modules - n_defective, n_defective],
                hole=0.60,
                marker=dict(colors=[COLOR_CLEAN, COLOR_DEFECT],
                            line=dict(color="rgba(0,0,0,0.3)", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
                hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"<b>{defect_rate:.1f}%</b><br><span style='font-size:11px'>Defect Rate</span>",
                x=0.5, y=0.5, font=dict(size=18, color="white"), showarrow=False,
            )
            fig_d.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                                            xanchor="center", x=0.5,
                                            font=dict(color="#e8edf5")), height=380)
            st.plotly_chart(fig_d, use_container_width=True)

        with rc:
            st.markdown("#### 📋 Dataset Summary")
            summary_df = pd.DataFrame({
                "Metric": ["Total Modules","Defective","Clean","Defect Rate",
                           f"Avg {loc_col or 'LOC'}", f"Avg {cyclo_col or 'v(g)'}",
                           "Feature Columns"],
                "Value":  [f"{total_modules:,}", f"{n_defective:,}",
                           f"{total_modules - n_defective:,}", f"{defect_rate:.2f}%",
                           f"{avg_loc:.1f}", f"{avg_cyclo:.2f}",
                           f"{df.select_dtypes(include='number').shape[1] - 1}"],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True, height=280)
            st.markdown("<br>", unsafe_allow_html=True)
            if defect_rate > 20:
                st.markdown(f"""<div class='risk-high'><div class='risk-title' style='color:#ef4444;'>
                    ⚠️ Elevated Defect Density</div><div class='risk-text'>
                    {defect_rate:.1f}% exceeds the 20% threshold. Prioritise code review.
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='risk-low'><div class='risk-title' style='color:#10b981;'>
                    ✅ Acceptable Defect Density</div><div class='risk-text'>
                    {defect_rate:.1f}% is within acceptable bounds.
                    </div></div>""", unsafe_allow_html=True)

        if loc_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📦 Defect Rate by LOC Quartile")
            df_q = df.copy()
            df_q["loc_quartile"] = pd.qcut(df_q[loc_col], q=4,
                labels=["Q1 (Smallest)","Q2","Q3","Q4 (Largest)"], duplicates="drop")
            qs = (df_q.groupby("loc_quartile", observed=True)["defective"]
                  .agg(["count","sum"])
                  .rename(columns={"count":"Total","sum":"Defective"})
                  .reset_index())
            qs["Defect Rate (%)"] = 100.0 * qs["Defective"] / qs["Total"]
            fig_bar = go.Figure(go.Bar(
                x=qs["loc_quartile"].astype(str), y=qs["Defect Rate (%)"],
                marker=dict(color=qs["Defect Rate (%)"],
                            colorscale=[[0,COLOR_CLEAN],[0.5,COLOR_AMBER],[1,COLOR_DEFECT]],
                            showscale=True, colorbar=dict(title="Defect %")),
                text=qs["Defect Rate (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside", textfont=dict(color="white"),
                hovertemplate="Quartile: %{x}<br>Defect Rate: %{y:.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=350,
                                  xaxis_title="LOC Quartile", yaxis_title="Defect Rate (%)")
            st.plotly_chart(fig_bar, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CODE METRICS DEEP-DIVE
# ═════════════════════════════════════════════════════════════════════════════
elif active == 1:
    st.markdown('<p class="section-heading">🔬 Code Metrics Deep-Dive</p>', unsafe_allow_html=True)

    if not DATA_LOADED or df is None:
        st.info("📁 Provide a valid dataset path in the sidebar.")
    else:
        cyclo_col_t2 = next((c for c in ["cyclomatic_complexity","v(g)","vg"] if c in df.columns), None)
        vol_col_t2   = next((c for c in ["halstead_volume","v","volume"]       if c in df.columns), None)
        loc_col_t2   = next((c for c in ["loc_total","loc_executable","loc"]   if c in df.columns), None)

        missing = [n for n, c in [("cyclomatic_complexity", cyclo_col_t2), ("halstead_volume", vol_col_t2)] if c is None]
        if missing:
            st.warning(f"⚠️ Could not find columns: {missing}")
        else:
            df_filtered = df.copy()
            if loc_col_t2 and min_loc   > 0: df_filtered = df_filtered[df_filtered[loc_col_t2]   >= min_loc]
            if cyclo_col_t2 and min_cyclo > 0: df_filtered = df_filtered[df_filtered[cyclo_col_t2] >= min_cyclo]
            n_total, n_filtered = len(df), len(df_filtered)

            st.markdown(f"""
            <div style='display:flex;gap:1.5rem;margin-bottom:1.2rem;'>
                <div style='background:rgba(79,142,247,0.12);border:1px solid rgba(79,142,247,0.25);
                            border-radius:10px;padding:0.6rem 1.2rem;font-size:0.85rem;'>
                    🔍 Showing <b>{n_filtered:,}</b> of <b>{n_total:,}</b> modules
                </div>
                <div style='background:rgba(255,255,255,0.04);border:1px solid var(--border);
                            border-radius:10px;padding:0.6rem 1.2rem;font-size:0.85rem;color:#6b7a96;'>
                    Filters: LOC ≥ {min_loc} | v(g) ≥ {min_cyclo}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### 🔵 Cyclomatic Complexity vs Halstead Volume")
            st.caption("Upper-right quadrant = highest-risk modules for code review.")

            df_plot = df_filtered.copy()
            df_plot["defect_label"] = df_plot["defective"].map({1:"🔴 Defective",0:"🟢 Clean"})

            fig_sc = px.scatter(
                df_plot, x=cyclo_col_t2, y=vol_col_t2,
                color="defect_label",
                color_discrete_map={"🔴 Defective":COLOR_DEFECT,"🟢 Clean":COLOR_CLEAN},
                opacity=0.75,
                hover_data={
                    cyclo_col_t2:":.2f", vol_col_t2:":.2f", "defect_label":True,
                    **({loc_col_t2:":.0f"} if loc_col_t2 else {}),
                },
                labels={cyclo_col_t2:"McCabe Cyclomatic Complexity",
                        vol_col_t2:"Halstead Volume"},
                title=f"Cyclomatic Complexity vs. Halstead Volume ({n_filtered:,} modules)",
            )
            fig_sc.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)")))
            if n_filtered > 0:
                fig_sc.add_vline(x=df_plot[cyclo_col_t2].median(), line_dash="dash",
                                 line_color="rgba(245,158,11,0.4)",
                                 annotation_text="Median v(g)", annotation_font_color="#f59e0b")
                fig_sc.add_hline(y=df_plot[vol_col_t2].median(), line_dash="dash",
                                 line_color="rgba(245,158,11,0.4)",
                                 annotation_text="Median Volume", annotation_font_color="#f59e0b")
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=520,
                                 legend=dict(title="Defect Status", bgcolor="rgba(0,0,0,0.3)"))
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("#### 📈 Complexity Distribution by Defect Status")
            fig_hist = px.histogram(
                df_plot.copy(), x=cyclo_col_t2, color="defect_label",
                color_discrete_map={"🔴 Defective":COLOR_DEFECT,"🟢 Clean":COLOR_CLEAN},
                barmode="overlay", opacity=0.7, nbins=40,
                labels={cyclo_col_t2:"Cyclomatic Complexity"},
                title="Distribution of Cyclomatic Complexity",
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL INSIGHTS + RETRAINING
# ═════════════════════════════════════════════════════════════════════════════
elif active == 2:
    st.markdown('<p class="section-heading">🤖 ML Model Insights</p>', unsafe_allow_html=True)

    if not MODEL_LOADED or artefact is None:
        st.warning("🔧 Model not loaded. Run `python src/train_model.py data/KC1.arff` first.")
    else:
        # ── Use retraining metrics if available ───────────────────────────
        display_metrics = st.session_state.retrain_metrics or artefact["metrics"]
        feat_names = display_metrics["feature_names"]
        feat_imps  = display_metrics["feature_importances"]
        threshold  = display_metrics["threshold"]
        cm_data    = np.array(display_metrics["confusion_matrix"])

        # ── Source badge ──────────────────────────────────────────────────
        if st.session_state.retrain_metrics:
            st.success("✅ Showing metrics from the most recent **retraining cycle**.")
        else:
            st.info("ℹ️ Showing metrics from the **originally trained model**.")

        # ── KPI Cards ─────────────────────────────────────────────────────
        st.markdown(f"#### 🏆 Model Performance  (threshold = {threshold:.2f})")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("🎯 Recall",    f"{display_metrics['recall']*100:.2f}%",
                   help="% of real bugs correctly flagged – PRIMARY metric.")
        mc2.metric("⚖️ F1-Score",  f"{display_metrics['f1']*100:.2f}%")
        mc3.metric("🔍 Precision", f"{display_metrics['precision']*100:.2f}%")
        mc4.metric("📐 ROC-AUC",   f"{display_metrics['roc_auc']*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────
        chart1, chart2 = st.columns([1, 1.4])

        with chart1:
            st.markdown("#### 🔲 Confusion Matrix")
            tn, fp, fn, tp = cm_data.ravel()
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp]],
                x=["Predicted: Clean","Predicted: Defective"],
                y=["Actual: Clean","Actual: Defective"],
                text=[[f"TN<br>{tn}",f"FP<br>{fp}"],[f"FN<br>{fn}",f"TP<br>{tp}"]],
                texttemplate="%{text}", textfont=dict(size=16, color="white"),
                colorscale=[[0,"rgba(15,22,38,0.9)"],[0.35,"rgba(34,211,197,0.35)"],
                            [1,"rgba(34,211,197,0.85)"]],
                showscale=False,
                hovertemplate="%{y} | %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.add_shape(type="rect", x0=-0.5, y0=0.5, x1=0.5, y1=1.5,
                             line=dict(color=COLOR_DEFECT, width=2.5),
                             fillcolor="rgba(0,0,0,0)")
            fig_cm.update_layout(**{**PLOTLY_LAYOUT,
                "xaxis": dict(side="bottom", gridcolor="rgba(0,0,0,0)"),
                "yaxis": dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                "height": 380,
            })
            st.plotly_chart(fig_cm, use_container_width=True)

        with chart2:
            st.markdown("#### 📊 Feature Importance Ranking")
            top_n    = min(15, len(feat_names))
            t_names  = feat_names[:top_n][::-1]
            t_imps   = feat_imps[:top_n][::-1]
            norm_imp = np.array(t_imps)
            if norm_imp.max() > 0: norm_imp = norm_imp / norm_imp.max()
            fig_fi = go.Figure(go.Bar(
                x=t_imps, y=t_names, orientation="h",
                marker=dict(color=norm_imp,
                            colorscale=[[0,"rgba(79,142,247,0.4)"],[1,"#22d3c5"]],
                            showscale=False),
                text=[f"{v:.3f}" for v in t_imps], textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.7)", size=10),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_fi.update_layout(**PLOTLY_LAYOUT, height=420,
                                 xaxis_title="Importance Score", bargap=0.3)
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Model Architecture Info ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ℹ️ Model Architecture")
        i1, i2, i3 = st.columns(3)
        i1.info("**Algorithm:** Random Forest\n\n300 trees (or grid-searched best).")
        i2.info("**Imbalance:** SMOTE + `balanced_subsample`\n\nPrevents majority-class bias.")
        i3.info(f"**Threshold:** {threshold:.2f}\n\nTuned to maximise Recall.")

        if display_metrics.get("best_params"):
            st.markdown("#### 🎛️ Best Hyperparameters (from RandomizedSearchCV)")
            st.json(display_metrics["best_params"])

        # ── RETRAINING SECTION ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 🔄 Continuous Learning – Trigger Retraining")
        st.caption(
            "Pulls all records from the live SQLite telemetry database, "
            "runs RandomizedSearchCV hyperparameter optimisation, and overwrites the production model."
        )

        rt_col1, rt_col2 = st.columns([1, 2])
        with rt_col1:
            retrain_btn = st.button("🔄 Trigger System Retraining Cycle", key="retrain_btn")

        with rt_col2:
            if st.session_state.retrain_done and st.session_state.retrain_metrics:
                m = st.session_state.retrain_metrics
                st.success(
                    f"✅ Last cycle complete — "
                    f"Recall: **{m['recall']*100:.2f}%** | "
                    f"F1: **{m['f1']*100:.2f}%** | "
                    f"ROC-AUC: **{m['roc_auc']*100:.2f}%**"
                )

        if retrain_btn:
            with st.spinner("⚙️ Running retraining cycle (SMOTE + RandomizedSearchCV) …"):
                try:
                    from src.train_model import retrain_pipeline
                    new_metrics = retrain_pipeline()
                    st.session_state.retrain_metrics = new_metrics
                    st.session_state.retrain_done    = True
                    # Clear the model cache so it reloads the new .pkl
                    load_model.clear()
                    st.success(
                        f"✅ Retraining complete! "
                        f"Recall: **{new_metrics['recall']*100:.2f}%** | "
                        f"F1: **{new_metrics['f1']*100:.2f}%**"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Retraining failed: {exc}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — INTERACTIVE RISK PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif active == 3:
    st.markdown('<p class="section-heading">⚡ Interactive Code Risk Predictor</p>', unsafe_allow_html=True)

    if not MODEL_LOADED or artefact is None:
        st.warning("🔧 Model not loaded. Train the model first.")
    else:
        model_clf    = artefact["model"]
        threshold    = artefact["threshold"]
        feature_list = artefact["feature_names"]

        st.markdown("""
        <div style='background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);
                    border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.5rem;
                    font-size:0.88rem;color:#94a3b8;'>
            🧪 Enter static code metrics below. The trained Random Forest computes
            the real-time defect probability for this module.
        </div>""", unsafe_allow_html=True)

        DISPLAY_NAMES = {
            "loc_blank":"Blank Lines","branch_count":"Branch Count",
            "loc_code_and_comment":"Code+Comment Lines","loc_comments":"Comment Lines",
            "cyclomatic_complexity":"Cyclomatic Complexity v(g)",
            "design_complexity":"Design Complexity iv(g)",
            "essential_complexity":"Essential Complexity ev(g)",
            "loc_executable":"Executable LOC","halstead_content":"Halstead Vocabulary",
            "halstead_difficulty":"Halstead Difficulty",
            "halstead_effort":"Halstead Effort","halstead_error_est":"Halstead Bug Est.",
            "halstead_length":"Halstead Length","halstead_level":"Halstead Level",
            "halstead_prog_time":"Prog. Time (s)","halstead_volume":"Halstead Volume",
            "num_operands":"Total Operands","num_operators":"Total Operators",
            "num_unique_operands":"Unique Operands","num_unique_operators":"Unique Operators",
            "loc_total":"Total LOC",
        }
        DEFAULTS = {
            "loc_blank":2,"branch_count":5,"loc_code_and_comment":0,"loc_comments":2,
            "cyclomatic_complexity":4,"design_complexity":3,"essential_complexity":3,
            "loc_executable":25,"halstead_content":18.5,"halstead_difficulty":8.5,
            "halstead_effort":3250,"halstead_error_est":0.09,"halstead_length":55,
            "halstead_level":0.12,"halstead_prog_time":180,"halstead_volume":210,
            "num_operands":18,"num_operators":37,"num_unique_operands":12,
            "num_unique_operators":10,"loc_total":28,
        }

        st.markdown("#### 📥 Module Metrics Input")
        user_values = {}
        chunks = [feature_list[i:i+3] for i in range(0, len(feature_list), 3)]
        for chunk in chunks:
            row = st.columns(3)
            for col, feat in zip(row, chunk):
                label   = DISPLAY_NAMES.get(feat, feat)
                default = float(DEFAULTS.get(feat, 10.0))
                with col:
                    user_values[feat] = st.number_input(
                        label=label, min_value=0.0, value=default,
                        step=1.0, format="%.2f", key=f"pred_{feat}",
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Analyze Component Risk", key="predict_button")

        if predict_btn:
            vec   = np.array([[user_values[f] for f in feature_list]], dtype=np.float32)
            proba = float(model_clf.predict_proba(vec)[0][1])
            risk  = proba * 100.0
            high  = proba >= threshold

            st.markdown("---")
            st.markdown("#### 📊 Risk Assessment Result")
            bar_col, det_col = st.columns([1.5, 1])

            with bar_col:
                st.markdown("<div style='font-size:0.85rem;color:#6b7a96;margin-bottom:0.4rem;'>Bug Risk Probability</div>",
                            unsafe_allow_html=True)
                st.progress(min(int(risk), 100), text=f"**{risk:.1f}%** defect probability")
                risk_color = COLOR_DEFECT if high else COLOR_CLEAN
                st.markdown(f"""
                <div style='font-size:3.5rem;font-weight:800;color:{risk_color};
                            margin-top:0.5rem;font-family:JetBrains Mono,monospace;'>
                    {risk:.1f}%
                </div>
                <div style='color:#6b7a96;font-size:0.85rem;'>
                    Defect Probability | Threshold: {threshold:.2f}
                </div>""", unsafe_allow_html=True)

            with det_col:
                if high:
                    st.markdown(f"""<div class='risk-high'>
                        <div class='risk-title' style='color:#ef4444;'>🚨 HIGH RISK — Refactor Required</div>
                        <div class='risk-text'>
                            {risk:.1f}% probability exceeds the {threshold*100:.0f}% safety threshold.<br><br>
                            <b>Actions:</b>
                            <ul><li>Reduce cyclomatic complexity</li>
                            <li>Add unit test coverage</li>
                            <li>Conduct peer code review</li></ul>
                        </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='risk-low'>
                        <div class='risk-title' style='color:#10b981;'>✅ LOW RISK — Safe to Deploy</div>
                        <div class='risk-text'>
                            {risk:.1f}% is below the {threshold*100:.0f}% safety threshold.<br><br>
                            Module passes quality gate. Standard monitoring applies.
                        </div></div>""", unsafe_allow_html=True)

            # ── Optionally log this prediction to the DB ──────────────────
            if st.checkbox("📝 Log this prediction to the telemetry database", key="log_pred"):
                try:
                    from src.database import log_entry
                    record = {**user_values, "defective": 1 if high else 0}
                    rid = log_entry(record, predicted_risk=proba, source="manual")
                    st.success(f"✅ Logged to DB (rowid={rid})")
                    load_telemetry_df.clear()
                except Exception as exc:
                    st.error(f"❌ DB logging failed: {exc}")

            # ── Radar chart ────────────────────────────────────────────────
            if DATA_LOADED and df is not None:
                radar_feats = [f for f in feature_list if f in df.columns][:8]
                df_max   = df[radar_feats].quantile(0.95)
                r_norm   = [min(user_values[f]/max(df_max[f],1e-6),2.0) for f in radar_feats]
                m_norm   = [min(df[f].median()/max(df_max[f],1e-6),2.0) for f in radar_feats]
                fig_rad  = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(
                    r=r_norm+[r_norm[0]], theta=radar_feats+[radar_feats[0]],
                    fill="toself", name="This Module",
                    fillcolor=f"rgba(239,68,68,0.15)" if high else "rgba(34,211,197,0.15)",
                    line=dict(color=COLOR_DEFECT if high else COLOR_CLEAN, width=2),
                ))
                fig_rad.add_trace(go.Scatterpolar(
                    r=m_norm+[m_norm[0]], theta=radar_feats+[radar_feats[0]],
                    fill="toself", name="Dataset Median",
                    fillcolor="rgba(79,142,247,0.08)",
                    line=dict(color=COLOR_BLUE, width=1.5, dash="dash"),
                ))
                fig_rad.update_layout(**PLOTLY_LAYOUT,
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True, color="#6b7a96",
                                               gridcolor="rgba(255,255,255,0.08)"),
                               angularaxis=dict(color="#e8edf5")),
                    height=420)
                st.markdown("<br>#### 🕸️ Module Metric Radar Profile")
                st.plotly_chart(fig_rad, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TELEMETRY & API
# ═════════════════════════════════════════════════════════════════════════════
elif active == 4:
    st.markdown('<p class="section-heading">🗄️ Telemetry & API Layer</p>', unsafe_allow_html=True)

    # ── Live DB Stats ──────────────────────────────────────────────────────
    st.markdown("#### 📡 Live Database Overview")
    try:
        from src.database import get_stats, init_db
        init_db()
        stats = get_stats()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📦 Total Records",  f"{stats.get('total_rows',0):,}")
        s2.metric("🐛 Defective",      f"{stats.get('defective_count',0):,}")
        s3.metric("📈 Defect Rate",    f"{stats.get('defect_rate_pct',0):.2f}%")
        s4.metric("🕐 Last Logged",    str(stats.get("last_logged_at","—"))[:19])
    except Exception as exc:
        st.warning(f"Database not yet initialised: {exc}")

    # ── Live Telemetry Table ───────────────────────────────────────────────
    st.markdown("<br>#### 📋 Engineering Telemetry Table")
    tel_df = load_telemetry_df()
    if tel_df.empty:
        st.info("No records yet. Submit a log entry below or run `python src/api.py`.")
    else:
        # Show most recent 200 rows, newest first.
        display_cols = ["id","source","predicted_risk","logged_at",
                        "cyclomatic_complexity","halstead_volume","loc_total","defective"]
        show_cols = [c for c in display_cols if c in tel_df.columns]
        st.dataframe(
            tel_df[show_cols].sort_values("id", ascending=False).head(200),
            use_container_width=True, hide_index=True, height=300,
        )
        if st.button("🔄 Refresh Table", key="refresh_tel"):
            load_telemetry_df.clear()
            st.rerun()

    # ── Manual Log Entry Form ──────────────────────────────────────────────
    st.markdown("<br>---")
    st.markdown("#### ✍️ Manual Log Entry")
    st.caption("Submit a module's metrics directly to the telemetry database.")

    with st.form("manual_log_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        cyc  = fc1.number_input("Cyclomatic Complexity", 1.0, 200.0, 4.0, 1.0)
        vol  = fc2.number_input("Halstead Volume",       0.0, 100000.0, 200.0, 10.0)
        loc  = fc3.number_input("Total LOC",             1.0, 5000.0, 30.0, 1.0)
        fc4, fc5, fc6 = st.columns(3)
        diff = fc4.number_input("Halstead Difficulty",   0.0, 200.0, 8.0, 1.0)
        eff  = fc5.number_input("Halstead Effort",       0.0, 500000.0, 3000.0, 100.0)
        br   = fc6.number_input("Branch Count",          0.0, 200.0, 5.0, 1.0)
        def_gt = st.selectbox("Known Defective?", [0, 1], index=0,
                              format_func=lambda x: "0 – Clean" if x==0 else "1 – Defective")
        src    = st.selectbox("Source Tag", ["manual","ci_hook","api"])
        submitted = st.form_submit_button("📥 Log Entry to Database")

    if submitted:
        try:
            from src.database import log_entry, FEATURE_COLS
            # Build a full record with defaults for missing cols.
            record = {c: 0.0 for c in FEATURE_COLS}
            record.update({
                "cyclomatic_complexity": cyc, "halstead_volume": vol,
                "loc_total": loc, "halstead_difficulty": diff,
                "halstead_effort": eff, "branch_count": br,
                "defective": def_gt,
            })
            proba = None
            if MODEL_LOADED and artefact:
                feat_order = artefact["feature_names"]
                vec  = np.array([[record.get(f,0.0) for f in feat_order]], dtype=np.float32)
                proba = float(artefact["model"].predict_proba(vec)[0][1])
            rid = log_entry(record, predicted_risk=proba, source=src)
            st.success(
                f"✅ Logged (rowid={rid}) | "
                f"Risk: {proba*100:.1f}%" if proba is not None else f"✅ Logged (rowid={rid})"
            )
            load_telemetry_df.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Failed to log: {exc}")

    # ── API Documentation ─────────────────────────────────────────────────
    st.markdown("<br>---")
    st.markdown("#### 🌐 REST API Reference")
    st.info(
        "Run the FastAPI server locally:  \n"
        "```bash\nuvicorn src.api:app --reload --port 8000\n```\n"
        "Then POST module metrics to `http://localhost:8000/log-telemetry/`"
    )
    with st.expander("📋 Example cURL Request"):
        st.code("""curl -X POST http://localhost:8000/log-telemetry/ \\
  -H "Content-Type: application/json" \\
  -d '{
    "loc_blank":2,"branch_count":5,"loc_code_and_comment":0,"loc_comments":2,
    "cyclomatic_complexity":4,"design_complexity":3,"essential_complexity":3,
    "loc_executable":25,"halstead_content":18.5,"halstead_difficulty":8.5,
    "halstead_effort":3250,"halstead_error_est":0.09,"halstead_length":55,
    "halstead_level":0.12,"halstead_prog_time":180,"halstead_volume":210,
    "num_operands":18,"num_operators":37,"num_unique_operands":12,
    "num_unique_operators":10,"loc_total":28,"defective":0,"source":"ci_hook"
  }'""", language="bash")
    with st.expander("📤 Example Response"):
        st.code("""{
  "status": "logged",
  "rowid": 2110,
  "defect_probability": 0.0342,
  "risk_level": "LOW",
  "threshold": 0.07,
  "message": "✅ LOW RISK – module passes the quality gate."
}""", language="json")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.06);margin-top:3rem;'>
<div style='text-align:center;color:#374151;font-size:0.78rem;padding-bottom:1rem;'>
    🛰️ CodeSentinel AI · NASA MDP Software Defect Platform ·
    Random Forest + SMOTE + Gemini · Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FLOATING AI ASSISTANT CHAT PANEL & FAB
# ─────────────────────────────────────────────────────────────────────────────
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# Render the floating toggle button
st.markdown('<div class="floating-chat-marker"></div>', unsafe_allow_html=True)
chat_label = "❌" if st.session_state.show_chat else "💬"
if st.button(chat_label, key="fab_chat_toggle"):
    st.session_state.show_chat = not st.session_state.show_chat
    st.rerun()

# If chat is open, render the floating chat panel
if st.session_state.show_chat:
    st.markdown('<div class="floating-chat-panel-marker"></div>', unsafe_allow_html=True)
    with st.container():
        # Chat panel header
        st.markdown("""
        <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:0.6rem;'>
            <div style='font-size:1.6rem;'>🧠</div>
            <div>
                <div style='font-weight:700;font-size:0.95rem;color:#e8edf5;line-height:1.2;'>CodeSentinel AI Assistant</div>
                <div style='font-size:0.75rem;color:#6b7a96;'>Contextual Audit & Dashboard Control</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check API key
        if not gemini_key:
            st.warning("⚠️ No Gemini API key loaded.")
        else:
            # Load model metrics and df for the agent
            model_metrics_for_agent = artefact["metrics"] if MODEL_LOADED and artefact else None
            
            # Initialize agent
            agent_hash = hash((gemini_key, id(df), id(model_metrics_for_agent)))
            if (st.session_state.agent is None or
                    getattr(st.session_state.agent, "_hash", None) != agent_hash):
                try:
                    from src.llm_agent import DefectAnalysisAgent
                    agent = DefectAnalysisAgent(
                        api_key=gemini_key,
                        df=df,
                        model_metrics=model_metrics_for_agent,
                    )
                    agent._hash = agent_hash
                    st.session_state.agent = agent
                except Exception as exc:
                    st.error(f"❌ Could not initialise agent: {exc}")
                    agent = None
            else:
                agent = st.session_state.agent

            if agent:
                # Clear chat button
                if st.button("🗑️ Clear Chat", key="clear_chat_fab"):
                    st.session_state.chat_history = []
                    st.rerun()

                # Chat scroll area (using st.container with height)
                chat_history_container = st.container(height=320)
                with chat_history_container:
                    if not st.session_state.chat_history:
                        st.markdown("""
                        <div style='color:#6b7a96;font-size:0.8rem;text-align:center;margin-top:2rem;'>
                            Ask me about the dataset, models, query SQL, or navigate tabs.
                        </div>
                        """, unsafe_allow_html=True)
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.markdown(f"""
                            <div class="chat-user">
                                <b>You</b><br>{msg["content"]}
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="chat-ai">
                                <b>🧠 CodeSentinel</b><br>{msg["content"]}
                            </div>""", unsafe_allow_html=True)
                            if msg.get("tool_calls"):
                                tools_html = " ".join(
                                    f'<span class="tool-badge">🔧 {t}</span>'
                                    for t in msg["tool_calls"]
                                )
                                st.markdown(tools_html, unsafe_allow_html=True)

                # Chat input
                user_prompt = st.chat_input(
                    "Ask CodeSentinel ...", key="chat_input_fab"
                )

                if user_prompt:
                    # Append user message.
                    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

                    with st.spinner("🧠 CodeSentinel is thinking …"):
                        try:
                            result = agent.chat(user_prompt)
                        except Exception as exc:
                            result = {"text": f"⚠️ Agent error: {exc}", "tool_calls": [], "state_action": None}

                    # Append assistant message.
                    st.session_state.chat_history.append({
                        "role":       "assistant",
                        "content":    result["text"],
                        "tool_calls": result.get("tool_calls", []),
                    })

                    # ── Process State Action (navigation / slider updates) ────────────
                    action = result.get("state_action")
                    if action and action.get("type") == "navigate":
                        tab_idx = action.get("tab", 0)
                        if 0 <= tab_idx < len(NAV_TABS):
                            st.session_state.active_tab = tab_idx
                            st.toast(f"🧭 Navigating to **{NAV_TABS[tab_idx]}**", icon="🛰️")

                    st.rerun()
