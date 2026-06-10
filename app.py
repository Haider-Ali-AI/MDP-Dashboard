"""
=============================================================================
app.py  –  NASA MDP Software Defect & Bug Prediction Dashboard
=============================================================================
A production-grade Streamlit web application providing:

  Tab 1 – Executive Overview
          High-level KPI cards + interactive donut chart of defect distribution.

  Tab 2 – Code Metrics Deep-Dive
          Sidebar-filtered scatter plot: Cyclomatic Complexity vs Halstead Volume,
          colour-coded by actual defect status.

  Tab 3 – ML Model Insights
          Confusion matrix heatmap + feature importance horizontal bar chart +
          Recall / F1 metric cards.

  Tab 4 – Interactive Code Risk Predictor
          User-facing numeric input form → real-time defect probability from
          the trained model + risk-level feedback.

Run
---
    streamlit run app.py

Dependencies
------------
    streamlit, plotly, pandas, numpy, joblib, scikit-learn, imbalanced-learn
    pip install streamlit plotly pandas numpy joblib scikit-learn imbalanced-learn
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow importing from src/ when running from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA MDP | Defect Prediction System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  –  dark glassmorphism theme with premium typography
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ─────────────────────────────────────────────────── */
    :root {
        --bg-deep:      #080c14;
        --bg-card:      rgba(15, 22, 38, 0.85);
        --bg-glass:     rgba(255, 255, 255, 0.04);
        --border:       rgba(255, 255, 255, 0.08);
        --accent-blue:  #4f8ef7;
        --accent-teal:  #22d3c5;
        --accent-amber: #f59e0b;
        --accent-red:   #ef4444;
        --accent-green: #10b981;
        --text-primary: #e8edf5;
        --text-muted:   #6b7a96;
        --gradient-hero: linear-gradient(135deg, #0d1b3e 0%, #0f2952 50%, #071428 100%);
    }

    /* ── App Background ─────────────────────────────────────────────────── */
    .stApp {
        background: var(--gradient-hero);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* ── Hide default Streamlit chrome ──────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2rem 3rem; }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(8, 15, 30, 0.95) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Tab Styling ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.03);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted) !important;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.875rem;
        border: none !important;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), #3b6fd4) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(79, 142, 247, 0.35);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── Metric Cards ───────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(20px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(79,142,247,0.15);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* ── Section Headings ───────────────────────────────────────────────── */
    .section-heading {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Hero Banner ────────────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg,
            rgba(79,142,247,0.18) 0%,
            rgba(34,211,197,0.10) 50%,
            rgba(79,142,247,0.05) 100%);
        border: 1px solid rgba(79,142,247,0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, var(--accent-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Risk Alert Boxes ────────────────────────────────────────────────── */
    .risk-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .risk-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .risk-text { font-size: 0.9rem; color: var(--text-muted); }

    /* ── Input Widgets ──────────────────────────────────────────────────── */
    [data-testid="stNumberInput"] input,
    [data-testid="stSlider"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }

    /* ── Button ─────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), #3b6fd4) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(79,142,247,0.35) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(79,142,247,0.5) !important;
    }

    /* ── Progress Bar ────────────────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
        border-radius: 99px !important;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(79,142,247,0.3); border-radius: 99px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  –  consistent dark appearance across all charts
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter, sans-serif", color="#e8edf5", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)

COLOR_CLEAN    = "#22d3c5"   # Teal for clean modules.
COLOR_DEFECT   = "#ef4444"   # Red for defective modules.
COLOR_BLUE     = "#4f8ef7"
COLOR_AMBER    = "#f59e0b"


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS  (cached to avoid redundant I/O on every rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the NASA MDP dataset using the data_pipeline module.
    Supports both ARFF and CSV formats. Returns the cleaned full DataFrame.
    """
    from src.data_pipeline import run_pipeline
    _, _, _, _, _, df_full = run_pipeline(data_path)
    return df_full


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> dict:
    """
    Deserialise the joblib model artefact saved by train_model.py.
    Returns the artefact dict: {model, threshold, feature_names,
                                feature_importances, metrics}.
    """
    return joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  –  dataset / model path configuration + filter controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:1rem 0 0.5rem;'>
            <div style='font-size:2.5rem;'>🛰️</div>
            <div style='font-weight:700; font-size:1.1rem; color:#e8edf5; margin-top:0.5rem;'>
                MDP Defect Predictor
            </div>
            <div style='color:#6b7a96; font-size:0.78rem; margin-top:0.2rem;'>
                NASA Metrics Data Program
            </div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Data Configuration")
    csv_path = st.text_input(
        "Dataset Path (.arff or .csv)",
        value="data/NASADefectDataset-master/OriginalData/MDP/KC1.arff",
        help="Relative or absolute path to the NASA MDP ARFF or CSV file.",
    )
    model_path = st.text_input(
        "Model Artefact Path",
        value="models/defect_model.pkl",
        help="Path to the trained defect_model.pkl file.",
    )

    st.markdown(
        """
        <hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'>
        <div style='color:#6b7a96; font-size:0.78rem; margin-bottom:0.5rem;'>
            📌 Tab 2 Filters
        </div>
        """,
        unsafe_allow_html=True,
    )

    min_loc = st.slider(
        "Min. Lines of Code (loc)",
        min_value=0, max_value=2000, value=0, step=10,
        help="Show only modules with ≥ this many lines of code.",
    )
    min_cyclo = st.slider(
        "Min. Cyclomatic Complexity v(g)",
        min_value=0, max_value=100, value=0, step=1,
        help="Show only modules with ≥ this cyclomatic complexity.",
    )

    st.markdown(
        """
        <hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'>
        <div style='color:#6b7a96; font-size:0.75rem; text-align:center;'>
            Built with ❤️ using Streamlit + Plotly<br>
            NASA MDP  |  SMOTE + Random Forest
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  –  with graceful error handling
# ─────────────────────────────────────────────────────────────────────────────

DATA_LOADED  = False
MODEL_LOADED = False
df           = None
artefact     = None

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
            "Run `python src/train_model.py data/KC1.csv` first. "
            "Tab 3 and Tab 4 will be unavailable until the model is trained."
        )
    except Exception as exc:
        st.warning(f"⚠️ Could not load model: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🛰️ NASA Software Defect Prediction</p>
        <p class="hero-subtitle">
            AI-powered static code analysis platform powered by the NASA Metrics Data Program (MDP).
            Identify high-risk software modules before they reach production.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Executive Overview",
    "🔬  Code Metrics Deep-Dive",
    "🤖  ML Model Insights",
    "⚡  Interactive Risk Predictor",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1  –  EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="section-heading">📊 Executive Overview</p>', unsafe_allow_html=True)

    if not DATA_LOADED or df is None:
        st.info("📁 Please provide a valid dataset path in the sidebar to view this tab.")
    else:
        # ── KPI Metrics ──────────────────────────────────────────────────
        total_modules    = len(df)
        n_defective      = int(df["defective"].sum())
        defect_rate      = 100.0 * n_defective / total_modules

        # Average cyclomatic complexity – try ARFF name first, then CSV names.
        cyclo_col = None
        for candidate in ["cyclomatic_complexity", "v(g)", "vg", "mccabe"]:
            if candidate in df.columns:
                cyclo_col = candidate
                break

        avg_cyclo = df[cyclo_col].mean() if cyclo_col else 0.0

        # LOC column – ARFF uses 'loc_total' or 'loc_executable'
        loc_col = None
        for candidate in ["loc_total", "loc_executable", "loc", "lines_of_code", "locode"]:
            if candidate in df.columns:
                loc_col = candidate
                break
        avg_loc = df[loc_col].mean() if loc_col else 0.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🗂️ Total Modules Analyzed", f"{total_modules:,}")
        with col2:
            st.metric("🐛 Defective Modules", f"{n_defective:,}",
                      delta=f"{defect_rate:.1f}% of total",
                      delta_color="inverse")
        with col3:
            st.metric("📈 Global Defect Rate", f"{defect_rate:.2f}%")
        with col4:
            label = cyclo_col.replace("_", " ").title() if cyclo_col else "Cyclomatic Complexity"
            st.metric(f"⚙️ Avg {label}", f"{avg_cyclo:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Donut Chart ───────────────────────────────────────────────────
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("#### 🍩 Defect Distribution")
            donut_labels = ["Clean Modules", "Defective Modules"]
            donut_values = [total_modules - n_defective, n_defective]

            fig_donut = go.Figure(
                go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.60,
                    marker=dict(colors=[COLOR_CLEAN, COLOR_DEFECT],
                                line=dict(color="rgba(0,0,0,0.3)", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=13, color="white"),
                    hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                )
            )
            fig_donut.add_annotation(
                text=f"<b>{defect_rate:.1f}%</b><br><span style='font-size:11px'>Defect Rate</span>",
                x=0.5, y=0.5,
                font=dict(size=18, color="white"),
                showarrow=False,
            )
            fig_donut.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5,
                    font=dict(color="#e8edf5"),
                ),
                height=380,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with right_col:
            st.markdown("#### 📋 Dataset Summary")

            # Build a summary table.
            summary_data = {
                "Metric": [
                    "Total Modules",
                    "Defective",
                    "Clean",
                    "Defect Rate",
                    f"Avg {loc_col}" if loc_col else "Avg LOC",
                    f"Avg {cyclo_col}" if cyclo_col else "Avg Complexity",
                    "Feature Columns",
                ],
                "Value": [
                    f"{total_modules:,}",
                    f"{n_defective:,}",
                    f"{total_modules - n_defective:,}",
                    f"{defect_rate:.2f}%",
                    f"{avg_loc:.1f}" if loc_col else "N/A",
                    f"{avg_cyclo:.2f}" if cyclo_col else "N/A",
                    f"{df.select_dtypes(include='number').shape[1] - 1}",
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True,
                height=280,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if defect_rate > 20:
                st.markdown(
                    f"""<div class='risk-high'>
                    <div class='risk-title' style='color:#ef4444;'>⚠️ Elevated Defect Density</div>
                    <div class='risk-text'>
                        {defect_rate:.1f}% defect rate exceeds the 20% safety threshold.
                        Prioritise code review and refactoring.
                    </div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class='risk-low'>
                    <div class='risk-title' style='color:#10b981;'>✅ Acceptable Defect Density</div>
                    <div class='risk-text'>
                        {defect_rate:.1f}% defect rate is within acceptable bounds.
                        Maintain current quality practices.
                    </div></div>""",
                    unsafe_allow_html=True,
                )

        # ── Additional Bar Chart: Defects by LOC Quartile ─────────────────
        if loc_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📦 Defect Rate by LOC Quartile")
            df_q = df.copy()
            df_q["loc_quartile"] = pd.qcut(
                df_q[loc_col], q=4,
                labels=["Q1 (Smallest)", "Q2", "Q3", "Q4 (Largest)"],
                duplicates="drop",
            )
            quartile_stats = (
                df_q.groupby("loc_quartile", observed=True)["defective"]
                .agg(["count", "sum"])
                .rename(columns={"count": "Total", "sum": "Defective"})
                .reset_index()
            )
            quartile_stats["Defect Rate (%)"] = (
                100.0 * quartile_stats["Defective"] / quartile_stats["Total"]
            )

            fig_bar = go.Figure(
                go.Bar(
                    x=quartile_stats["loc_quartile"].astype(str),
                    y=quartile_stats["Defect Rate (%)"],
                    marker=dict(
                        color=quartile_stats["Defect Rate (%)"],
                        colorscale=[[0, COLOR_CLEAN], [0.5, COLOR_AMBER], [1, COLOR_DEFECT]],
                        showscale=True,
                        colorbar=dict(title="Defect %", tickfont=dict(color="#e8edf5")),
                    ),
                    text=quartile_stats["Defect Rate (%)"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                    textfont=dict(color="white"),
                    hovertemplate=(
                        "Quartile: %{x}<br>Defect Rate: %{y:.2f}%<extra></extra>"
                    ),
                )
            )
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title="Defect Rate Across Lines-of-Code Quartiles",
                xaxis_title="LOC Quartile",
                yaxis_title="Defect Rate (%)",
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2  –  CODE METRICS DEEP-DIVE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p class="section-heading">🔬 Code Metrics Deep-Dive</p>', unsafe_allow_html=True)

    if not DATA_LOADED or df is None:
        st.info("📁 Please provide a valid dataset path in the sidebar to view this tab.")
    else:
        # ── Resolve column names – ARFF names first, then CSV fallbacks ───
        cyclo_col_opts = ["cyclomatic_complexity", "v(g)", "vg"]
        vol_col_opts   = ["halstead_volume", "v", "volume"]
        loc_col_opts   = ["loc_total", "loc_executable", "loc", "locode", "lines_of_code"]

        cyclo_col_t2  = next((c for c in cyclo_col_opts if c in df.columns), None)
        vol_col_t2    = next((c for c in vol_col_opts   if c in df.columns), None)
        loc_col_t2    = next((c for c in loc_col_opts   if c in df.columns), None)

        missing = [n for n, c in [("v(g)", cyclo_col_t2), ("Halstead v", vol_col_t2)] if c is None]
        if missing:
            st.warning(
                f"⚠️ Could not find columns for: {missing}. "
                "Tab 2 requires standard NASA MDP column names."
            )
        else:
            # ── Apply sidebar filters ─────────────────────────────────────
            df_filtered = df.copy()
            if loc_col_t2 and min_loc > 0:
                df_filtered = df_filtered[df_filtered[loc_col_t2] >= min_loc]
            if cyclo_col_t2 and min_cyclo > 0:
                df_filtered = df_filtered[df_filtered[cyclo_col_t2] >= min_cyclo]

            n_total    = len(df)
            n_filtered = len(df_filtered)

            # Filter status banner.
            st.markdown(
                f"""
                <div style='display:flex; gap:1.5rem; margin-bottom:1.2rem;'>
                    <div style='background:rgba(79,142,247,0.12); border:1px solid rgba(79,142,247,0.25);
                                border-radius:10px; padding:0.6rem 1.2rem; font-size:0.85rem;'>
                        🔍  Showing <b>{n_filtered:,}</b> of <b>{n_total:,}</b> modules
                    </div>
                    <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                border-radius:10px; padding:0.6rem 1.2rem; font-size:0.85rem; color:#6b7a96;'>
                        Filters: LOC ≥ {min_loc}  |  v(g) ≥ {min_cyclo}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Scatter Plot ──────────────────────────────────────────────
            st.markdown("#### 🔵 Cyclomatic Complexity vs Halstead Volume")
            st.caption(
                "Each point is a software module. "
                "Modules in the **upper-right quadrant** (high complexity + high volume) "
                "are the highest-risk targets for code review."
            )

            df_filtered = df_filtered.copy()
            df_filtered["defect_label"] = df_filtered["defective"].map(
                {1: "🔴 Defective", 0: "🟢 Clean"}
            )

            fig_scatter = px.scatter(
                df_filtered,
                x=cyclo_col_t2,
                y=vol_col_t2,
                color="defect_label",
                color_discrete_map={"🔴 Defective": COLOR_DEFECT, "🟢 Clean": COLOR_CLEAN},
                opacity=0.75,
                hover_data={
                    cyclo_col_t2:   ":.2f",
                    vol_col_t2:     ":.2f",
                    "defect_label": True,
                    **({loc_col_t2: ":.0f"} if loc_col_t2 and loc_col_t2 in df_filtered.columns else {}),
                },
                labels={
                    cyclo_col_t2: "McCabe Cyclomatic Complexity v(g)",
                    vol_col_t2:   "Halstead Volume",
                },
                title=(
                    f"Cyclomatic Complexity vs. Halstead Volume  "
                    f"({n_filtered:,} modules)"
                ),
            )
            fig_scatter.update_traces(
                marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)"))
            )
            # Quadrant reference lines.
            if n_filtered > 0:
                med_x = df_filtered[cyclo_col_t2].median()
                med_y = df_filtered[vol_col_t2].median()
                fig_scatter.add_vline(
                    x=med_x, line_dash="dash",
                    line_color="rgba(245,158,11,0.4)",
                    annotation_text="Median v(g)",
                    annotation_font_color="#f59e0b",
                )
                fig_scatter.add_hline(
                    y=med_y, line_dash="dash",
                    line_color="rgba(245,158,11,0.4)",
                    annotation_text="Median Volume",
                    annotation_font_color="#f59e0b",
                )

            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                legend=dict(
                    title="Defect Status",
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1,
                ),
                height=520,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # ── Histogram: Complexity Distribution ────────────────────────
            st.markdown("#### 📈 Complexity Distribution by Defect Status")
            fig_hist = px.histogram(
                df_filtered.copy(),
                x=cyclo_col_t2,
                color="defect_label",
                color_discrete_map={"🔴 Defective": COLOR_DEFECT, "🟢 Clean": COLOR_CLEAN},
                barmode="overlay",
                opacity=0.7,
                nbins=40,
                labels={cyclo_col_t2: "Cyclomatic Complexity v(g)"},
                title="Distribution of Cyclomatic Complexity (Clean vs. Defective)",
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3  –  ML MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<p class="section-heading">🤖 Machine Learning Model Insights</p>', unsafe_allow_html=True)

    if not MODEL_LOADED or artefact is None:
        st.warning(
            "🔧 Model not loaded. "
            "Run `python src/train_model.py data/KC1.csv` from the project root, "
            "then refresh this page."
        )
    else:
        metrics    = artefact["metrics"]
        feat_names = artefact["metrics"]["feature_names"]
        feat_imps  = artefact["metrics"]["feature_importances"]
        threshold  = artefact["threshold"]
        cm_data    = np.array(artefact["metrics"]["confusion_matrix"])

        # ── Model Performance Cards ───────────────────────────────────────
        st.markdown("#### 🏆 Optimised Performance Metrics")
        st.caption(
            f"Metrics computed at optimised decision threshold **{threshold:.2f}** "
            "(tuned to maximise Recall)."
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("🎯 Recall (Sensitivity)", f"{metrics['recall']*100:.2f}%",
                      help="% of real bugs correctly flagged. PRIMARY metric.")
        with mc2:
            st.metric("⚖️ F1-Score", f"{metrics['f1']*100:.2f}%",
                      help="Harmonic mean of Precision and Recall.")
        with mc3:
            st.metric("🔍 Precision", f"{metrics['precision']*100:.2f}%",
                      help="% of predicted bugs that are actually bugs.")
        with mc4:
            st.metric("📐 ROC-AUC", f"{metrics['roc_auc']*100:.2f}%",
                      help="Area under the ROC curve – overall discriminative power.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confusion Matrix + Feature Importance ─────────────────────────
        chart_col1, chart_col2 = st.columns([1, 1.4])

        with chart_col1:
            st.markdown("#### 🔲 Confusion Matrix")

            tn, fp, fn, tp = cm_data.ravel()

            fig_cm = go.Figure(
                go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Predicted: Clean", "Predicted: Defective"],
                    y=["Actual: Clean", "Actual: Defective"],
                    text=[[f"TN<br>{tn}", f"FP<br>{fp}"],
                          [f"FN<br>{fn}", f"TP<br>{tp}"]],
                    texttemplate="%{text}",
                    textfont=dict(size=16, color="white"),
                    colorscale=[
                        [0.0,  "rgba(15,22,38,0.9)"],
                        [0.35, "rgba(34,211,197,0.35)"],
                        [1.0,  "rgba(34,211,197,0.85)"],
                    ],
                    showscale=False,
                    hovertemplate=(
                        "%{y} | %{x}<br>Count: %{z}<extra></extra>"
                    ),
                )
            )
            # Highlight FN (high-cost error) with red border.
            fig_cm.add_shape(
                type="rect", x0=-0.5, y0=0.5, x1=0.5, y1=1.5,
                line=dict(color=COLOR_DEFECT, width=2.5),
                fillcolor="rgba(0,0,0,0)",
            )
            fig_cm.add_annotation(
                x=0, y=1.0,
                text="⚠️ High-Cost<br>Error",
                font=dict(size=10, color=COLOR_DEFECT),
                showarrow=False,
                yshift=-38,
                xshift=0,
            )
            fig_cm.update_layout(
                **{
                    **PLOTLY_LAYOUT,
                    "xaxis": dict(side="bottom", gridcolor="rgba(0,0,0,0)"),
                    "yaxis": dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                    "height": 380,
                }
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with chart_col2:
            st.markdown("#### 📊 Feature Importance Ranking")
            st.caption("Which static code metrics are the strongest bug predictors?")

            # Show top-15 features for readability.
            top_n      = min(15, len(feat_names))
            top_names  = feat_names[:top_n][::-1]   # Reverse for horizontal bar.
            top_imps   = feat_imps[:top_n][::-1]

            # Gradient colour by importance.
            norm_imps = np.array(top_imps)
            if norm_imps.max() > 0:
                norm_imps = norm_imps / norm_imps.max()

            fig_fi = go.Figure(
                go.Bar(
                    x=top_imps,
                    y=top_names,
                    orientation="h",
                    marker=dict(
                        color=norm_imps,
                        colorscale=[[0, "rgba(79,142,247,0.4)"], [1, COLOR_TEAL := "#22d3c5"]],
                        showscale=False,
                        line=dict(color="rgba(0,0,0,0)", width=0),
                    ),
                    text=[f"{v:.3f}" for v in top_imps],
                    textposition="outside",
                    textfont=dict(color="rgba(255,255,255,0.7)", size=10),
                    hovertemplate=(
                        "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                    ),
                )
            )
            fig_fi.update_layout(
                **PLOTLY_LAYOUT,
                height=420,
                xaxis_title="Feature Importance Score",
                yaxis_title="",
                bargap=0.3,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Model Info Box ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ℹ️ Model Architecture")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(
                "**Algorithm:** Random Forest Classifier\n\n"
                "300 decision trees with `sqrt` feature subsampling."
            )
        with info_col2:
            st.info(
                "**Imbalance Handling:** SMOTE + `balanced_subsample`\n\n"
                "Prevents bias toward the majority (clean) class."
            )
        with info_col3:
            st.info(
                f"**Decision Threshold:** {threshold:.2f}\n\n"
                "Tuned to maximise Recall (minimise missed defects)."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4  –  INTERACTIVE CODE RISK PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<p class="section-heading">⚡ Interactive Code Risk Predictor</p>', unsafe_allow_html=True)

    if not MODEL_LOADED or artefact is None:
        st.warning(
            "🔧 Model not loaded. "
            "Run `python src/train_model.py data/KC1.csv` from the project root, "
            "then refresh this page."
        )
    else:
        model_clf    = artefact["model"]
        threshold    = artefact["threshold"]
        feature_list = artefact["feature_names"]   # Original training order.

        st.markdown(
            """
            <div style='background:rgba(79,142,247,0.08); border:1px solid rgba(79,142,247,0.2);
                        border-radius:14px; padding:1rem 1.5rem; margin-bottom:1.5rem;
                        font-size:0.88rem; color:#94a3b8;'>
                🧪 Enter the static code metrics for a software module below.
                The trained Random Forest model will compute the real-time probability
                that this module contains a defect.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Input Form ────────────────────────────────────────────────────
        # Build a friendly display name mapping.
        DISPLAY_NAMES = {
            "loc":              "Lines of Code (loc)",
            "v(g)":             "McCabe Cyclomatic Complexity v(g)",
            "ev(g)":            "Essential Complexity ev(g)",
            "iv(g)":            "Design Complexity iv(g)",
            "n":                "Halstead Total N",
            "v":                "Halstead Volume",
            "l":                "Halstead Length",
            "d":                "Halstead Difficulty",
            "i":                "Halstead Intelligence",
            "e":                "Halstead Effort",
            "b":                "Halstead Bugs (est.)",
            "t":                "Halstead Time (sec)",
            "locode":           "Executable Lines (lOCode)",
            "locomment":        "Comment Lines (lOComment)",
            "loblank":          "Blank Lines (lOBlank)",
            "locodeandcomment": "Code + Comment Lines",
            "uniq_op":          "Unique Operators",
            "uniq_opnd":        "Unique Operands",
            "total_op":         "Total Operators",
            "total_opnd":       "Total Operands",
            "branchcount":      "Branch Count",
        }

        # Sensible default values for common MDP metrics.
        DEFAULTS = {
            "loc": 50.0, "v(g)": 5.0, "ev(g)": 3.0, "iv(g)": 3.0,
            "n": 150.0, "v": 500.0, "l": 0.1, "d": 15.0,
            "i": 35.0,  "e": 7500.0, "b": 0.15, "t": 416.0,
            "locode": 40.0, "locomment": 5.0, "loblank": 5.0, "locodeandcomment": 2.0,
            "uniq_op": 12.0, "uniq_opnd": 20.0, "total_op": 80.0, "total_opnd": 70.0,
            "branchcount": 6.0,
        }

        st.markdown("#### 📥 Module Metrics Input")

        user_values = {}
        cols_per_row = 3
        feature_chunks = [
            feature_list[i: i + cols_per_row]
            for i in range(0, len(feature_list), cols_per_row)
        ]

        for chunk in feature_chunks:
            row_cols = st.columns(cols_per_row)
            for col, feat in zip(row_cols, chunk):
                display = DISPLAY_NAMES.get(feat.lower(), feat)
                default = float(DEFAULTS.get(feat.lower(), 10.0))
                with col:
                    user_values[feat] = st.number_input(
                        label=display,
                        min_value=0.0,
                        value=default,
                        step=1.0,
                        format="%.2f",
                        key=f"input_{feat}",
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict Button ────────────────────────────────────────────────
        predict_btn = st.button("🔍 Analyze Component Risk", key="predict_button")

        if predict_btn:
            # Construct feature vector in the exact training order.
            input_vector = np.array(
                [[user_values[f] for f in feature_list]], dtype=np.float32
            )

            # Get defect probability.
            proba_defective = float(model_clf.predict_proba(input_vector)[0][1])
            risk_pct        = proba_defective * 100.0
            is_high_risk    = proba_defective >= threshold

            st.markdown("---")
            st.markdown("#### 📊 Risk Assessment Result")

            # Progress bar (risk gauge).
            bar_col, detail_col = st.columns([1.5, 1])

            with bar_col:
                st.markdown(
                    f"""
                    <div style='font-size:0.85rem; color:#6b7a96; margin-bottom:0.4rem;'>
                        Bug Risk Probability
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(
                    min(int(risk_pct), 100),
                    text=f"**{risk_pct:.1f}%** defect probability",
                )

                # Gauge-style big number.
                risk_color = COLOR_DEFECT if is_high_risk else COLOR_CLEAN
                st.markdown(
                    f"""
                    <div style='font-size:3.5rem; font-weight:800; color:{risk_color};
                                margin-top:0.5rem; font-family:JetBrains Mono, monospace;'>
                        {risk_pct:.1f}%
                    </div>
                    <div style='color:#6b7a96; font-size:0.85rem; margin-top:-0.3rem;'>
                        Defect Probability  |  Threshold: {threshold:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with detail_col:
                if is_high_risk:
                    st.markdown(
                        f"""
                        <div class='risk-high'>
                            <div class='risk-title' style='color:#ef4444;'>
                                🚨 HIGH RISK — Refactor Required
                            </div>
                            <div class='risk-text'>
                                This module shows a <b>{risk_pct:.1f}%</b> probability
                                of containing a defect, exceeding the
                                <b>{threshold*100:.0f}%</b> deployment safety threshold.<br><br>
                                <b>Recommended Actions:</b>
                                <ul style='margin-top:0.4rem; padding-left:1.2rem;'>
                                    <li>Reduce cyclomatic complexity</li>
                                    <li>Add unit test coverage</li>
                                    <li>Conduct peer code review</li>
                                    <li>Consider refactoring into smaller units</li>
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='risk-low'>
                            <div class='risk-title' style='color:#10b981;'>
                                ✅ LOW RISK — Safe to Deploy
                            </div>
                            <div class='risk-text'>
                                This module shows only a <b>{risk_pct:.1f}%</b> defect
                                probability — below the <b>{threshold*100:.0f}%</b>
                                safety threshold.<br><br>
                                <b>Status:</b> Module passes quality gate.
                                Standard monitoring applies.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ── Radar Chart of Input Metrics ──────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🕸️ Module Metric Radar Profile")
            st.caption(
                "Normalised view of the input metrics relative to the dataset median. "
                "Spikes beyond the boundary indicate unusually high values."
            )

            if DATA_LOADED and df is not None:
                radar_features = [f for f in feature_list if f in df.columns][:8]
                medians = df[radar_features].median()

                radar_vals   = [user_values[f] for f in radar_features]
                radar_median = [medians[f] for f in radar_features]

                # Normalise both by dataset max for display.
                df_max = df[radar_features].quantile(0.95)
                r_norm = [
                    min(user_values[f] / max(df_max[f], 1e-6), 2.0)
                    for f in radar_features
                ]
                m_norm = [
                    min(medians[f] / max(df_max[f], 1e-6), 2.0)
                    for f in radar_features
                ]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_norm + [r_norm[0]],
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    name="This Module",
                    fillcolor=f"rgba(239,68,68,0.15)" if is_high_risk else "rgba(34,211,197,0.15)",
                    line=dict(color=COLOR_DEFECT if is_high_risk else COLOR_CLEAN, width=2),
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=m_norm + [m_norm[0]],
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    name="Dataset Median",
                    fillcolor="rgba(79,142,247,0.08)",
                    line=dict(color=COLOR_BLUE, width=1.5, dash="dash"),
                ))
                fig_radar.update_layout(
                    **PLOTLY_LAYOUT,
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True,
                            color="#6b7a96",
                            gridcolor="rgba(255,255,255,0.08)",
                        ),
                        angularaxis=dict(color="#e8edf5"),
                    ),
                    legend=dict(
                        bgcolor="rgba(0,0,0,0.3)",
                        bordercolor="rgba(255,255,255,0.1)",
                    ),
                    height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style='border-color:rgba(255,255,255,0.06); margin-top:3rem;'>
    <div style='text-align:center; color:#374151; font-size:0.78rem; padding-bottom:1rem;'>
        🛰️ NASA MDP Software Defect Prediction  ·
        Powered by Random Forest + SMOTE  ·
        Built with Streamlit &amp; Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
