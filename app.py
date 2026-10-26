from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "target"
LABELS = ["malignant", "benign"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
PALETTE = ["#3157d5", "#0e9f91", "#f59e0b", "#e11d48", "#7c3aed"]
SAMPLE_PATH = ROOT / "test_data.csv"

PAGES = ["🏠 Overview", "📊 Model Comparison", "🔍 Diagnostics", "🧪 Predict", "🗂️ Data Explorer", "ℹ️ About"]

st.set_page_config(page_title="AIML Model Studio", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
LIGHT = dict(
    dark=False, bg="linear-gradient(135deg,#f5f8ff 0%,#f8fbff 45%,#eef8f7 100%)",
    ink="#16233a", head="#172554", muted="#65748b", card="#ffffff", line="#e3eaf6",
    tab="#eaf0ff", note="#eef5ff", noteink="#334766", foot="#8491a7", accent="#3157d5",
    sidebar="#0f1b3d", shadow="rgba(25,50,90,.05)", code="#f3f6fc",
)
DARK = dict(
    dark=True, bg="linear-gradient(135deg,#0a1020 0%,#0d1527 45%,#0a1a1d 100%)",
    ink="#e4ebfa", head="#c7d7ff", muted="#98a6bf", card="#131c2f", line="#24314b",
    tab="#1a2540", note="#13213d", noteink="#c3d0ea", foot="#6d7b95", accent="#5b7cff",
    sidebar="#080e1f", shadow="rgba(0,0,0,.35)", code="#0e1626",
)

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] * { color:var(--ink) !important; }
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1400px; }
#MainMenu, footer { visibility:hidden; }

/* text colours that follow the chosen theme (not Streamlit's native one) */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color:var(--head); font-family:'Manrope',sans-serif; }
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"] p, .stApp label, .stApp [data-testid="stRadio"] p { color:var(--ink); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color:var(--muted); }

/* sidebar (always dark navy) */
section[data-testid="stSidebar"] { background:var(--sidebar); }
section[data-testid="stSidebar"] * { color:#e6ecff !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background:#fff !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] * { color:#16233a !important; }
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.15); }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.08); border:1px dashed rgba(255,255,255,.3); }
section[data-testid="stSidebar"] button { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.3); }
.brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.3rem; margin-bottom:.1rem; }
.brand-sub { font-size:.75rem; opacity:.7; margin-bottom:.8rem; }

/* hero */
.hero { background:linear-gradient(120deg,#172554 0%,#2549a8 58%,#0f766e 100%); padding:1.8rem 2.2rem;
  border-radius:22px; color:#fff; box-shadow:0 16px 42px rgba(28,55,120,.18); margin-bottom:1.2rem; }
.hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:700; opacity:.78; }
.stApp [data-testid="stMarkdownContainer"] .hero h1 { font-size:2.1rem; margin:.4rem 0 .4rem 0; color:#fff; padding:0; }
.stApp [data-testid="stMarkdownContainer"] .hero p { color:#e2eaff; margin:0; max-width:780px; }
.chips { display:flex; gap:.6rem; flex-wrap:wrap; margin-top:1rem; }
.chip { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); border-radius:999px;
  padding:.3rem .85rem; font-size:.8rem; }

/* headings, cards, metrics */
.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:var(--head); font-size:1.3rem; margin:.4rem 0 .15rem 0; }
.section-sub { color:var(--muted); margin-bottom:.8rem; }
div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); padding:.9rem 1.1rem; border-radius:16px;
  box-shadow:0 5px 18px var(--shadow); }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color:var(--muted) !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color:var(--head) !important; font-family:'Manrope',sans-serif; }
.card { background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.3rem;
  box-shadow:0 5px 18px var(--shadow); height:100%; }
.stApp [data-testid="stMarkdownContainer"] .card h4 { margin:0 0 .4rem 0; color:var(--head); }
.pill { display:inline-block; padding:.25rem .8rem; border-radius:999px; font-weight:700; font-size:.85rem; }
.pill-ok { background:#dcfce7; color:#166534; } .pill-bad { background:#fee2e2; color:#991b1b; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:var(--tab); border-radius:10px 10px 0 0; padding:.55rem 1rem; }
.stApp .stTabs [data-baseweb="tab"] p { color:var(--ink); }
.stTabs [aria-selected="true"] { background:var(--accent) !important; }
.stApp .stTabs [aria-selected="true"] p { color:#fff !important; }

/* inputs, popovers, alerts, expanders, code */
.block-container div[data-baseweb="select"] > div, .block-container div[data-baseweb="input"],
.block-container div[data-baseweb="base-input"], .block-container input { background:var(--card); color:var(--ink); border-color:var(--line); }
.block-container div[data-baseweb="select"] * { color:var(--ink); }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background:var(--card) !important; color:var(--ink) !important; }
.block-container div[data-testid="stAlert"] { background:var(--card); border:1px solid var(--line); }
.block-container div[data-testid="stAlert"] * { color:var(--ink) !important; }
.block-container [data-testid="stExpander"] { background:var(--card); border:1px solid var(--line); border-radius:12px; }
.block-container [data-testid="stExpander"] summary, .block-container [data-testid="stExpander"] summary * { color:var(--ink) !important; }
.block-container pre, .block-container pre * { background:var(--code) !important; color:var(--ink) !important; }
.block-container [data-testid="stTickBarMin"], .block-container [data-testid="stTickBarMax"] { color:var(--muted); }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.block-container .stButton button, .block-container .stDownloadButton button { background:var(--card); border:1px solid var(--line); border-radius:10px; font-weight:600; }
.block-container .stButton button p, .block-container .stDownloadButton button p { color:var(--ink); }

div[role="dialog"] { background:var(--card) !important; border:1px solid var(--line); border-radius:20px !important; }
div[role="dialog"] h2, div[role="dialog"] h2 * { color:var(--head) !important; font-family:'Manrope',sans-serif; }
div[role="dialog"] button[aria-label="Close"] { color:var(--ink); }
button[kind="primary"], button[data-testid="stBaseButton-primary"] { background:var(--accent) !important; border:1px solid var(--accent) !important; border-radius:10px; font-weight:700; }
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color:#fff !important; }
.note { background:var(--note); border-left:4px solid var(--accent); border-radius:10px; padding:.8rem 1rem; color:var(--noteink); margin-top:1.2rem; }
.note b { color:var(--head); }
.footer { color:var(--foot); text-align:center; font-size:.8rem; padding-top:2rem; }
"""


def apply_theme(dark: bool):
    """Inject CSS for the chosen theme, style matplotlib to match, and return the palette."""
    T = DARK if dark else LIGHT
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in T.items() if k != "dark") + "}"
    # st.dataframe is canvas-based and follows Streamlit's own theme; invert it when that differs from ours.
    try:
        native_dark = (st.context.theme.type or "light") == "dark"
    except Exception:
        native_dark = False
    extra = ""
    if native_dark != dark:
        extra = 'div[data-testid="stDataFrame"] { filter:invert(.92) hue-rotate(180deg); }'
    st.markdown(f"<style>{STATIC_CSS}{root}{extra}</style>", unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True, "text.color": T["ink"], "axes.labelcolor": T["ink"],
        "axes.edgecolor": T["line"], "axes.titlecolor": T["ink"], "xtick.color": T["muted"],
        "ytick.color": T["muted"], "grid.color": T["line"], "legend.facecolor": T["card"],
        "legend.edgecolor": T["line"], "legend.labelcolor": T["ink"],
    })
    return T


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_data
def read_sample():
    return pd.read_csv(SAMPLE_PATH)


def get_scores(model, X):
    """Return (scores, kind) where kind is 'proba', 'decision' or None."""
    try:
        return model.predict_proba(X)[:, 1], "proba"
    except Exception:
        try:
            return model.decision_function(X), "decision"
        except Exception:
            return None, None


def predict_at(model, X, threshold):
    scores, kind = get_scores(model, X)
    if kind == "proba":
        return (scores >= threshold).astype(int), scores, kind
    return model.predict(X), scores, kind


def compute_metrics(y, pred, scores):
    auc = np.nan
    if scores is not None and y.nunique() == 2:
        auc = roc_auc_score(y, scores)
    return {
        "Accuracy": accuracy_score(y, pred), "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1 Score": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def fmt(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=.25)


def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#16233a")
    fig.tight_layout()
    return fig


def plot_grouped_bars(comp, cols):
    fig, ax = plt.subplots(figsize=(9, 3.9))
    n = len(cols); w = 0.8 / n; x = np.arange(len(comp))
    for i, c in enumerate(cols):
        ax.bar(x - 0.4 + w * (i + 0.5), comp[c].fillna(0), w, label=c, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, comp.index, rotation=15, ha="right")
    ax.set_ylim(max(0.0, float(np.nanmin(comp[cols].values)) - 0.1), 1.02)
    ax.legend(ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    style_ax(ax); fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------------
models = load_models()
if not models:
    st.error("No saved models were found. Ensure the model/ directory is included in the repository.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">🧠 Model Studio</div><div class="brand-sub">ML Assignment 2 · BITS Pilani</div>',
                unsafe_allow_html=True)
    _toggle = getattr(st, "toggle", st.checkbox)
    dark = _toggle("🌙 Dark mode", key="dark_mode", help="Switch between light and dark appearance.")
    T = apply_theme(dark)
    page = st.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed")
    st.divider()
    selected_model = st.selectbox("Active classifier", list(models.keys()))
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Applies to models that output probabilities. "
                               "Lower it to catch more benign cases; raise it to be stricter.")
    st.divider()
    st.markdown("**Test data**")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="Must include all 30 features and a 'target' column (0 = malignant, 1 = benign).")
    if SAMPLE_PATH.exists():
        st.download_button("⬇ Sample test CSV", SAMPLE_PATH.read_bytes(), "test_data.csv", "text/csv",
                           use_container_width=True)
    st.caption("Saved models · no retraining · 0 = malignant, 1 = benign")
    help_clicked = st.button("❓ How to use", use_container_width=True)

# ----------------------------------------------------------------------------
# Welcome guide (pop-up on first load, reopen with "How to use" in the sidebar)
# ----------------------------------------------------------------------------
WELCOME_HTML = """<style>
@keyframes wl-pop { 0%{transform:scale(.5);opacity:0} 70%{transform:scale(1.1);opacity:1} 100%{transform:scale(1);opacity:1} }
@keyframes wl-rise { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
@keyframes wl-float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes wl-pulse { 0%{box-shadow:0 0 0 0 rgba(49,87,213,.5)} 70%{box-shadow:0 0 0 10px rgba(49,87,213,0)} 100%{box-shadow:0 0 0 0 rgba(49,87,213,0)} }
@keyframes wl-shine { to{background-position:200% center} }
.wl-top { text-align:center; margin-bottom:.4rem; }
.wl-logo { font-size:3.2rem; display:inline-block; animation:wl-pop .7s ease both, wl-float 3s ease-in-out .8s infinite; }
.wl-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.5rem; margin-top:.2rem;
  background:linear-gradient(90deg,#3157d5,#0e9f91,#3157d5); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; color:transparent;
  animation:wl-shine 4s linear infinite; }
.wl-sub { color:var(--muted); font-size:.95rem; margin-top:.2rem; animation:wl-rise .6s ease .2s both; }
.wl-step { display:flex; gap:.9rem; align-items:flex-start; padding:.75rem .95rem; margin:.55rem 0;
  border:1px solid var(--line); border-radius:14px; background:var(--card); opacity:0; animation:wl-rise .55s ease forwards; }
.wl-d1 { animation-delay:.35s; } .wl-d2 { animation-delay:.6s; } .wl-d3 { animation-delay:.85s; } .wl-d4 { animation-delay:1.1s; }
.wl-num { flex:0 0 2rem; height:2rem; border-radius:50%; background:var(--accent); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; animation:wl-pulse 2s infinite; }
.wl-h { font-weight:700; color:var(--head); }
.wl-p { color:var(--ink); font-size:.9rem; margin-top:.1rem; }
.wl-tip { text-align:center; color:var(--muted); font-size:.85rem; margin:.7rem 0 .4rem 0; animation:wl-rise .6s ease 1.35s both; }
@media (prefers-reduced-motion: reduce) { .wl-logo,.wl-title,.wl-num { animation:none; } .wl-step,.wl-sub,.wl-tip { animation:none; opacity:1; } }
</style>
<div class="wl-top">
<div class="wl-logo">🧠</div>
<div class="wl-title">Breast Cancer Classification Studio</div>
<div class="wl-sub">Compare five machine-learning models in four easy steps.</div>
</div>
<div class="wl-step wl-d1"><div class="wl-num">1</div><div><div class="wl-h">🧠 Pick a classifier</div><div class="wl-p">Use <b>Active classifier</b> in the left sidebar to choose one of the five saved models.</div></div></div>
<div class="wl-step wl-d2"><div class="wl-num">2</div><div><div class="wl-h">📁 Choose your data</div><div class="wl-p">The built-in test set loads automatically. Or upload your own CSV with the 30 features and a <b>target</b> column.</div></div></div>
<div class="wl-step wl-d3"><div class="wl-num">3</div><div><div class="wl-h">🧭 Explore the pages</div><div class="wl-p"><b>Overview</b> for the quick picture, <b>Model Comparison</b> to rank models, <b>Diagnostics</b> to see mistakes, <b>Predict</b> to try your own values.</div></div></div>
<div class="wl-step wl-d4"><div class="wl-num">4</div><div><div class="wl-h">🎚️ Fine-tune and personalise</div><div class="wl-p">Slide the <b>decision threshold</b> or switch <b>Dark mode</b> any time from the sidebar.</div></div></div>
<div class="wl-tip">💡 Reopen this guide any time with <b>❓ How to use</b> in the sidebar.</div>
"""


def _go_home():
    st.session_state["nav_page"] = PAGES[0]
    st.session_state["welcomed"] = True


def _welcome_body():
    st.markdown(WELCOME_HTML, unsafe_allow_html=True)
    return st.button("✅ Got it — take me to the app", type="primary", use_container_width=True, on_click=_go_home)


_dialog = getattr(st, "dialog", None)
if _dialog is not None:
    @_dialog("👋 Welcome to Model Studio", width="large")
    def welcome_dialog():
        if _welcome_body():
            st.rerun()

    if help_clicked or not st.session_state.get("welcomed"):
        st.session_state["welcomed"] = True  # a dismissed pop-up won't keep reopening
        welcome_dialog()
else:  # older Streamlit without st.dialog: show the guide as a first screen instead
    if help_clicked:
        st.session_state["welcomed"] = False
    if not st.session_state.get("welcomed"):
        with st.container(border=True):
            _welcome_body()
        st.stop()


# ----------------------------------------------------------------------------
# Data loading & validation
# ----------------------------------------------------------------------------
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = f"Uploaded: {uploaded.name}"
    elif SAMPLE_PATH.exists():
        df = read_sample()
        data_source = "Bundled test split"
    else:
        st.error("No test data available. Upload a CSV from the sidebar.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

model = models[selected_model]
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
elif SAMPLE_PATH.exists():
    features = [c for c in read_sample().columns if c != TARGET]
else:
    features = [c for c in df.columns if c != TARGET]

has_target = TARGET in df.columns
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns ({len(missing)}): {', '.join(missing[:8])}" + ("…" if len(missing) > 8 else ""))
    st.stop()
if df.empty:
    st.warning("The CSV contains no rows.")
    st.stop()
try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int) if has_target else None
except Exception as exc:
    st.error(f"Feature and target columns must be numeric: {exc}")
    st.stop()
if has_target and not set(y.unique()).issubset({0, 1}):
    st.error("Target values must be 0 (malignant) or 1 (benign).")
    st.stop()
extra = [c for c in df.columns if c not in features + [TARGET]]


@st.cache_data(show_spinner=False)
def evaluate_all(_models, X, y, thr, key):
    """Metrics, predictions and scores for every model. `key` busts the cache when models change."""
    out = {}
    for name, clf in _models.items():
        pred, scores, kind = predict_at(clf, X, thr)
        out[name] = {"pred": pred, "scores": scores, "kind": kind,
                     "metrics": compute_metrics(y, pred, scores)}
    return out


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">BITS Pilani · M.Tech AIML · Machine Learning Assignment 2</div>
  <h1>Breast Cancer Classification Studio</h1>
  <p>Benchmark five classical classifiers, compare them side by side, inspect errors and try live predictions.</p>
  <div class="chips">
    <span class="chip">👤 Yaythish Kannaa G S</span>
    <span class="chip">🆔 2026AJ05374</span>
    <span class="chip">🧬 Breast Cancer Wisconsin (Diagnostic)</span>
    <span class="chip">📁 {data_source} · {len(df):,} rows</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not has_target and page in ("🏠 Overview", "📊 Model Comparison", "🔍 Diagnostics"):
    st.warning("This CSV has no 'target' column, so evaluation pages are unavailable. "
               "Use the **Predict** page for batch predictions, or upload a labeled file.")
    page = "🧪 Predict"

results = evaluate_all(models, X, y, threshold, f"{data_source}-{len(df)}") if has_target else None

# ============================================================================
# Overview
# ============================================================================
if page == "🏠 Overview":
    section("Overview", f"Performance of **{selected_model}** on the current test set.")
    m = results[selected_model]["metrics"]
    cols = st.columns(6)
    for col, name in zip(cols, METRICS):
        col.metric(name, fmt(m[name]))

    best = max(results, key=lambda k: results[k]["metrics"]["F1 Score"])
    st.caption(f"Positive class for precision/recall/F1/AUC: benign (1) · Best F1 on this data: **{best}** "
               f"({fmt(results[best]['metrics']['F1 Score'])})")
    if results[selected_model]["kind"] != "proba":
        st.info("The selected model does not expose probabilities, so the threshold slider has no effect on it.")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y, results[selected_model]["pred"], labels=[0, 1])
        st.pyplot(plot_confusion(cm, selected_model), use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✅ {tp + tn} correct · ❌ {fp} malignant predicted benign · ❌ {fn} benign predicted malignant")
    with right:
        st.markdown("##### ROC curves (all models)")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for c, (name, r) in zip(PALETTE, results.items()):
            if r["scores"] is not None and y.nunique() == 2:
                fpr, tpr, _ = roc_curve(y, r["scores"])
                ax.plot(fpr, tpr, color=c, lw=3 if name == selected_model else 1.6,
                        alpha=1 if name == selected_model else .7,
                        label=f"{name} (AUC {r['metrics']['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8, loc="lower right"); style_ax(ax); fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ============================================================================
# Model comparison
# ============================================================================
elif page == "📊 Model Comparison":
    section("Model comparison", "All classifiers evaluated on the same data at the current threshold.")
    comp = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T[METRICS]
    comp.index.name = "Model"

    focus = st.multiselect("Metrics to chart", METRICS, default=["Accuracy", "AUC", "F1 Score", "MCC"])
    t1, t2, t3 = st.tabs(["Table", "Bar chart", "Radar"])
    with t1:
        st.dataframe(comp.style.format("{:.4f}", na_rep="N/A").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)
        rank = comp.rank(ascending=False).mean(axis=1).sort_values()
        st.success(f"🏆 Best overall (mean rank across metrics): **{rank.index[0]}**")
        st.download_button("⬇ Download comparison CSV", comp.reset_index().to_csv(index=False),
                           "evaluation_metrics.csv", "text/csv")
    with t2:
        if focus:
            fig = plot_grouped_bars(comp, focus)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Select at least one metric.")
    with t3:
        radar_metrics = [m for m in METRICS if not comp[m].isna().any()]
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for c, (name, row) in zip(PALETTE, comp.iterrows()):
            vals = [max(row[m], 0) for m in radar_metrics] + [max(row[radar_metrics[0]], 0)]
            ax.plot(angles, vals, color=c, lw=2, label=name)
            ax.fill(angles, vals, color=c, alpha=.07)
        ax.set_xticks(angles[:-1], radar_metrics)
        lo = max(0.0, float(np.nanmin(comp[radar_metrics].values)) - 0.05)
        ax.set_ylim(lo, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# ============================================================================
# Diagnostics
# ============================================================================
elif page == "🔍 Diagnostics":
    section("Diagnostics", f"Deep-dive into **{selected_model}**.")
    r = results[selected_model]
    t1, t2, t3, t4 = st.tabs(["Classification report", "Precision–Recall", "Score distribution", "Misclassified rows"])
    with t1:
        report = classification_report(y, r["pred"], labels=[0, 1], target_names=LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)
        with st.expander("Plain-text report"):
            st.code(classification_report(y, r["pred"], labels=[0, 1], target_names=LABELS, zero_division=0))
    with t2:
        if r["scores"] is not None and y.nunique() == 2:
            p, rc, _ = precision_recall_curve(y, r["scores"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rc, p, color=PALETTE[0], lw=2.5)
            ax.fill_between(rc, p, alpha=.1, color=PALETTE[0])
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"Average precision = {average_precision_score(y, r['scores']):.4f}")
            style_ax(ax); fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        else:
            st.info("Scores unavailable for this model or data.")
    with t3:
        if r["scores"] is not None:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            for lbl, name, c in [(0, "malignant", PALETTE[3]), (1, "benign", PALETTE[1])]:
                ax.hist(r["scores"][y.values == lbl], bins=25, alpha=.6, color=c, label=f"Actual {name}")
            if r["kind"] == "proba":
                ax.axvline(threshold, color=T["ink"], ls="--", label=f"Threshold {threshold:.2f}")
            ax.set_xlabel("Model score (benign)"); ax.set_ylabel("Count")
            ax.legend(); style_ax(ax); fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        else:
            st.info("Scores unavailable for this model.")
    with t4:
        wrong = df.copy()
        wrong["predicted"] = r["pred"]
        wrong = wrong[wrong["predicted"] != y.values]
        if wrong.empty:
            st.success("🎉 No misclassified rows for this model on this data.")
        else:
            st.warning(f"{len(wrong)} misclassified of {len(df)} rows")
            st.dataframe(wrong, use_container_width=True)
            st.download_button("⬇ Download misclassified rows", wrong.to_csv(index=False),
                               "misclassified.csv", "text/csv")

# ============================================================================
# Predict
# ============================================================================
elif page == "🧪 Predict":
    section("Predict", "Try a single sample interactively, or score the whole file and download the results.")
    t1, t2 = st.tabs(["Single sample", "Batch predictions"])

    with t1:
        idx = st.number_input("Start from row", 0, len(X) - 1, 0, help="Row of the current data used as starting values.")
        base = X.iloc[int(idx)]
        if has_target:
            st.caption(f"Actual label for this row: **{LABELS[int(y.iloc[int(idx)])]}**")
        groups = {
            "Mean": [f for f in features if f.startswith("mean")],
            "Error": [f for f in features if "error" in f],
            "Worst": [f for f in features if f.startswith("worst")],
        }
        other = [f for f in features if not any(f in g for g in groups.values())]
        if other:
            groups["Other"] = other
        groups = {k: v for k, v in groups.items() if v}
        values = {}
        tabs = st.tabs(list(groups.keys()))
        for tab, (gname, flist) in zip(tabs, groups.items()):
            with tab:
                cols = st.columns(3)
                for i, f in enumerate(flist):
                    lo, hi = float(X[f].min()), float(X[f].max())
                    pad = (hi - lo) * 0.5 or 1.0
                    values[f] = cols[i % 3].slider(f, lo - pad * 0.0, hi + pad, float(np.clip(base[f], lo, hi + pad)),
                                                   step=(hi - lo + pad) / 200 or 0.01,
                                                   key=f"{f}_{int(idx)}", format="%.4f")
        row = pd.DataFrame([values])[features]
        pred, scores, kind = predict_at(model, row, threshold)
        label = LABELS[int(pred[0])]
        ok = int(pred[0]) == 1
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<div class="card"><h4>Prediction</h4>'
                        f'<span class="pill {"pill-ok" if ok else "pill-bad"}">{label.upper()}</span>'
                        f'<p style="margin-top:.7rem;color:#65748b">Model: {selected_model}</p></div>',
                        unsafe_allow_html=True)
        with c2:
            if kind == "proba":
                st.markdown("**Probability**")
                st.progress(float(scores[0]), text=f"Benign {scores[0]:.1%} · Malignant {1 - scores[0]:.1%}")
            elif kind == "decision":
                st.metric("Decision function score", f"{scores[0]:.4f}")
        st.markdown("##### What do all models say?")
        votes = []
        for n, clf in models.items():
            p, s, k = predict_at(clf, row, threshold)
            votes.append({"Model": n, "Prediction": LABELS[int(p[0])],
                          "P(benign)": float(s[0]) if k == "proba" else np.nan})
        st.dataframe(pd.DataFrame(votes).style.format({"P(benign)": "{:.3f}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)

    with t2:
        pred, scores, kind = predict_at(model, X, threshold)
        out = df.copy()
        out["predicted"] = pred
        out["predicted_label"] = [LABELS[i] for i in pred]
        if kind == "proba":
            out["p_benign"] = scores
        k1, k2, k3 = st.columns(3)
        k1.metric("Rows scored", f"{len(out):,}")
        k2.metric("Predicted benign", int((pred == 1).sum()))
        k3.metric("Predicted malignant", int((pred == 0).sum()))
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("⬇ Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

# ============================================================================
# Data explorer
# ============================================================================
elif page == "🗂️ Data Explorer":
    section("Data explorer", "Understand the data behind the metrics.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{df.shape[0]:,}")
    k2.metric("Columns", df.shape[1])
    k3.metric("Missing values", int(df.isna().sum().sum()))
    if has_target:
        k4.metric("Benign share", f"{(y == 1).mean():.1%}")
    t1, t2, t3, t4 = st.tabs(["Preview", "Statistics", "Feature distribution", "Correlation"])
    with t1:
        st.dataframe(df.head(50), use_container_width=True)
        if extra:
            st.info(f"Extra columns ignored by the models: {', '.join(extra[:8])}" + ("…" if len(extra) > 8 else ""))
    with t2:
        st.dataframe(X.describe().T.style.format("{:.3f}"), use_container_width=True)
    with t3:
        feat = st.selectbox("Feature", features)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        if has_target:
            for lbl, name, c in [(0, "malignant", PALETTE[3]), (1, "benign", PALETTE[1])]:
                ax.hist(X.loc[y == lbl, feat], bins=25, alpha=.6, color=c, label=name)
            ax.legend()
        else:
            ax.hist(X[feat], bins=25, color=PALETTE[0], alpha=.8)
        ax.set_xlabel(feat); ax.set_ylabel("Count"); style_ax(ax); fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    with t4:
        top_n = st.slider("Number of features", 5, len(features), min(15, len(features)))
        if has_target:
            chosen = X.corrwith(y).abs().sort_values(ascending=False).head(top_n).index.tolist()
            st.caption("Showing the features most correlated with the target.")
        else:
            chosen = features[:top_n]
        corr = X[chosen].corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(chosen)), chosen, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(chosen)), chosen, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# ============================================================================
# About
# ============================================================================
else:
    section("About this app")
    a, b = st.columns(2)
    with a:
        st.markdown("""<div class="card"><h4>What it does</h4>
        Loads five pre-trained classifiers and evaluates them on a labeled test CSV.
        Nothing is retrained in the app.<br><br>
        <b>Pages</b><br>
        🏠 Overview – KPIs, confusion matrix, ROC<br>
        📊 Comparison – table, bars, radar, best model<br>
        🔍 Diagnostics – report, PR curve, score histogram, errors<br>
        🧪 Predict – single-sample sliders and batch scoring<br>
        🗂️ Data Explorer – stats, distributions, correlations</div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="card"><h4>Metric guide</h4>
        <b>Accuracy</b> – share of correct predictions<br>
        <b>AUC</b> – ranking quality across thresholds<br>
        <b>Precision</b> – of predicted benign, how many are truly benign<br>
        <b>Recall</b> – of true benign, how many were found<br>
        <b>F1</b> – harmonic mean of precision and recall<br>
        <b>MCC</b> – balanced correlation score, robust to class imbalance<br><br>
        Target encoding: 0 = malignant, 1 = benign.</div>""", unsafe_allow_html=True)
    st.markdown("**Loaded models**: " + ", ".join(models.keys()))
    missing_models = [n for n in MODEL_FILES if n not in models]
    if missing_models:
        st.warning("Model files not found for: " + ", ".join(missing_models))

st.markdown('<div class="footer">Designed and developed by Yaythish Kannaa G S · BITS ID 2026AJ05374 · Machine Learning Assignment 2</div>',
            unsafe_allow_html=True)
