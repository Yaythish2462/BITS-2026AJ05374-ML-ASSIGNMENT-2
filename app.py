from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

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

st.set_page_config(page_title="AIML Model Studio | Assignment 2", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#16233a; --muted:#65748b; --blue:#3157d5; --teal:#0e9f91; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f5f8ff 0%,#f8fbff 45%,#eef8f7 100%); color:var(--ink); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1400px; }
.hero { background:linear-gradient(120deg,#172554 0%,#2549a8 58%,#0f766e 100%); padding:2rem 2.2rem; border-radius:24px; color:white; box-shadow:0 16px 42px rgba(28,55,120,.18); margin-bottom:1.1rem; }
.hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; opacity:.78; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2.25rem; line-height:1.15; margin:.45rem 0 .55rem 0; color:#fff; }
.hero p { color:#e2eaff; font-size:1rem; margin:0; max-width:760px; }
.identity { background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.24); border-radius:15px; padding:1rem 1.1rem; margin-top:1.2rem; }
.identity small { color:#dbeafe; display:block; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
.identity strong { display:block; color:#fff; font-size:1rem; margin-top:.18rem; }
.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:#172554; font-size:1.35rem; margin:.65rem 0 .2rem 0; }
.section-sub { color:#6b7890; margin-bottom:.85rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e3eaf6; padding:1rem 1.1rem; border-radius:16px; box-shadow:0 5px 18px rgba(25,50,90,.045); }
div[data-testid="stMetricLabel"] { color:#687894; }
div[data-testid="stMetricValue"] { color:#172554; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#eaf0ff; border-radius:10px 10px 0 0; padding:.6rem 1rem; }
.stTabs [aria-selected="true"] { background:#3157d5 !important; color:white !important; }
div[data-testid="stDataFrame"] { border:1px solid #e3eaf6; border-radius:12px; overflow:hidden; }
div.stButton > button { border-radius:10px; font-weight:700; }
.note { background:#eef5ff; border-left:4px solid #3157d5; border-radius:10px; padding:.8rem 1rem; color:#334766; }
.footer { color:#8491a7; text-align:center; font-size:.8rem; padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">BITS Pilani · M.Tech AIML · Machine Learning Assignment 2</div>
  <h1>Model Studio</h1>
  <p>An interactive benchmark for five classical classification algorithms, with transparent metrics, model comparison, and test-set diagnostics.</p>
  <div class="identity">
    <div class="hero-kicker">Student profile</div>
    <div style="display:flex;gap:2.5rem;flex-wrap:wrap;margin-top:.55rem">
      <div><small>Student name</small><strong>Yaythish Kannaa G S</strong></div>
      <div><small>BITS ID</small><strong>2026AJ05374</strong></div>
      <div><small>Dataset</small><strong>Breast Cancer Wisconsin (Diagnostic)</strong></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()
if not models:
    st.error("No saved models were found. Ensure the model/ directory is included in the repository.")
    st.stop()

sample_path = ROOT / "test_data.csv"
with st.sidebar:
    st.markdown("## Workspace")
    st.caption("Saved models · No retraining in this app")
    selected_model = st.selectbox("Choose a classifier", list(models.keys()))
    st.divider()
    st.markdown("### Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"], help="CSV must include all 30 input features and the target column named 'target'.")
    if sample_path.exists():
        st.download_button("Download sample test CSV", sample_path.read_bytes(), "test_data.csv", "text/csv", use_container_width=True)
    st.markdown("---")
    st.caption("Target encoding: 0 = malignant · 1 = benign")

@st.cache_data
def read_sample():
    return pd.read_csv(sample_path)

try:
    df = pd.read_csv(uploaded) if uploaded is not None else read_sample()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

st.markdown('<div class="section-title">Evaluation dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Use the provided held-out test split or upload your own labeled test data. Uploaded rows are evaluated only; models are not retrained.</div>', unsafe_allow_html=True)

if TARGET not in df.columns:
    st.error("The uploaded CSV must contain the target column named 'target'.")
    st.stop()

required_features = list(models[selected_model].feature_names_in_) if hasattr(models[selected_model], "feature_names_in_") else None
# Pipelines may expose feature_names_in_; if not, derive expected columns from the bundled test data.
if required_features is None:
    required_features = [c for c in read_sample().columns if c != TARGET]
missing = [c for c in required_features if c not in df.columns]
extra = [c for c in df.columns if c not in required_features + [TARGET]]
if missing:
    st.error(f"Missing required feature columns ({len(missing)}): {', '.join(missing[:8])}" + ("…" if len(missing) > 8 else ""))
    st.stop()
if df.empty:
    st.warning("The uploaded CSV contains no rows.")
    st.stop()
try:
    X_eval = df[required_features].apply(pd.to_numeric, errors="raise")
    y_eval = pd.to_numeric(df[TARGET], errors="raise").astype(int)
except Exception as exc:
    st.error(f"Feature and target columns must contain numeric values: {exc}")
    st.stop()
if not set(y_eval.unique()).issubset({0,1}):
    st.error("Target values must be 0 (malignant) or 1 (benign).")
    st.stop()

model = models[selected_model]
pred = model.predict(X_eval)
try:
    scores = model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, scores) if y_eval.nunique() == 2 else float("nan")
except Exception:
    try:
        scores = model.decision_function(X_eval)
        auc = roc_auc_score(y_eval, scores) if y_eval.nunique() == 2 else float("nan")
    except Exception:
        auc = float("nan")

acc = accuracy_score(y_eval, pred)
precision = precision_score(y_eval, pred, zero_division=0)
recall = recall_score(y_eval, pred, zero_division=0)
f1 = f1_score(y_eval, pred, zero_division=0)
mcc = matthews_corrcoef(y_eval, pred)

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, label, val in [
    (c1,"Accuracy",acc),(c2,"AUC",auc),(c3,"Precision",precision),
    (c4,"Recall",recall),(c5,"F1 Score",f1),(c6,"MCC",mcc)
]:
    col.metric(label, "N/A" if np.isnan(val) else f"{val:.4f}")

st.caption(f"Evaluated {len(df):,} rows · Selected model: {selected_model} · Positive class for precision/recall/F1/AUC: benign (1)")
tab1,tab2,tab3,tab4 = st.tabs(["Model comparison","Confusion matrix","Classification report","Data preview"])

with tab1:
    st.markdown("### All classifiers on the same uploaded test set")
    rows = []
    for name, clf in models.items():
        p = clf.predict(X_eval)
        try:
            s = clf.predict_proba(X_eval)[:,1]
            a = roc_auc_score(y_eval,s) if y_eval.nunique()==2 else np.nan
        except Exception:
            try:
                s = clf.decision_function(X_eval)
                a = roc_auc_score(y_eval,s) if y_eval.nunique()==2 else np.nan
            except Exception:
                a = np.nan
        rows.append({"Model":name,"Accuracy":accuracy_score(y_eval,p),"AUC":a,
                     "Precision":precision_score(y_eval,p,zero_division=0),
                     "Recall":recall_score(y_eval,p,zero_division=0),
                     "F1 Score":f1_score(y_eval,p,zero_division=0),
                     "MCC":matthews_corrcoef(y_eval,p)})
    comp = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)
    st.bar_chart(comp[["Accuracy","AUC","F1 Score","MCC"]], use_container_width=True)
    st.download_button("Download comparison metrics", comp.reset_index().to_csv(index=False), "evaluation_metrics.csv", "text/csv")

with tab2:
    st.markdown("### Confusion matrix")
    cm = confusion_matrix(y_eval, pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5.5,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{selected_model}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0,1], LABELS)
    ax.set_yticks([0,1], LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Rows are true classes; columns are predicted classes.")

with tab3:
    report = classification_report(y_eval, pred, labels=[0,1], target_names=LABELS, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)
    st.code(classification_report(y_eval, pred, labels=[0,1], target_names=LABELS, zero_division=0), language="text")

with tab4:
    st.write(f"Rows: **{df.shape[0]:,}** · Columns: **{df.shape[1]}**")
    st.dataframe(df.head(30), use_container_width=True)
    if extra:
        st.info(f"Extra columns ignored: {', '.join(extra[:8])}" + ("…" if len(extra)>8 else ""))

st.markdown('<div class="note"><b>Interpretation note:</b> This is an educational classification benchmark, not a clinical diagnostic tool. Metrics are computed on the labeled test CSV supplied in the current session.</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Designed and developed by Yaythish Kannaa G S · BITS ID 2026AJ05374 · Machine Learning Assignment 2</div>', unsafe_allow_html=True)
