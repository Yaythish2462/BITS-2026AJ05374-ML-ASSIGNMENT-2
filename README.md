<div align="center">

# 🧠 AIML Model Studio

### Breast Cancer Classification · Machine Learning Assignment 2

[![Live App](https://img.shields.io/badge/🚀_Live_App-Open_on_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bits-2026aj05374-ml-assignment-2.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-5_models-F7931E?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Breast_Cancer_Wisconsin-0e9f91)

| | |
|---|---|
| 👤 **Student** | Yaythish Kannaa G S |
| 🆔 **BITS ID** | 2026AJ05374 |
| 🎓 **Programme** | M.Tech AIML, BITS Pilani WILP |

**🔗 Live app: [bits-2026aj05374-ml-assignment-2.streamlit.app](https://bits-2026aj05374-ml-assignment-2.streamlit.app/)**

</div>

---

## 📑 Contents

1. [Problem statement](#a-problem-statement)
2. [Dataset description](#b-dataset-description)
3. [Links](#c-links)
4. [Models and evaluation metrics](#d-models-used-and-evaluation-metrics)
5. [How to use the app](#-how-to-use-the-app)
6. [Application features](#-application-features)
7. [Repository structure](#-repository-structure)
8. [Run locally](#-run-locally)

---

## a. Problem statement

Build and compare five supervised classification models on a public dataset, evaluate each with **Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC)**, and demonstrate the evaluation through an interactive Streamlit application. The app loads saved estimators and evaluates labeled test CSV files **without retraining**.

## b. Dataset description

| Item | Details |
|---|---|
| **Dataset** | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (available through `sklearn.datasets.load_breast_cancer`) |
| **Task** | Binary classification of diagnostic samples into malignant (0) and benign (1) |
| **Size** | 569 instances, 30 numeric predictor features, plus one target column |
| **Input features** | Ten cell-nucleus characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each as mean, standard error and worst-value measurements |
| **Split** | Stratified 80/20 train/test split with `random_state=42`. The held-out test set has 114 instances, and all five models use the same split |
| **Target column** | `target` (0 = malignant, 1 = benign) |
| **Preprocessing** | `StandardScaler` fitted within pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes use the raw numeric inputs |

> ⚠️ **Scope:** This is an educational benchmark, not a clinical diagnostic tool.

## c. Links

| Resource | Link |
|---|---|
| 🚀 **Live Streamlit app** | https://bits-2026aj05374-ml-assignment-2.streamlit.app/ |
| 💻 **GitHub repository** | _Add your public GitHub repository URL after pushing this project._ |

## d. Models used and evaluation metrics

Metrics are computed on the same stratified held-out test split. AUC uses class-1 (benign) scores. Precision, recall and F1 also treat benign (1) as the positive class.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 |     0.9861 | 0.9623 |
| Decision Tree          |     0.9123 | 0.9157 |      0.9559 |   0.9028 |     0.9286 | 0.8174 |
| K-Nearest Neighbors    |     0.9561 | 0.9788 |      0.9589 |   0.9722 |     0.9655 | 0.9054 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 |     0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9950 |      0.9861 |   0.9861 |     0.9861 | 0.9623 |

### 🔍 Observations on model performance

| ML Model | Observation about model performance |
|---|---|
| **Logistic Regression** | A scaled linear baseline. It provides a useful reference for a problem with informative numeric measurements; check its precision/recall balance rather than accuracy alone. |
| **Decision Tree** | A non-linear, rule-based model that is easy to interpret. Its performance can vary with tree depth and may be more sensitive to overfitting than regularized models. |
| **K-Nearest Neighbors** | A distance-based method; feature scaling is applied because the measurements use different numeric ranges. Performance depends on neighborhood size and local class structure. |
| **Gaussian Naive Bayes** | A probabilistic baseline that assumes conditional independence and Gaussian feature distributions within each class. Correlated diagnostic features can limit this assumption. |
| **Support Vector Machine** | An RBF-kernel SVM with scaled inputs can model non-linear boundaries. Probability estimates are enabled to calculate AUC. |
| 🏆 **Overall winner for this split** | **Logistic Regression** has the highest F1 Score (0.9861) and the highest AUC (0.9954) on this test split. A different metric may select a different model, so the winner depends on the evaluation objective. |

---

## 🧭 How to use the app

**👉 Open the app: https://bits-2026aj05374-ml-assignment-2.streamlit.app/**

> 💤 Streamlit Community Cloud puts idle apps to sleep. If you see a "wake up" screen, click the button and wait a few seconds.

### Quick start (4 steps)

| Step | What to do |
|:---:|---|
| **1** | **Open the link above.** A welcome pop-up explains the app. Click **✅ Got it — take me to the app** to reach the Overview page. |
| **2** | **Pick a classifier** from **Active classifier** in the left sidebar (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes or SVM). |
| **3** | **Choose your data.** The bundled test set loads automatically, so you can start right away. To use your own file, upload a CSV in the sidebar (see the format below). |
| **4** | **Explore the pages** from the sidebar navigation and read the metrics, charts and predictions. |

### 🗺️ Pages at a glance

| Page | What you get |
|---|---|
| 🏠 **Overview** | Six metric cards, the confusion matrix with a plain-language error summary, and ROC curves for all models |
| 📊 **Model Comparison** | Heat-shaded metrics table, a "best overall" pick, bar chart with a metric picker, radar chart, and a CSV download |
| 🔍 **Diagnostics** | Classification report, precision–recall curve, score distribution, and a downloadable list of misclassified rows |
| 🧪 **Predict** | Adjust the 30 features with sliders for a single sample (with every model's vote), or score a whole file and download the predictions |
| 🗂️ **Data Explorer** | Data preview, summary statistics, per-feature distributions by class, and a correlation heatmap |
| ℹ️ **About** | Short guide to the pages and the metrics |

### 🎛️ Sidebar controls

- **Active classifier:** the model used across the app.
- **Decision threshold:** moves the cut-off for models that output probabilities. Lower it to label more samples benign; raise it to be stricter.
- **🌙 Dark mode:** switches between the light and dark theme.
- **Upload test CSV / Download sample test CSV:** bring your own data, or grab the sample file to see the expected format.
- **❓ How to use:** reopens the welcome guide at any time.

### 📄 Uploading your own CSV

Your file must contain:

- All **30 feature columns** with the same names as in the sample file (for example `mean radius`, `worst perimeter`).
- A column named **`target`** with values **0** (malignant) or **1** (benign). Without it the evaluation pages are unavailable, but the **Predict** page still works for batch predictions.
- Numeric values only. Extra columns are ignored.

> 💡 **Tip:** Click **Download sample test CSV** in the sidebar, edit it, and upload it back to see exactly the right format.

### 🧪 Try these

1. Open **Model Comparison** and see which model ranks first overall.
2. Go to **Diagnostics** and read the misclassified rows for the Decision Tree.
3. Open **Predict**, load a row, nudge a few sliders and watch all five models' predictions change.
4. Drag the **Decision threshold** and see how precision and recall trade off on the Overview page.

---

## ✨ Application features

- 🎉 Animated welcome guide that opens on first load, with a **How to use** button to reopen it
- 🌗 Light and dark theme toggle
- 🧭 Multi-page navigation: Overview, Model Comparison, Diagnostics, Predict, Data Explorer, About
- 📤 Upload a labeled test CSV, or use the bundled test split
- 🧠 Select any of the five saved classifiers
- 📏 Accuracy, AUC, Precision, Recall, F1 and MCC for every model
- 🆚 Side-by-side comparison with table, bar chart and radar chart
- 📉 Confusion matrix, ROC curves, precision–recall curve and score distribution
- 🎚️ Adjustable decision threshold
- 🔬 Single-sample prediction with sliders and batch prediction with CSV download
- 🗂️ Data explorer with statistics, distributions and correlations
- ⬇️ Download comparison metrics, misclassified rows, predictions and the sample test CSV
- 🔒 The app does **not** retrain models on uploaded data

## 📁 Repository structure

```text
ML_Assignment_2_Yaythish/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
└── submission_report.pdf
```

## 💻 Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
---

<div align="center">

Designed and developed by **Yaythish Kannaa G S** · BITS ID 2026AJ05374 · Machine Learning Assignment 2

</div>
