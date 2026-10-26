# AIML Model Studio — Machine Learning Assignment 2

**Student:** Yaythish Kannaa G S  
**BITS ID:** 2026AJ05374  
**Programme:** M.Tech AIML, BITS Pilani WILP

## a. Problem statement
Build and compare five supervised classification models on a public dataset, evaluate each with Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and demonstrate evaluation through an interactive Streamlit application. The app loads saved estimators and evaluates labeled test CSV files without retraining.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (available through `sklearn.datasets.load_breast_cancer`).
- **Task:** Binary classification of diagnostic samples into malignant (0) and benign (1).
- **Size:** 569 instances and 30 numeric predictor features, plus one target column.
- **Input features:** Ten cell-nucleus characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each represented by mean, standard error, and worst-value measurements.
- **Split:** Stratified 80/20 train/test split with `random_state=42`. The held-out test set contains 114 instances. All five models use the same split.
- **Target column:** `target` (0 = malignant, 1 = benign).
- **Preprocessing:** StandardScaler is fitted within pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes use the raw numeric inputs.
- **Scope:** This is an educational benchmark, not a clinical diagnostic tool.

## c. GitHub Repository Link
**Repository URL:** Add your public GitHub repository URL after creating and pushing this project.  
**Live Streamlit App:** Add the deployed Streamlit Community Cloud URL after deployment.  
These links are intentionally left as placeholders; no external repository or app has been published by this package.

## d. Models used and evaluation metrics
Metrics are computed on the same stratified held-out test split. AUC uses class-1 (benign) scores. Precision, recall and F1 also treat benign (1) as the positive class.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 |     0.9861 | 0.9623 |
| Decision Tree          |     0.9123 | 0.9157 |      0.9559 |   0.9028 |     0.9286 | 0.8174 |
| K-Nearest Neighbors    |     0.9561 | 0.9788 |      0.9589 |   0.9722 |     0.9655 | 0.9054 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 |     0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.995  |      0.9861 |   0.9861 |     0.9861 | 0.9623 |

### Observations on model performance

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | A scaled linear baseline. It provides a useful reference for a problem with informative numeric measurements; check its precision/recall balance rather than accuracy alone. |
| Decision Tree | A non-linear, rule-based model that is easy to interpret. Its performance can vary with tree depth and may be more sensitive to overfitting than regularized models. |
| K-Nearest Neighbors | A distance-based method; feature scaling is applied because the measurements use different numeric ranges. Performance depends on neighborhood size and local class structure. |
| Gaussian Naive Bayes | A probabilistic baseline that assumes conditional independence and Gaussian feature distributions within each class. Correlated diagnostic features can limit this assumption. |
| Support Vector Machine | An RBF-kernel SVM with scaled inputs can model non-linear boundaries. Probability estimates are enabled to calculate AUC. |
| Overall winner for this split | **Logistic Regression** has the highest F1 Score (0.9861) on this test split. **Logistic Regression** has the highest AUC (0.9954). A different metric may select a different model, so the winner depends on the evaluation objective. |

## Application features
- Upload a labeled test CSV.
- Select a saved classifier.
- View Accuracy, AUC, Precision, Recall, F1 and MCC.
- Compare all five models on the same uploaded test data.
- View confusion matrix and classification report.
- Preview test data and download computed comparison metrics.
- Download the bundled sample test CSV.
- The app does not retrain models on uploaded data.

## Repository structure
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

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload all project files and the `model/` directory.
2. In Streamlit Community Cloud, choose **New app**, select the repository and branch, and set the main file to `app.py`.
3. Deploy, open the app URL, test the bundled CSV, and then test a second labeled CSV if available.
4. Replace the GitHub and live-app placeholders in this README and the submission PDF with the actual URLs.
5. Capture one genuine screenshot showing the assignment running in BITS Virtual Lab and insert it in the submission PDF.

## Submission checklist
- [ ] Push all source files, requirements, README, test CSV and five saved models to GitHub.
- [ ] Verify the GitHub repository link.
- [ ] Deploy and verify the Streamlit app link.
- [ ] Capture the required BITS Virtual Lab execution screenshot.
- [ ] Update links and add the screenshot to the final submission PDF.
- [ ] Export/submit one PDF containing repository link, app link, screenshot, and README content.
