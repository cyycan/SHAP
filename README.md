# 🔍 SHAP Governance Framework for ML Models

> A comprehensive, production-ready framework for implementing SHAP-based interpretability standards across ML models — covering feature attribution analysis, bias auditing, drift monitoring, and explainability best practices.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![SHAP](https://img.shields.io/badge/shap-latest-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## 📋 What's Inside

The framework is delivered as a single, self-contained Jupyter notebook structured into 8 sections:

| Section | Description |
|---------|-------------|
| **0. Setup & Data Loading** | Drop-in data interface — replace one block to use your own dataset |
| **1. Explainer Selection** | Auto-detects the optimal SHAP explainer for any model type |
| **2. Feature Attribution Analysis** | Local & global SHAP values with stability testing |
| **3. Visualisation Suite** | Waterfall, summary, dependence, heatmap, and decision plots |
| **4. Governance Checks** | Automated quality gates with configurable thresholds |
| **5. Bias & Fairness Audit** | Protected attribute contribution analysis |
| **6. Drift Monitoring** | KS-test based SHAP attribution drift detection |
| **7. Model Card Generator** | Auto-generates a Markdown explainability model card |
| **8. Production Checklist** | Deployment readiness scoring & artefact export |

---

## 🤖 Supported Model Types

| Model Type | SHAP Explainer | Models |
|------------|---------------|--------|
| 🌲 **Tree models** | `TreeExplainer` | XGBoost, LightGBM, RandomForest, GradientBoosting |
| 📈 **Linear models** | `LinearExplainer` | LogisticRegression, Ridge, Lasso |
| 🧠 **Neural networks** | `KernelExplainer` | sklearn MLP, PyTorch*, TensorFlow* |
| ⬛ **Model-agnostic** | `KernelExplainer` | Any sklearn-compatible black-box model |

*For PyTorch/TensorFlow models, use `DeepExplainer` — swap in Section 1.2*

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/shap-governance-framework.git
cd shap-governance-framework
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook shap_governance_framework.ipynb
```

### 4. Load your data

In **Section 0.3**, replace the demo data block with your own:

```python
# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Configure
TARGET_COLUMN        = 'your_target_column'
TASK_TYPE            = 'classification'       # or 'regression'
PROTECTED_ATTRIBUTES = ['age', 'gender']      # for bias auditing
MODEL_NAME           = 'my_production_model'
```

Everything else adapts automatically.

---

## ⚙️ Configuration

All governance thresholds are centralised in **Section 4.1**:

```python
GOVERNANCE_CONFIG = {
    'min_top3_coverage':             0.60,  # Top-3 features explain ≥ 60% of SHAP mass
    'max_single_feature_dominance':  0.70,  # No feature explains > 70% of SHAP mass
    'max_cv_threshold':              0.15,  # Max attribution stability (coefficient of variation)
    'min_feature_count':             3,     # Minimum meaningful features
    'trivial_shap_threshold':        0.01,  # Below this = negligible contribution
}
```

Drift monitoring threshold:
```python
DRIFT_KS_THRESHOLD = 0.10   # KS statistic above which drift is flagged
```

Bias audit threshold:
```python
BIAS_THRESHOLD = 0.05        # Flag if protected attribute explains > 5% of SHAP mass
```

---

## 📦 Output Artefacts

After running the full notebook, the following artefacts are produced:

```
.
├── MODEL_CARD.md                          # Auto-generated explainability model card
├── shap_global_importance_comparison.png  # Global SHAP importance across all models
├── shap_dependence_plots.png              # Top-3 feature dependence plots
├── shap_heatmap.png                       # Attribution heatmap across samples
├── shap_drift_monitor.png                 # SHAP drift visualisation
└── shap_artefacts/
    ├── {model}_model.pkl                  # Saved model
    ├── {model}_explainer.pkl              # Saved SHAP explainer
    ├── {model}_feature_importance.json    # Feature importance by mean |SHAP|
    └── {model}_governance.json           # Governance check results
```

---

## 🏛️ Governance Framework

### Quality Gates (Section 4)

The framework runs automated quality checks before any model goes to production:

- **Top-3 coverage** — Are explanations concentrated enough to be meaningful?
- **Single-feature dominance** — Is the model over-reliant on one feature?
- **Attribution stability** — Are SHAP values consistent across bootstrap samples?
- **SHAP additivity** — Do SHAP values correctly sum to the prediction delta?
- **Non-trivial feature count** — Does the model use a meaningful number of features?

### Bias Audit (Section 5)

Set `PROTECTED_ATTRIBUTES = ['col1', 'col2']` to audit whether sensitive attributes are disproportionately influencing predictions. Features exceeding the `BIAS_THRESHOLD` are flagged for review.

### Drift Monitoring (Section 6)

Compares SHAP value distributions between a reference window and current data using the Kolmogorov-Smirnov test. Integrate this section into your MLOps pipeline to detect silent model degradation early.

---

## 🗂️ Repository Structure

```
shap-governance-framework/
├── shap_governance_framework.ipynb   # Main notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── MODEL_CARD.md                     # Generated after first run
```

---

## 📦 Requirements

```
shap
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
scipy
joblib
ipywidgets
```

Install all with:
```bash
pip install -r requirements.txt
```

Optional for deep learning:
```bash
pip install torch torchvision    # PyTorch
pip install tensorflow           # TensorFlow / Keras
```

---

## 📖 References

- [Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [EU AI Act — Explainability Requirements](https://artificialintelligenceact.eu/)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0.pdf)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to support teams implementing explainable AI standards in production ML systems.*
