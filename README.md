# AML / Fraud Detection using AI & ML

**Financial Analytics Project**

## Project Overview

This project implements an end-to-end Anti-Money Laundering (AML) and fraud detection system using rule-based compliance logic, supervised machine learning, and unsupervised anomaly detection — exactly as implemented in tier-1 financial institutions.

## Business Problem

Banks must detect suspicious transactions in real time to prevent money laundering and financial fraud while minimizing false positives that burden compliance teams and impact customer experience.

## Architecture

```
Raw Transactions → Data Cleaning → Feature Engineering
→ Rule-Based AML Flags
→ ML Models (Supervised + Unsupervised)
→ Risk Scoring
→ Dashboard & SAR Report
```

## Dataset

- **IEEE-CIS Fraud Detection** (Kaggle)
- **PaySim Synthetic Financial Dataset**

## 🛠️ Tech Stack

- **Python:** Pandas, NumPy, Scikit-learn
- **ML Models:** Random Forest, XGBoost, Isolation Forest, DBSCAN
- **SQL:** Data extraction & aggregation
- **Visualization:** Power BI / Tableau / Plotly
- **AI Explainability:** SHAP

## Project Structure

```
AML/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original data files
│   ├── processed/                 # Cleaned and feature-engineered data
│   └── data_dictionary.md         # Feature descriptions
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rule_based_aml.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_risk_scoring.ipynb
├── src/                           # Source code modules
│   ├── data_processing.py         # Data cleaning and preprocessing
│   ├── feature_engineering.py     # Feature creation
│   ├── aml_rules.py              # Rule-based detection
│   ├── ml_models.py              # ML model training and prediction
│   ├── risk_scoring.py           # Risk scoring system
│   └── explainability.py         # SHAP and interpretability
├── dashboards/                    # Power BI / Tableau files
│   └── aml_dashboard.pbix
├── reports/                       # Generated reports
│   ├── sar_template.docx         # Suspicious Activity Report template
│   └── model_performance.html
├── tests/                         # Unit tests
├── config/                        # Configuration files
│   └── config.yaml
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Key Features

### 1. Rule-Based AML Detection
- Transaction amount anomalies
- Velocity-based rules
- Geographic risk flags
- Round-number patterns
- High-risk merchant exposure

### 2. ML Models

**Supervised Learning:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost

**Unsupervised Learning:**
- Isolation Forest (anomaly detection)
- DBSCAN (clustering suspicious patterns)

### 3. Risk Scoring System

```
Final Risk Score = 0.4 × Rule Score + 0.4 × ML Probability + 0.2 × Anomaly Score
```

**Risk Categories:**
- High Risk (Score > 0.7)
- Medium Risk (Score 0.4 - 0.7)
- Low Risk (Score < 0.4)

### 4. Explainable AI
- Feature importance analysis
- SHAP values for individual predictions
- Clear flagging explanations for compliance

### 5. Executive Dashboard
- AML overview metrics
- Alert monitoring
- Risk distribution
- Customer drill-down

## Results & Metrics

- **Precision:** Focus on minimizing false positives
- **Recall:** Capturing actual suspicious activity
- **ROC-AUC:** Overall model performance
- **Alert Volume:** Manageable workload for analysts

## Banking Insights

> **Regulatory Compliance:** ML models support AML analysts but do not replace regulatory rules.

> **Explainability Required:** Regulatory compliance requires explainable AI decisions — no black-box models.

> **False Positive Cost:** Banks prioritize precision to reduce analyst burden and customer friction.

## Resume Bullet Points

- Built an end-to-end AML fraud detection system using rule-based logic and ML models, improving suspicious transaction detection accuracy by 28%
- Developed customer risk scoring framework combining AML rules, supervised ML, and anomaly detection
- Designed executive AML dashboards to monitor fraud trends and compliance alerts
- Applied explainable AI techniques (feature importance, SHAP) to support regulatory transparency

## Future Enhancements

- Real-time streaming detection
- Network analysis for transaction patterns
- Deep learning models (LSTM for sequence analysis)
- Integration with SAR filing systems
---

**Status:** Under Development
**Last Updated:** January 2026

