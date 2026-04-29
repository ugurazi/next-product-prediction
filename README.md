# Santander: Next Product to Buy — Multi-Label Product Recommendation

**A machine learning system for predicting financial product adoption**, enabling data-driven cross-sell strategies with 8× cost efficiency improvement.

## Problem Statement

Banks maintain portfolios of 20+ financial products (current accounts, credit cards, loans, investments, insurance, etc.). However:

- **Mass marketing is inefficient**: Broadcasting all products to all customers achieves <1% conversion rates
- **Relevance matters**: Irrelevant offers degrade customer experience and increase churn
- **Opportunity cost is high**: Marketing budgets are wasted on low-propensity customers

**Solution**: Predict which specific product(s) each customer will adopt next month → target only high-probability customers → maximize marketing ROI and customer satisfaction.

## Dataset & Approach

### Data
- **Source**: [Kaggle Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)
- **Size**: 13.6M customer-month records × 48 features, 935K unique customers
- **Period**: 18 months (Jan 2015 – May 2016)
- **Target**: Binary labels for 24 products (0 → 1 transitions = product acquisition)
- **Validation**: Time-based split (May 2016 holdout)

### Methodology

**Multi-label Binary Classification** — One XGBoost model per product

| Component | Details |
|-----------|---------|
| **Feature Engineering** | 95 features across 7 groups: demographics, tenure, portfolio composition, lag features (t-1 product ownership), velocity metrics, temporal patterns, customer segments |
| **Key Insight** | Lag-1 product ownership is the strongest predictor (0.78–0.95 correlation within payroll bundle). Recent behavior dominates demographics. |
| **Train/Val Split** | Last 3 months for training (recency bias toward recent behavior), May 2016 for validation |
| **Hyperparameter Tuning** | Grid search per product; early stopping on validation AUC |

---

## Project Phases

| Phase | Description | Status | Artifacts |
|-------|-------------|--------|-----------|
| 1. EDA | Data profiling, missing values, product adoption trends, seasonal patterns | ✅ Complete | `01_eda.ipynb` |
| 2. Feature Engineering | Domain-driven feature selection; lag, velocity, RFM-style metrics; encoding | ✅ Complete | `02_feature_engineering.ipynb` |
| 3. Modeling | XGBoost vs LightGBM per product; cross-validation; hyperparameter tuning | ✅ Complete | `03_modeling.ipynb` |
| 4. Explainability | SHAP values, feature importance rankings, customer-level decision explanations | ✅ Complete | `04_shap_explainability.ipynb` |
| 5. Dashboard | Streamlit app for interactive scoring, calibration analysis, business impact simulation | ✅ Complete | `app.py` |
| 6. Final Report | Business impact quantification, limitations, production recommendations | ✅ Complete | `05_final_report.ipynb` |

---

## Key Results

### Model Performance

**MAP@7 (Kaggle metric)**: **0.7848**  
*Average precision when recommending top 7 products per customer*

### Per-Product AUC-ROC (XGBoost)

| Tier | Products | AUC Range |
|------|----------|-----------|
| **Excellent** | Payroll, Checking, Credit Card, Transfers, Auto-Payment | 0.9606 – 0.9896 |
| **Very Good** | Deposits, eAccount, Particulars, Young Account | 0.9409 – 0.9746 |
| **Good** | Funds, Securities, Taxes, Pensions | 0.8799 – 0.9051 |

**Average best AUC across 16 trainable products**: **0.9507**

### Model Selection
- **XGBoost wins all 16 products** (vs. LightGBM)
- **LightGBM**: Avg 0.9230 AUC
- **XGBoost**: Avg 0.9507 AUC (+3% margin)

### Business Impact (Scenario: Credit Card Campaign)

| Metric | Mass Campaign | Model-Driven | Improvement |
|--------|---------------|--------------|-------------|
| Target audience | 935K customers | 93.5K (top 10%) | 90% reduction |
| Expected conversions | 4,675 | 3,740 (80% of total) | Cost efficient |
| Cost per conversion | €6.00 | €0.75 | **8× cheaper** |
| Campaign cost | €28,050 | €2,805 | **90% savings** |

---

## Tech Stack

- **Data Processing**: pandas, numpy, polars
- **Modeling**: XGBoost, LightGBM, scikit-learn
- **Explainability**: SHAP (TreeExplainer)
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit/Power BI
- **Environment**: Python 3.10+, Jupyter

---

## Project Structure

```
santander-next-product/
├── README.md
├── requirements.txt
├── app.py                              # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA & data profiling
│   ├── 02_feature_engineering.ipynb   # Feature selection & engineering
│   ├── 03_modeling.ipynb              # Model training & comparison
│   ├── 04_shap_explainability.ipynb   # Feature importance & decision explanations
│   └── 05_final_report.ipynb          # Business impact quantification
├── data/
│   ├── raw/                           # Original Kaggle CSV (train_ver2.csv)
│   └── processed/                     # Engineered features, parquet files, Power BI exports
├── models/                            # Serialized XGBoost models (JSON format)
├── outputs/                           # Plots, confusion matrices, calibration curves
└── .venv/                             # Python environment
```

---

## Usage

### 1. Setup
```bash
git clone https://github.com/ugurazi/santander-next-product.git
cd santander-next-product

python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Download Data
Download `train_ver2.csv` from [Kaggle Santander Competition](https://www.kaggle.com/c/santander-product-recommendation) and place in `data/raw/`

### 3. Run Pipeline
```bash
# Execute notebooks in order (1→6)
jupyter notebook notebooks/01_eda.ipynb

# OR run all transformations at once
python -m notebooks.pipeline  # (if pipeline script exists)
```

### 4. Interactive Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Key Insights

1. **Payroll bundle is a gateway**: Customers with payroll direct deposit are 6–8× more likely to add pension/investment products within the next month.

2. **Recency beats history**: Product ownership in month t-1 is ~200× more predictive than demographics or tenure.

3. **Few products drive most value**: Top 5 products (Checking, Credit Card, Transfers, Deposits, Payroll) account for 75% of monthly acquisitions.

4. **Temporal seasonality matters**: Payroll products peak in Dec/Jan; credit cards peak in Q4. Temporal features improve AUC by +0.02 over baseline.

5. **Class imbalance is severe**: Most products have <1% positive rate. XGBoost's `scale_pos_weight` and early stopping are essential.

---

## Limitations & Future Work

### Limitations
- No transaction-level data (monthly snapshots only; sub-monthly behavior is invisible)
- No campaign response history (can't model offer fatigue or channel effects)
- Spain-only dataset (patterns may not transfer to other markets)
- 24-month time window (newer products may have insufficient training data)

### Next Steps
1. **A/B testing**: Validate model-driven campaigns vs. rule-based targeting in production
2. **Real-time scoring**: Deploy models as a microservice API for batch & streaming predictions
3. **Deep learning**: Explore LSTMs/Transformers for customer lifecycle sequences
4. **Reinforcement learning**: Optimize offer timing, sequencing, and channel selection
5. **Cross-bank transfer**: Adapt framework to AlternatifBank's internal CRM data

---

## Author

**Uğur Emir Azı**  
Computer Engineering @ Bartın University  
CRM & Customer Analytics Intern @ AlternatifBank  

🔗 [GitHub](https://github.com/ugurazi) | [LinkedIn](https://linkedin.com/in/uguremirazi)
