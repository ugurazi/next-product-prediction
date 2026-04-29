#  Santander: Next Product to Buy — Predictive Model

## Business Problem
Banks offer dozens of products (credit cards, loans, deposits, insurance, etc.) but marketing all products to all customers is inefficient and annoying. **Which product should we recommend to which customer, and when?**

This project builds a machine learning pipeline that predicts the next financial product a customer is likely to purchase, enabling:
- **Targeted marketing campaigns** with higher conversion rates
- **Reduced churn** by proactively offering relevant products
- **Increased cross-sell/upsell revenue** per customer

## Dataset
[Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) — 13.6M rows × 48 columns, 1.5 years of monthly customer data (Jan 2015 – May 2016) with 24 financial product ownership flags.

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1. EDA | Data profiling, missing values, distributions, product adoption trends | 🔄 In Progress |
| 2. Feature Engineering | Lag features, RFM-style metrics, product velocity, demographic encoding | ⬜ |
| 3. Modeling | XGBoost / LightGBM multi-label classification | ⬜ |
| 4. Explainability | SHAP values, feature importance, customer-level explanations | ⬜ |
| 5. Dashboard | Streamlit app for interactive predictions & insights | ⬜ |
| 6. Documentation | Final report, business impact summary | ⬜ |

## Tech Stack
- **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Modeling**: XGBoost, LightGBM
- **Explainability**: SHAP
- **Dashboard**: Streamlit
- **Environment**: Jupyter / VS Code

## Key Results
*(Will be updated as project progresses)*

## How to Run
```bash
# Clone the repo
git clone https://github.com/ugurazi/santander-next-product.git
cd santander-next-product

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (requires Kaggle account)
# Go to: https://www.kaggle.com/c/santander-product-recommendation/data
# Download train_ver2.csv.zip and extract it
# Place train_ver2.csv into data/raw/

# Alternative: using Kaggle CLI
kaggle competitions download -c santander-product-recommendation
unzip santander-product-recommendation.zip -d data/raw/

# Run notebooks in order
jupyter notebook notebooks/
```

## Author
**Uğur Emir Azı** — Computer Engineering @ Bartın University  
CRM & Customer Analytics Intern @ AlternatifBank  
[LinkedIn](https://linkedin.com/in/uguremirazi) | [GitHub](https://github.com/ugurazi)
