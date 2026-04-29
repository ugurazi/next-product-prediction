"""
Phase 5: Streamlit Dashboard — Next Product to Buy
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
st.set_page_config(
    page_title="Next Product to Buy — Santander",
    page_icon="🏦",
    layout="wide"
)

PROCESSED_PATH = Path('data/processed')
MODEL_PATH = Path('models')

product_names_map = {
    'ind_cco_fin_ult1': 'Current Account',
    'ind_recibo_ult1': 'Direct Debit',
    'ind_tjcr_fin_ult1': 'Credit Card',
    'ind_nomina_ult1': 'Payroll',
    'ind_nom_pens_ult1': 'Pensions (Payroll)',
    'ind_ecue_fin_ult1': 'e-Account',
    'ind_cno_fin_ult1': 'Payroll Account',
    'ind_reca_fin_ult1': 'Taxes',
    'ind_ctop_fin_ult1': 'Particular Account',
    'ind_dela_fin_ult1': 'Long-term Deposits',
    'ind_ctma_fin_ult1': 'Más Particular',
    'ind_valo_fin_ult1': 'Securities',
    'ind_fond_fin_ult1': 'Funds',
    'ind_ctpp_fin_ult1': 'Particular Plus',
}


@st.cache_data
def load_data():
    val_df = pd.read_parquet(PROCESSED_PATH / 'val.parquet')
    with open(PROCESSED_PATH / 'feature_cols.json') as f:
        feature_cols = json.load(f)
    with open(MODEL_PATH / 'metadata.json') as f:
        metadata = json.load(f)
    return val_df, feature_cols, metadata


@st.cache_resource
def load_models(metadata):
    models = {}
    for target, info in metadata['models'].items():
        product_short = target.replace('_added', '').replace('ind_', '').replace('_ult1', '')
        try:
            if info['type'] == 'lgb':
                models[target] = ('lgb', lgb.Booster(model_file=str(MODEL_PATH / f'{product_short}_lgb.txt')))
            else:
                m = xgb.Booster()
                m.load_model(str(MODEL_PATH / f'{product_short}_xgb.json'))
                models[target] = ('xgb', m)
        except Exception as e:
            st.warning(f"Could not load model for {product_short}: {e}")
    return models


def predict_customer(customer_features, models, feature_cols):
    """Predict product probabilities for a single customer."""
    results = {}
    X = pd.DataFrame([customer_features[feature_cols].values], columns=feature_cols)
    
    for target, (model_type, model) in models.items():
        product_short = target.replace('_added', '').replace('ind_', '').replace('_ult1', '')
        product_label = product_names_map.get(f'ind_{product_short}_ult1',
                         product_names_map.get(f'ind_{product_short}', product_short))
        
        if model_type == 'lgb':
            prob = model.predict(X)[0]
        else:
            prob = model.predict(xgb.DMatrix(X))[0]
        
        results[product_label] = float(prob)
    
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


# --- MAIN APP ---
def main():
    st.title("🏦 Next Product to Buy")
    st.markdown("**Predicting which banking product a customer will adopt next**")
    
    try:
        val_df, feature_cols, metadata = load_data()
        models = load_models(metadata)
    except FileNotFoundError:
        st.error("Data files not found. Please run notebooks 02 and 03 first.")
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("", ["📊 Overview", "🔍 Customer Lookup", "📈 Model Performance"])
    
    if page == "📊 Overview":
        show_overview(val_df, metadata, models, feature_cols)
    elif page == "🔍 Customer Lookup":
        show_customer_lookup(val_df, models, feature_cols)
    elif page == "📈 Model Performance":
        show_model_performance(metadata)


def show_overview(val_df, metadata, models, feature_cols):
    st.header("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(val_df):,}")
    col2.metric("Products Modeled", f"{len(models)}")
    col3.metric("MAP@7", f"{metadata.get('map7', 'N/A'):.4f}" if isinstance(metadata.get('map7'), float) else "N/A")
    col4.metric("Features", f"{len(feature_cols)}")
    
    st.subheader("Model AUC by Product")
    auc_data = {
        product_names_map.get(
            f"ind_{k.replace('_added', '').replace('ind_', '').replace('_ult1', '')}_ult1",
            k.replace('_added', '')
        ): v['auc']
        for k, v in metadata['models'].items()
    }
    auc_df = pd.DataFrame(list(auc_data.items()), columns=['Product', 'AUC']).sort_values('AUC', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(auc_df) * 0.35)))
    colors = ['#e74c3c' if v < 0.7 else '#f39c12' if v < 0.8 else '#2ecc71' for v in auc_df['AUC']]
    ax.barh(auc_df['Product'], auc_df['AUC'], color=colors, edgecolor='white')
    ax.set_xlabel('AUC-ROC')
    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3, label='Good (0.8)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)


def show_customer_lookup(val_df, models, feature_cols):
    st.header("🔍 Customer Product Recommendations")
    
    # Customer selector
    customer_ids = val_df['ncodpers'].unique()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_id = st.selectbox("Customer ID", customer_ids[:1000], index=0)
        st.caption(f"Showing first 1,000 of {len(customer_ids):,} customers")
    
    customer = val_df[val_df['ncodpers'] == selected_id].iloc[0]
    
    with col2:
        # Customer profile
        st.markdown("**Customer Profile**")
        profile_cols = st.columns(4)
        profile_cols[0].metric("Age", f"{customer.get('age', 'N/A'):.0f}" if pd.notna(customer.get('age')) else "N/A")
        profile_cols[1].metric("Tenure (months)", f"{customer.get('antiguedad', 'N/A'):.0f}" if pd.notna(customer.get('antiguedad')) else "N/A")
        profile_cols[2].metric("Products Held", f"{customer.get('total_products_lag1', 'N/A'):.0f}" if pd.notna(customer.get('total_products_lag1')) else "N/A")
        profile_cols[3].metric("Income", f"€{customer.get('renta', 0):,.0f}" if pd.notna(customer.get('renta')) else "N/A")
    
    # Predictions
    st.subheader("Recommended Products")
    predictions = predict_customer(customer, models, feature_cols)
    
    # Top recommendations
    top_recs = list(predictions.items())[:7]
    
    for i, (product, prob) in enumerate(top_recs):
        col_rank, col_name, col_bar = st.columns([0.5, 2, 4])
        col_rank.markdown(f"**#{i+1}**")
        col_name.markdown(f"**{product}**")
        col_bar.progress(min(prob, 1.0), text=f"{prob:.1%}")
    
    # Full prediction table
    with st.expander("All Product Probabilities"):
        pred_df = pd.DataFrame(list(predictions.items()), columns=['Product', 'Probability'])
        pred_df['Probability'] = pred_df['Probability'].map('{:.4%}'.format)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)


def show_model_performance(metadata):
    st.header("📈 Model Performance Details")
    
    perf_data = []
    for target, info in metadata['models'].items():
        product_short = target.replace('_added', '').replace('ind_', '').replace('_ult1', '')
        product_label = product_names_map.get(f'ind_{product_short}_ult1',
                         product_names_map.get(f'ind_{product_short}', product_short))
        perf_data.append({
            'Product': product_label,
            'Model': info['type'].upper(),
            'AUC-ROC': info['auc'],
            'Quality': '🟢 Good' if info['auc'] >= 0.8 else '🟡 Fair' if info['auc'] >= 0.7 else '🔴 Weak'
        })
    
    perf_df = pd.DataFrame(perf_data).sort_values('AUC-ROC', ascending=False)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.subheader("Model Selection")
    st.markdown("""
    For each product, the best-performing model was selected between **LightGBM** and **XGBoost** 
    based on validation AUC-ROC. Key modeling decisions:
    
    - **Time-based split**: Train on months before April 2016, validate on May 2016
    - **Class imbalance**: Handled via `is_unbalance` (LGB) / `scale_pos_weight` (XGB)
    - **Early stopping**: 50 rounds patience to prevent overfitting
    - **Features**: Lag-1 product ownership, demographics, portfolio metrics, temporal signals
    """)


if __name__ == '__main__':
    main()
