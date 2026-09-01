import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# ================= 1. Page Configuration =================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ================= 2. Load Data and Model =================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/dashboard_data.csv")
    
    # Calculate HourOfDay if missing
    if 'HourOfDay' not in df.columns and 'TransactionDT' in df.columns:
        df['HourOfDay'] = (df['TransactionDT'] // 3600) % 24
        
    # Generate mock probabilities to prevent crashes if missing
    if 'Fraud_Probability' not in df.columns:
        np.random.seed(42)
        df['Fraud_Probability'] = np.random.uniform(0.0, 1.0, size=len(df))
        
    # --- THE FIX: Convert object columns to categories globally ---
    # This ensures n_categories > 0 for XGBoost DMatrix at all times
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    return df

@st.cache_resource
def load_model():
    return joblib.load("dashboard/xgb_model.pkl")

df = load_data()
model = load_model()

# ================= 3. Sidebar Navigation =================
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", ["Overview", "Transaction Explorer", "SHAP Explainer"])

# ================= PAGE 1: OVERVIEW =================
if page == "Overview":
    st.title("Fraud Operations Overview")
    
    # Calculate Key Metrics
    total_transactions = len(df)
    total_fraud = len(df[df['isFraud'] == 1]) if 'isFraud' in df.columns else 0
    detection_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
    avg_fraud_amount = df[df['isFraud'] == 1]['TransactionAmt'].mean() if total_fraud > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Total Fraud Count", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Transaction Amount Distribution")
        if 'TransactionAmt' in df.columns and 'isFraud' in df.columns:
            fig1 = px.histogram(df, x="TransactionAmt", color="isFraud", log_y=True, nbins=50)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("TransactionAmt or isFraud column is missing from the data.")
        
    with col_chart2:
        st.subheader("Fraud Count by Hour of Day")
        if 'HourOfDay' in df.columns and 'isFraud' in df.columns:
            fraud_df = df[df['isFraud'] == 1]
            fig2 = px.histogram(fraud_df, x="HourOfDay", nbins=24)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("HourOfDay column not found in the data.")

# ================= PAGE 2: TRANSACTION EXPLORER =================
elif page == "Transaction Explorer":
    st.title("Live Transaction Explorer")
    
    min_prob = st.sidebar.slider("Filter by Minimum Fraud Probability", 0.0, 1.0, 0.0)
    filtered_df = df[df['Fraud_Probability'] >= min_prob]
    
    search_id = st.text_input("Search by exact TransactionID:")
    if search_id:
        filtered_df = filtered_df[filtered_df['TransactionID'].astype(str) == search_id]
        
    cols_to_show = ['TransactionID', 'TransactionAmt', 'HourOfDay', 'Fraud_Probability', 'isFraud']
    available_cols = [c for c in cols_to_show if c in filtered_df.columns]
    st.dataframe(filtered_df[available_cols])

# ================= PAGE 3: SHAP EXPLAINER =================
elif page == "SHAP Explainer":
    st.title("SHAP Explainer (Explainable AI)")
    st.write("Understand exactly WHY the AI flagged a transaction.")
    
    txn_id = st.text_input("Enter TransactionID to Generate Explanation:")
    
    if txn_id:
        txn_data = df[df['TransactionID'].astype(str) == txn_id]
        
        if len(txn_data) == 0:
            st.error("TransactionID not found in the dataset.")
        else:
            prob = txn_data['Fraud_Probability'].values[0]
            st.success(f"Transaction Found! Fraud Risk Probability: {prob:.4f}")
            
            # Drop unnecessary columns
            cols_to_drop = ['TransactionID', 'isFraud', 'Fraud_Probability']
            drop_cols = [c for c in cols_to_drop if c in txn_data.columns]
            features_only = txn_data.drop(columns=drop_cols)
            
            # Ensure the features align perfectly with what the model expects
            booster = model.get_booster() if hasattr(model, 'get_booster') else model
            expected_cols = booster.feature_names
            if expected_cols:
                valid_cols = [c for c in expected_cols if c in features_only.columns]
                features_only = features_only[valid_cols]
            
            try:
                # Create DMatrix with enable_categorical=True
                # The data is already correctly formatted as categorical from load_data()
                dmatrix = xgb.DMatrix(features_only, enable_categorical=True)
                
                # Extract SHAP values directly from XGBoost C++ backend
                shap_contribs = booster.predict(dmatrix, pred_contribs=True)
                shap_values_matrix = shap_contribs[:, :-1]
                base_value = shap_contribs[0, -1]
                
                # Construct SHAP explanation object
                explanation = shap.Explanation(
                    values=shap_values_matrix[0],
                    base_values=base_value,
                    data=features_only.iloc[0],
                    feature_names=features_only.columns.tolist()
                )
                
                # Render Waterfall Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating SHAP chart: {e}")
            
            # Display business context
            st.markdown("### Business Explanation")
            if prob >= 0.75:
                st.error("This transaction is classified as **CRITICAL RISK**. The red bars in the chart show the suspicious features (like unusual time or amount) that pushed the risk score very high.")
            elif prob >= 0.40:
                st.warning("This transaction is classified as **SUSPICIOUS**. Manual analyst review is recommended. Some risk factors were elevated but partially offset by normal behavior (blue bars).")
            else:
                st.info("This transaction is classified as **CLEAR (SAFE)**. The blue bars represent normal customer behavior that lowered the fraud risk score.")
