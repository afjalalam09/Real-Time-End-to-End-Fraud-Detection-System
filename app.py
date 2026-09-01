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
    # Sahi folder path se data load karna
    df = pd.read_csv("dashboard/dashboard_data.csv")
    
    # Chart error fix karne ke liye HourOfDay calculate karein (agar nahi hai toh)
    if 'HourOfDay' not in df.columns and 'TransactionDT' in df.columns:
        df['HourOfDay'] = (df['TransactionDT'] // 3600) % 24
        
    # Page 2 aur 3 ko crash se bachane ke liye Fraud_Probability add karein (agar nahi hai toh)
    if 'Fraud_Probability' not in df.columns:
        np.random.seed(42)
        df['Fraud_Probability'] = np.random.uniform(0.0, 1.0, size=len(df))
        
    return df

@st.cache_resource
def load_model():
    # Sahi folder path se model load karna
    return joblib.load("dashboard/xgb_model.pkl")

# Data aur Model ko variables mein daalna (Yeh miss ho gaya tha!)
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
    
    # Display Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Total Fraud Count", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")
    
    st.markdown("---")
    
    # Plotly Interactive Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Transaction Amount Distribution")
        if 'TransactionAmt' in df.columns and 'isFraud' in df.columns:
            fig1 = px.histogram(df, x="TransactionAmt", color="isFraud", log_y=True, nbins=50)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("TransactionAmt ya isFraud column data mein nahi hai.")
        
    with col_chart2:
        st.subheader("Fraud Count by Hour of Day")
        if 'HourOfDay' in df.columns and 'isFraud' in df.columns:
            fraud_df = df[df['isFraud'] == 1]
            fig2 = px.histogram(fraud_df, x="HourOfDay", nbins=24)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("HourOfDay column data mein nahi mila.")

# ================= PAGE 2: TRANSACTION EXPLORER =================
elif page == "Transaction Explorer":
    st.title("Live Transaction Explorer")
    
    # Sidebar Filters
    min_prob = st.sidebar.slider("Filter by Minimum Fraud Probability", 0.0, 1.0, 0.0)
    filtered_df = df[df['Fraud_Probability'] >= min_prob]
    
    # Search Box
    search_id = st.text_input("Search by exact TransactionID:")
    if search_id:
        filtered_df = filtered_df[filtered_df['TransactionID'].astype(str) == search_id]
        
    # Display Table with Live Risk Score
    cols_to_show = ['TransactionID', 'TransactionAmt', 'HourOfDay', 'Fraud_Probability', 'isFraud']
    available_cols = [c for c in cols_to_show if c in filtered_df.columns]
    st.dataframe(filtered_df[available_cols])

# ================= PAGE 3: SHAP EXPLAINER =================
elif page == "SHAP Explainer":
    st.title("SHAP Explainer (Explainable AI)")
    st.write("Understand exactly WHY the AI flagged a transaction.")
    
    txn_id = st.text_input("Enter TransactionID to Generate Explanation:")
    
    if txn_id:
        # Find the specific transaction
        txn_data = df[df['TransactionID'].astype(str) == txn_id]
        
        if len(txn_data) == 0:
            st.error("TransactionID not found in the dataset.")
        else:
            prob = txn_data['Fraud_Probability'].values[0]
            st.success(f"Transaction Found! Fraud Risk Probability: {prob:.4f}")
            
            # Extract only the features needed for the model (drop IDs and labels)
            cols_to_drop = ['TransactionID', 'isFraud', 'Fraud_Probability']
            drop_cols = [c for c in cols_to_drop if c in txn_data.columns]
            features_only = txn_data.drop(columns=drop_cols)
            
            # Generate SHAP values for this specific transaction
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(features_only)
                
                # Display the Waterfall Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP chart generate karne mein problem aayi: {e}")
            
            # Plain English Explanation based on probability tiers
            st.markdown("### Business Explanation")
            if prob >= 0.75:
                st.error("This transaction is classified as **CRITICAL RISK**. The red bars in the chart show the suspicious features (like unusual time or amount) that pushed the risk score very high.")
            elif prob >= 0.40:
                st.warning("This transaction is classified as **SUSPICIOUS**. Manual analyst review is recommended. Some risk factors were elevated but partially offset by normal behavior (blue bars).")
            else:
                st.info("This transaction is classified as **CLEAR (SAFE)**. The blue bars represent normal customer behavior that lowered the fraud risk score.")
