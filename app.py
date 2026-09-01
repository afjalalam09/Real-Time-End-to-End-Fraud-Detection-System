import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# 2. Load Data and Model (Cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv("dashboard/dashboard_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("dashboard/xgb_model.pkl")

df = load_data()
model = load_model()

# 3. Sidebar Navigation
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", ["Overview", "Transaction Explorer", "SHAP Explainer"])

# ================= PAGE 1: OVERVIEW =================
if page == "Overview":
    st.title("Fraud Operations Overview")
    
    # Calculate Key Metrics
    total_transactions = len(df)
    total_fraud = len(df[df['isFraud'] == 1])
    detection_rate = (total_fraud / total_transactions) * 100
    avg_fraud_amount = df[df['isFraud'] == 1]['TransactionAmt'].mean()
    
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
        fig1 = px.histogram(df, x="TransactionAmt", color="isFraud", log_y=True, nbins=50)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.subheader("Fraud Count by Hour of Day")
        fraud_df = df[df['isFraud'] == 1]
        fig2 = px.histogram(fraud_df, x="HourOfDay", nbins=24)
        st.plotly_chart(fig2, use_container_width=True)

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
    st.dataframe(filtered_df[['TransactionID', 'TransactionAmt', 'HourOfDay', 'Fraud_Probability', 'isFraud']])

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
            features_only = txn_data.drop(columns=['TransactionID', 'isFraud', 'Fraud_Probability'])
            
            # Generate SHAP values for this specific transaction
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(features_only)
            
            # Display the Waterfall Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            
            # Plain English Explanation based on probability tiers
            st.markdown("### Business Explanation")
            if prob >= 0.75:
                st.error("This transaction is classified as **CRITICAL RISK**. The red bars in the chart show the suspicious features (like unusual time or amount) that pushed the risk score very high.")
            elif prob >= 0.40:
                st.warning("This transaction is classified as **SUSPICIOUS**. Manual analyst review is recommended. Some risk factors were elevated but partially offset by normal behavior (blue bars).")
            else:
                st.info("This transaction is classified as **CLEAR (SAFE)**. The blue bars represent normal customer behavior that lowered the fraud risk score.")
