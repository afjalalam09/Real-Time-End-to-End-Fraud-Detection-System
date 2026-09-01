# Real-Time End-to-End Fraud Detection System

A production-ready, interactive machine learning web application built to detect fraudulent transactions using **XGBoost**, featuring **Explainable AI (SHAP)** and deployed live on **Streamlit Cloud**.

🔗 **Live Demo:** [View Live Application](https://real-time-end-to-end-fraud-detection-system-crciqmln9zaiighqbj.streamlit.app/)

---

## 🚀 Key Features

1. **Fraud Operations Overview:**
   - Real-time KPI metrics displaying total transactions, fraud counts, detection rates, and average fraud amounts.
   - Interactive Plotly charts for transaction amount distribution and hourly fraud patterns.

2. **Live Transaction Explorer:**
   - Filter transactions based on custom fraud probability sliders.
   - Instantly search for specific transactions using exact `TransactionID`.

3. **Explainable AI (SHAP Explainer):**
   - Deep-dive into individual predictions to understand *why* the AI flagged a specific transaction.
   - Robust fail-safe architecture ensuring seamless fallback visualization and business risk categorization.

---

## 🛠️ Tech Stack

- **Core Language:** Python
- **Machine Learning:** XGBoost, Scikit-Learn
- **Explainable AI:** SHAP (SHapley Additive exPlanations)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web Framework & Deployment:** Streamlit Cloud, GitHub

---

## 📁 Project Structure

```text
Real-Time-End-to-End-Fraud-Detection-System/
│
├── app.py                     # Main Streamlit web application
├── fraud_detection_project.ipynb # Jupyter Notebook containing data preprocessing and model training pipeline
├── requirements.txt           # Python dependencies for deployment
└── dashboard/
    ├── dashboard_data.csv     # Optimized production transaction sample dataset
    └── xgb_model.pkl          # Trained XGBoost classification model
