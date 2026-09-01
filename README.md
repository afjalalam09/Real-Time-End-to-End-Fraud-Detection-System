# Real-Time End-to-End Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link.streamlit.app)

An interactive, machine-learning-powered web application designed to detect fraudulent financial transactions. This system not only predicts the probability of fraud but also utilizes Explainable AI (SHAP) to provide transparent, business-friendly reasoning for every prediction.

## 🚀 Key Features

*   **Fraud Operations Overview:** A high-level executive dashboard tracking total transactions, fraud counts, detection rates, and average fraud amounts. Includes interactive Plotly charts for transaction distributions.
*   **Live Transaction Explorer:** A dynamic data grid allowing risk analysts to filter transactions by minimum fraud probability and search for specific `TransactionID`s.
*   **Explainable AI (SHAP):** Transparency is critical in finance. By entering a `TransactionID`, the system generates a SHAP waterfall chart and a plain-English business explanation detailing exactly *why* the XGBoost model flagged a transaction as Critical Risk, Suspicious, or Safe.

## 🛠️ Tech Stack

*   **Frontend / Deployment:** Streamlit, Streamlit Community Cloud
*   **Machine Learning:** XGBoost, Scikit-Learn
*   **Explainable AI:** SHAP (SHapley Additive exPlanations)
*   **Data Processing:** Pandas, NumPy
*   **Data Visualization:** Plotly, Matplotlib, Seaborn

## 📂 Repository Structure

```text
├── app.py                              # Main Streamlit application script
├── requirements.txt                    # Python dependencies for deployment
├── fraud_detection_project.ipynb       # Original Jupyter Notebook for EDA and Model Training
├── dashboard/
│   ├── dashboard_data.csv              # Downsampled dataset for dashboard visualization
│   └── xgb_model.pkl                   # Trained XGBoost model
└── README.md                           # Project documentation
