# Real-Time End-to-End Fraud Detection System 🛡️

A production-ready, interactive machine learning web application built to detect fraudulent transactions using **XGBoost**. The system features an **Explainable AI (SHAP)** module to interpret model decisions and is deployed live on **Streamlit Cloud** for real-time monitoring.

🔗 **Live Deployment:** [View Real-Time Dashboard](https://real-time-end-to-end-fraud-detection-system-crciqmln9zaiighqbj.streamlit.app/)

---

## 🧠 Data Optimization & Deployment Strategy

**Handling Big Data in the Cloud:**
To ensure a highly responsive, real-time dashboard while strictly adhering to GitHub's file size constraints and Streamlit Cloud's memory limits, this deployed application utilizes an optimized **5,000-row production sample** (`dashboard_data.csv`).

* **Uncompromised Model Integrity:** The XGBoost classification model (`xgb_model.pkl`) was trained offline on the *complete, massive historical dataset*. The model retains 100% of its learned intelligence and predictive power.
* **Real-Time Feed Simulation:** In enterprise environments, fraud dashboards do not load terabytes of historical data into the UI. Instead, they process recent batches. The 5,000 rows in this repository act as a "live transaction feed," demonstrating the model's inference capabilities in a realistic, low-latency production setting.

---

## 🚀 Key Features

### 1. Fraud Operations Overview
* **Live KPI Tracking:** Monitors total transaction volume, total fraud count, real-time detection rate, and average fraudulent transaction amounts.
* **Interactive Analytics:** Utilizes `Plotly` to render logarithmic transaction amount distributions and hourly fraud patterns, allowing analysts to spot macro-trends instantly.

### 2. Live Transaction Explorer
* **Dynamic Filtering:** Adjust minimum fraud probability thresholds via sidebar sliders to isolate high-risk transactions.
* **Precision Search:** Instantly query the system using an exact `TransactionID` to pull up full details and live risk scores for manual review.

### 3. Explainable AI (SHAP Explainer)
* **Local Interpretability:** Generates dynamic SHAP Waterfall plots to explain exactly *why* a specific transaction was flagged, quantifying the positive or negative impact of individual features (e.g., transaction amount, time of day).
* **Fail-Safe Architecture:** Engineered with graceful degradation. If a specific transaction's categorical data causes a SHAP matrix mismatch, the system seamlessly falls back to displaying the **Global Feature Importance** chart, ensuring zero downtime and continuous operation.
* **Business Logic Translation:** Automatically translates raw probability scores into plain English risk categories (Critical Risk, Suspicious, or Clear/Safe) for non-technical stakeholders.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3
* **Machine Learning:** XGBoost, Scikit-Learn
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Data Engineering:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Matplotlib
* **Deployment & CI/CD:** Streamlit Cloud, GitHub

---

## 📁 Project Architecture

```text
Real-Time-End-to-End-Fraud-Detection-System/
│
├── app.py                        # Main Streamlit web application & UI logic
├── requirements.txt              # Production dependencies for Streamlit Cloud
├── dashboard/
│   ├── dashboard_data.csv        # Optimized 5,000-row sample for live inference
│   └── xgb_model.pkl             # Pre-trained XGBoost model 
└── fraud_detection_project.ipynb # Original notebook for EDA and Model Training
