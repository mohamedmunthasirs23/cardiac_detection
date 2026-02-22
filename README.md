# 🫀 Advanced Cardiac Abnormality Detection System
**AI-Powered Real-Time ECG Monitoring & Explainable Diagnosis**

---

## 🎯 Overview
This project is an automated healthcare diagnostic system that detects and classifies cardiac abnormalities from ECG signals in real-time. It leverages Explainable AI (XAI) to provide clinicians with interpretable insights behind every prediction.

## 🚀 Key Features
- **Real-Time Monitoring**: Live ECG streaming via WebSockets with sub-second latency.
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Visual heatmaps showing which parts of the ECG signal influenced the model.
  - **SHAP Values**: Global and local feature importance reports from the actual ML model.
- **Lightweight ML**: Modern Random Forest classifier using 16 extracted cardiometric features (Fast & Efficient).
- **Comprehensive Analysis**: Detects Normal, Arrhythmia, MI, and other abnormalities.
- **Professional Reporting**: One-click PDF report generation for clinical documentation.
- **RBAC**: Secure Role-Based Access Control (Admin vs. Viewer).

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### 2. Setup (Windows/PowerShell)
```powershell
# Setup environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```powershell
python app/main_advanced.py
```
Navigate to **http://localhost:5000** in your browser.

## 📁 Project Structure
- `app/`: Flask web application (Backend & UI).
- `src/`: Core signal processing, feature extraction, and ML logic.
- `models/`: Pre-trained ML model files (`.pkl` and `.keras`).
- `data/`: Sample ECG signals in CSV/JSON formats.

## 📊 ML Pipeline
```
ECG Signal → Feature Extraction (16 features) → 
→ Random Forest Model → Prediction + Confidence →
→ Explainability (Grad-CAM + SHAP) → Visualization
```

---
*Developed for research and educational purposes. Not for clinical diagnosis.*
