# 🫀 Advanced Cardiac Abnormality Detection System
**AI-Powered Real-Time ECG Monitoring & Explainable Diagnosis**

---

## 🎯 Overview
This project is an automated healthcare diagnostic system that detects and classifies cardiac abnormalities from ECG signals in real-time. Designed to assist medical professionals, it leverages Explainable AI (XAI) to provide interpretable insights behind every prediction. The system not only displays real-time telemetry but also offers a comprehensive patient management dashboard, cloud data persistence, and an integrated AI assistant.

## 🧠 Machine Learning Architecture & Algorithms
The core classification engine uses an **Advanced Soft-Voting Ensemble Model** that combines multiple robust machine learning algorithms:
- **Random Forest Classifier**: Handles non-linear relationships and provides feature importance (carries the highest voting weight).
- **Gradient Boosting Classifier**: Minimizes bias and optimizes prediction accuracy sequentially.
- **Extra Trees Classifier**: Reduces variance and prevents overfitting.

The ensemble's outputs are calibrated using **CalibratedClassifierCV** (Sigmoid method) to provide reliable probability estimates, confidence scores, and uncertainty metrics across four classifications:
- Normal
- Arrhythmia
- Myocardial Infarction (MI)
- Other Abnormalities

## 📈 Statistical Feature Extraction
Instead of relying solely on raw signal data, the processing pipeline extracts **over 50 advanced cardiometric and statistical features** to feed the ensemble model:
- **Time Domain Features**: Signal mean, median, peak-to-peak, standard deviation, skewness, and kurtosis.
- **Peak Detection & HRV (Heart Rate Variability)**: R-peak detection, RR intervals, RMSSD, pNN50, and estimated Heart Rate (BPM).
- **Frequency Domain Features**: FFT-based power spectrum analysis across VLF, LF, and HF bands, LF/HF ratio, and spectral entropy.
- **Wavelet Features**: Low-pass (approximation) and high-pass (detail) filter coefficients.
- **Morphological & Nonlinear Features**: Zero crossings, mean crossing rates, signal complexity, QRS width estimation, sample entropy, and detrended fluctuation analysis.

## 🚀 Key System Features
- **Real-Time ECG Monitoring**: Live ECG streaming via WebSockets with continuous telemetry and sub-second latency.
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Visual heatmaps highlighting the specific regions of the ECG signal that influenced the model.
  - **SHAP Values**: Feature importance reports explaining the specific physiological metrics driving the prediction.
- **Advanced Patient Interface**: A state-of-the-art UI featuring a patient selector, historical risk trend charts, real-time risk gauges, color-coded probability bars, and comprehensive patient records.
- **AI Assistant Widget**: A floating, interactive chat interface providing users with immediate AI-driven help and diagnostic context directly within the application.
- **Cloud Database Integration**: Secure user registration, authentication, and data persistence powered by **MongoDB Atlas**.
- **Professional Reporting**: One-click PDF report generation for clinical documentation.
- **RBAC**: Secure Role-Based Access Control distinguishing Admin and Viewer privileges.

## 🛠️ Development Journey & Recent Enhancements
Throughout the development lifecycle, this project has undergone extensive improvements, evolving from a baseline script to a production-ready application. Recent milestones include:
- **UI/UX Overhaul**: Completely redesigned the dashboard to eliminate spacing issues, add responsive components (risk gauges, trend charts), and introduce a patient history timeline.
- **AI Assistant Integration**: Built and embedded a responsive, floating chat widget to assist clinicians seamlessly without leaving the dashboard.
- **Cloud Infrastructure & Deployment**: Migrated local data persistence to **MongoDB Atlas**, resolving complex cloud login issues and ensuring newly registered users can successfully authenticate on deployed platforms like **Render**.
- **System Stability & Environment**: Debugged Windows environment anomalies, resolved tricky Unicode encoding crashes, eliminated port conflicts, and successfully finalized remote Git version control pushes.

## ⚙️ Quick Start

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)
- MongoDB Atlas account (for cloud database)

### 2. Setup (Windows/PowerShell)
```powershell
# Setup environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory based on `.env.example` and add your MongoDB Atlas URI and other necessary configuration keys.

### 4. Run the App
```powershell
python app/main_advanced.py
# or if using the safety wrapper:
python app_safe.py
```
Navigate to **http://localhost:5000** in your browser.

## 📁 Project Structure
- `app/`: Flask web application routing, backend configurations, MongoDB integration, and UI templates.
- `src/`: Core ECG signal processing, advanced feature extraction, and model ensembles.
- `models/`: Pre-trained ML model files (`.pkl` and `.keras`).
- `data/`: Sample ECG signals in CSV/JSON formats.

---
*Developed for research and educational purposes. Not for clinical diagnosis.*
