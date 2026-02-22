# Getting Started Guide
## Cardiac Abnormality Detection System

This guide will walk you through setting up and running the cardiac abnormality detection system step by step.

---

## 📋 Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Virtual environment** (recommended)
- **4GB+ RAM** for model training
- **Web browser** for the application interface

---

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\user1\.gemini\antigravity\scratch\cardiac-abnormality-detection
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

```powershell
.\venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install all necessary packages including TensorFlow, Flask, and signal processing libraries.

---

## 📊 Working with Data

### Option 1: Use Sample Data (Quickstart)

The easiest way to get started is to generate sample ECG data directly in the web application:

1. Skip to **Step 5: Run the Web Application**
2. Click "Generate Sample ECG" button
3. Analyze the sample data

### Option 2: Download Real ECG Dataset

#### Download MIT-BIH Database

```python
# Create a Python script or use interactive Python
python
```

```python
import wfdb
from pathlib import Path

# Create data directory
Path('data/raw/mitdb').mkdir(parents=True, exist_ok=True)

# Download sample records
records = ['100', '101', '102', '103', '104']
for record in records:
    print(f"Downloading record {record}...")
    wfdb.dl_database('mitdb', 'data/raw/mitdb', records=[record])

print("✓ Download complete!")
```

### Option 3: Use Your Own Data

Place your ECG data in CSV format in `data/raw/custom/`:

**CSV Format:**
```
value1,value2,value3,...,label
0.1,0.2,0.15,...,0
```

---

## 🧠 Training a Model

### Quick Training (with sample data)

If you have prepared your dataset:

```powershell
# Train a CNN model
python src/models/train.py --model cnn --epochs 10 --batch-size 32

# Train an LSTM model
python src/models/train.py --model lstm --epochs 10 --batch-size 32

# Train a Hybrid CNN-LSTM model
python src/models/train.py --model hybrid --epochs 10 --batch-size 32
```

### Full Training Pipeline

1. **Prepare Data:**

```python
from src.data.data_loader import ECGDataLoader
from src.data.preprocessing import ECGPreprocessor

# Load your data
loader = ECGDataLoader()
X, y = loader.load_dataset_from_csv('data/raw/custom/your_data.csv')

# Preprocess
preprocessor = ECGPreprocessor()
X_processed = preprocessor.preprocess_batch(X, target_length=1000)

# Split and save
X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_test_split(X_processed, y)
loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
```

2. **Train Model:**

```powershell
python src/models/train.py --model hybrid --epochs 50
```

3. **Evaluate Model:**

```powershell
python src/models/evaluate.py --model-path models/hybrid_YYYYMMDD_HHMMSS_best.keras
```

---

## 🌐 Running the Web Application

### Step 5: Start the Flask Server

```powershell
python app/main.py
```

You should see:
```
Cardiac Abnormality Detection System
====================================
Loading model...
Starting Flask server...
Navigate to: http://localhost:5000
```

### Step 6: Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

### Step 7: Use the Application

1. **Upload ECG Data:**
   - Click the upload area
   - Select a CSV, TXT, or JSON file with ECG data
   - Or click "Generate Sample ECG" for demo

2. **Visualize:**
   - View the ECG signal plot
   - Check signal quality

3. **Analyze:**
   - Click "Analyze ECG"
   - Wait for prediction results

4. **View Results:**
   - See diagnosis (Normal/Abnormal)
   - Check confidence scores
   - Review cardiometric features

---

## 🎯 Quick Demo Workflow

**For immediate testing without training:**

1. Activate virtual environment
2. Install dependencies
3. Run: `python app/main.py`
4. Open browser to `http://localhost:5000`
5. Click "Generate Sample ECG"
6. Click "Analyze ECG"

**Note:** Without a trained model, you'll need to train one first or the prediction will fail.

---

## 📁 Project Structure Overview

```
cardiac-abnormality-detection/
├── app/                      # Web application
│   ├── main.py              # Flask server
│   ├── templates/           # HTML files
│   └── static/              # CSS, JS
├── src/                     # Source code
│   ├── data/                # Data loading & preprocessing
│   ├── features/            # Feature extraction
│   ├── models/              # ML models
│   └── utils/               # Utilities
├── data/                    # Datasets
├── models/                  # Saved models
└── notebooks/               # Jupyter notebooks
```

---

## 🔧 Common Commands

### Training Models

```powershell
# CNN model
python src/models/train.py --model cnn --epochs 30

# LSTM model
python src/models/train.py --model lstm --epochs 30

# Hybrid model (recommended)
python src/models/train.py --model hybrid --epochs 50

# ResNet-style CNN
python src/models/train.py --model resnet --epochs 50

# GRU model
python src/models/train.py --model gru --epochs 30

# Attention-based hybrid
python src/models/train.py --model attention --epochs 50
```

### Evaluation

```powershell
python src/models/evaluate.py --model-path models/YOUR_MODEL.keras
```

### Running Web App

```powershell
python app/main.py
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```powershell
pip install tensorflow
```

### Issue: "No model loaded"

**Solution:** Train a model first:
```powershell
python src/models/train.py --model cnn --epochs 10
```

### Issue: "No processed data found"

**Solution:** Prepare your data first or use sample generation in the web app.

### Issue: Port 5000 already in use

**Solution:** Change port in `app/main.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

---

## 📚 Next Steps

1. **Explore Notebooks:** Check `notebooks/` for data exploration
2. **Customize Models:** Modify architectures in `src/models/`
3. **Add Features:** Extend feature extraction in `src/features/`
4. **Improve UI:** Customize `app/templates/` and `app/static/`
5. **Deploy:** Consider deploying to cloud platforms

---

## 🎓 Learning Resources

- **ECG Analysis:** PhysioNet tutorials
- **Deep Learning:** TensorFlow documentation
- **Signal Processing:** SciPy documentation
- **Web Development:** Flask documentation

---

## ⚠️ Important Notes

- This system is for **research and educational purposes only**
- **Not for clinical diagnosis** or medical decision-making
- Always validate results with medical professionals
- Ensure data privacy and compliance with regulations

---

## 🤝 Need Help?

- Check the main `README.md` for detailed information
- Review code comments and docstrings
- Explore example notebooks
- Check dataset documentation in `data/README.md`

---

**Happy Analyzing! 🫀**
