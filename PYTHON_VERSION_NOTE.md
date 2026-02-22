# ⚠️ Important: Python Version Compatibility

## Current Situation

You're using **Python 3.14**, which is very new. Unfortunately, **TensorFlow** (the deep learning library) doesn't support Python 3.14 yet.

## Two Options to Run This Project:

### Option 1: Demo Mode (Works Now!) ✅

Run the application in **demo mode** to see the interface and functionality:

```powershell
python app/main_demo.py
```

**What works:**
- ✅ Beautiful web interface
- ✅ ECG visualization
- ✅ File upload and sample generation
- ✅ Simulated predictions (for testing UI)

**What doesn't work:**
- ❌ Real machine learning predictions (needs TensorFlow)

### Option 2: Install Python 3.11 for Full Functionality 🎯

For real ML predictions, you need Python 3.11 or 3.12:

1. **Download Python 3.11:**
   - Visit: https://www.python.org/downloads/
   - Download Python 3.11.x (latest 3.11 version)
   - Install it

2. **Create new virtual environment with Python 3.11:**
   ```powershell
   # Use Python 3.11 explicitly
   py -3.11 -m venv venv311
   
   # Activate it
   .\venv311\Scripts\activate
   
   # Install all packages (including TensorFlow)
   pip install -r requirements.txt
   
   # Run the full application
   python app/main.py
   ```

## Quick Start (Demo Mode)

Since the basic packages are installing now, you can run the demo version:

```powershell
# Wait for installation to complete, then:
python app/main_demo.py
```

Then open: **http://localhost:5000**

## What's the Difference?

| Feature | Demo Mode | Full Mode (Python 3.11) |
|---------|-----------|------------------------|
| Web Interface | ✅ Yes | ✅ Yes |
| ECG Visualization | ✅ Yes | ✅ Yes |
| File Upload | ✅ Yes | ✅ Yes |
| ML Predictions | ❌ Simulated | ✅ Real CNN/LSTM/Hybrid |
| Model Training | ❌ No | ✅ Yes |
| Feature Extraction | ✅ Yes | ✅ Yes |

## Recommendation

**For now:** Use demo mode to explore the interface and understand the workflow.

**For production:** Install Python 3.11 to get full ML capabilities.

## Check Your Python Version

```powershell
python --version
```

## Alternative: Wait for TensorFlow Update

TensorFlow will eventually support Python 3.14. Check:
- https://github.com/tensorflow/tensorflow/releases

---

**Current Status:** Installing packages for demo mode... ⏳
