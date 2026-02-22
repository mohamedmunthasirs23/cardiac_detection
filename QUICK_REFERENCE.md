# Quick Reference - Cardiac Abnormality Detection

## 🚀 Common Commands (PowerShell)

### Setup (One-time)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```powershell
# Make sure venv is activated first
.\venv\Scripts\activate

# Run web application
python app/main.py

# Open browser to: http://localhost:5000
```

### Training Models
```powershell
# Train CNN model (fast, good baseline)
python src/models/train.py --model cnn --epochs 30

# Train LSTM model (temporal patterns)
python src/models/train.py --model lstm --epochs 30

# Train Hybrid model (best accuracy - RECOMMENDED)
python src/models/train.py --model hybrid --epochs 50

# Train with custom settings
python src/models/train.py --model hybrid --epochs 100 --batch-size 64
```

### Evaluating Models
```powershell
# Evaluate a trained model
python src/models/evaluate.py --model-path models/hybrid_20260127_230000_best.keras
```

### Data Preparation
```powershell
# Download sample MIT-BIH data (run in Python)
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb', records=['100'])"
```

## 📝 PowerShell vs Bash

**PowerShell uses `;` instead of `&&` for command chaining:**

❌ Wrong (Bash syntax):
```bash
command1 && command2 && command3
```

✅ Correct (PowerShell):
```powershell
command1; command2; command3
```

## 🎯 Quick Testing Workflow

1. **Activate environment:**
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Run web app:**
   ```powershell
   python app/main.py
   ```

3. **In browser (http://localhost:5000):**
   - Click "Generate Sample ECG"
   - Click "Analyze ECG"
   - View results!

## 🔧 Troubleshooting

### "python not found"
```powershell
# Use full path or check Python installation
py -m venv venv
```

### "Module not found"
```powershell
# Make sure venv is activated
.\venv\Scripts\activate
# Reinstall dependencies
pip install -r requirements.txt
```

### "Port 5000 in use"
Edit `app/main.py` line 143:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Check if packages installed
```powershell
pip list
# Should show tensorflow, flask, numpy, etc.
```

## 📊 Model Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| CNN | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Quick baseline |
| LSTM | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | Temporal patterns |
| Hybrid | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | Production (recommended) |
| ResNet | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | Complex patterns |
| GRU | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | Lighter LSTM |
| Attention | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | Interpretability |

## 🎨 Web Interface Features

- **Upload Methods:**
  - Drag & drop files
  - Click to browse
  - Generate sample ECG

- **Supported Formats:**
  - CSV (comma or newline separated)
  - TXT (one value per line)
  - JSON (`{"signal": [...]}`)

- **Results Display:**
  - Diagnosis with confidence
  - Probability bars for all classes
  - Key cardiometric features
  - Visual indicators

## 📁 Important Files

| File | Purpose |
|------|---------|
| `app/main.py` | Web server |
| `src/models/train.py` | Training script |
| `src/models/evaluate.py` | Evaluation script |
| `requirements.txt` | Dependencies |
| `GETTING_STARTED.md` | Detailed guide |
| `README.md` | Project overview |

## 💡 Tips

- **Always activate venv** before running commands
- **Start with hybrid model** for best results
- **Use sample ECG** for quick testing
- **Check model path** when evaluating
- **Monitor training** with TensorBoard (logs in models/logs/)

## 🔗 Useful Links

- MIT-BIH Database: https://physionet.org/content/mitdb/
- TensorFlow Docs: https://www.tensorflow.org/
- Flask Docs: https://flask.palletsprojects.com/

---

**Quick Start Reminder:**
```powershell
.\venv\Scripts\activate
python app/main.py
# Open: http://localhost:5000
```
