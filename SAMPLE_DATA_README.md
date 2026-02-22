# Sample ECG Data Files

I've created 3 sample ECG data files for you to test the application:

## 📁 Available Sample Files

### 1. **sample_ecg_normal.csv**
- **Format:** CSV (comma-separated values)
- **Samples:** 3,600 data points
- **Duration:** 10 seconds
- **Sampling Rate:** 360 Hz
- **Type:** Normal ECG pattern
- **Use:** Drag and drop this file into the web interface

### 2. **sample_ecg_normal.json**
- **Format:** JSON with metadata
- **Samples:** 3,600 data points
- **Contains:**
  - `signal`: Array of ECG values
  - `sampling_rate`: 360 Hz
  - `duration`: 10 seconds
  - `patient_id`: SAMPLE_001
- **Type:** Normal ECG pattern
- **Use:** Test JSON file upload

### 3. **sample_ecg_abnormal.txt**
- **Format:** Plain text (one value per line)
- **Samples:** 3,600 data points
- **Duration:** 10 seconds
- **Type:** Abnormal ECG pattern (irregular)
- **Use:** Test abnormal detection

---

## 🚀 How to Use These Files

### Method 1: Drag & Drop
1. Open http://localhost:5000 in your browser
2. Drag any of these files and drop them on the upload area
3. The ECG will be visualized automatically
4. Click "Analyze ECG" to see the prediction

### Method 2: Click to Browse
1. Click on the upload area
2. Select one of the sample files
3. View the ECG visualization
4. Click "Analyze ECG"

---

## 📊 Expected Results

### For Normal ECG Files (CSV, JSON):
- **Prediction:** Normal
- **Confidence:** ~85%
- **Heart Rate:** ~72 bpm
- **Features:** Regular rhythm, normal intervals

### For Abnormal ECG File (TXT):
- **Prediction:** Arrhythmia or Other Abnormality
- **Confidence:** ~70%
- **Heart Rate:** Irregular
- **Features:** Abnormal rhythm detected

---

## 📝 File Format Examples

### CSV Format (sample_ecg_normal.csv):
```
0.123456
0.234567
0.345678
...
```

### JSON Format (sample_ecg_normal.json):
```json
{
  "signal": [0.123456, 0.234567, 0.345678, ...],
  "sampling_rate": 360,
  "duration": 10,
  "patient_id": "SAMPLE_001"
}
```

### TXT Format (sample_ecg_abnormal.txt):
```
0.123456
0.234567
0.345678
...
```

---

## 🎯 Quick Test

**Try this now:**

1. **Start the server** (if not already running):
   ```powershell
   .\venv\Scripts\python.exe app/main_demo.py
   ```

2. **Open browser:** http://localhost:5000

3. **Drag and drop** `sample_ecg_normal.csv` onto the upload area

4. **Click "Analyze ECG"**

5. **See the results!** 🎉

---

## 💡 Creating Your Own Data

You can create your own ECG data files:

### CSV Format:
- One value per line, or comma-separated
- Values should be numeric (floats)
- Typical range: -2.0 to 2.0

### JSON Format:
```json
{
  "signal": [array of numbers]
}
```
or
```json
{
  "ecg_signal": [array of numbers]
}
```

### TXT Format:
- One value per line
- Numeric values only

---

## 📍 File Locations

All sample files are in the project root directory:
```
C:\Users\user1\.gemini\antigravity\scratch\cardiac-abnormality-detection\
├── sample_ecg_normal.csv
├── sample_ecg_normal.json
└── sample_ecg_abnormal.txt
```

---

## ⚠️ Note

These are **synthetic ECG signals** generated for testing purposes. They simulate realistic ECG patterns but are not real patient data.

**Current Mode:** DEMO - Predictions are simulated
**For Real ML:** Install Python 3.11 + TensorFlow

---

**Happy Testing! 🫀💙**
