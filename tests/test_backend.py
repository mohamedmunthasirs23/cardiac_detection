# Test Backend Connection
# Run this to verify the backend API is working

import requests
import json
import numpy as np

# Generate sample ECG data
print("Generating sample ECG data...")
fs = 360
duration = 10
num_samples = fs * duration
t = np.linspace(0, duration, num_samples)

# Create ECG-like signal
ecg_signal = []
for i in range(num_samples):
    time = i / fs
    heartRate = 1.2
    signal = 0
    signal += 0.2 * np.sin(2 * np.pi * heartRate * time)
    signal += 1.0 * np.sin(2 * np.pi * heartRate * 3 * time)
    signal += 0.3 * np.sin(2 * np.pi * heartRate * 2 * time)
    signal += 0.05 * (np.random.random() - 0.5)
    ecg_signal.append(signal)

print(f"[OK] Generated {len(ecg_signal)} samples")

# Test the API
print("\nTesting backend API...")
url = "http://localhost:5000/api/predict"

payload = {
    "ecg_signal": ecg_signal
}

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n[OK] BACKEND IS WORKING!")
        print("\n" + "="*50)
        print("PREDICTION RESULTS:")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {max(result['probabilities'].values()) * 100:.1f}%")
        print(f"\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob * 100:.1f}%")
        print(f"\nKey Features:")
        for feature_name, value in result['features'].items():
            print(f"  {feature_name}: {value:.2f}")
        print("="*50)
        
        if result.get('demo_mode'):
            print("\n[WARNING]   Note: Running in DEMO mode (simulated predictions)")
            print("Install Python 3.11 + TensorFlow for real ML predictions")
    else:
        print(f"[FAILED] Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("[FAILED] Cannot connect to server!")
    print("Make sure the Flask server is running:")
    print("  python app/main_demo.py")
except Exception as e:
    print(f"[FAILED] Error: {e}")
