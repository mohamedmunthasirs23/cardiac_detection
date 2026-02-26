"""
Flask web application for cardiac abnormality detection - DEMO VERSION
This version works without TensorFlow for demonstration purposes.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)

# Demo mode - no model required
print("\n" + "="*60)
print("[WARNING]   RUNNING IN DEMO MODE (No TensorFlow)")
print("="*60)
print("The interface will work, but predictions are simulated.")
print("To use real ML models, install TensorFlow for Python 3.11 or lower.")
print("="*60 + "\n")


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Simulate cardiac abnormality prediction.
    Returns demo predictions for testing the interface.
    """
    try:
        data = request.get_json()
        
        if 'ecg_signal' not in data:
            return jsonify({'error': 'No ECG signal provided'}), 400
        
        # Parse ECG signal
        ecg_signal = np.array(data['ecg_signal'])
        
        if len(ecg_signal) == 0:
            return jsonify({'error': 'Empty ECG signal'}), 400
        
        # DEMO: Simulate prediction based on signal statistics
        signal_mean = np.mean(ecg_signal)
        signal_std = np.std(ecg_signal)
        signal_max = np.max(ecg_signal)
        
        # Simple heuristic for demo
        if signal_std > 0.5 and abs(signal_mean) < 0.3:
            predicted_class = 0  # Normal
            probabilities = {
                'Normal': 0.85,
                'Arrhythmia': 0.10,
                'Myocardial Infarction': 0.03,
                'Other Abnormality': 0.02
            }
        else:
            predicted_class = 1  # Arrhythmia
            probabilities = {
                'Normal': 0.15,
                'Arrhythmia': 0.70,
                'Myocardial Infarction': 0.10,
                'Other Abnormality': 0.05
            }
        
        # Simulate features
        key_features = {
            'Heart Rate': float(60 + np.random.randn() * 10),
            'Mean RR Interval': float(0.8 + np.random.randn() * 0.1),
            'SDNN (HRV)': float(45 + np.random.randn() * 10),
            'Signal Energy': float(np.sum(ecg_signal**2))
        }
        
        class_labels = {0: 'Normal', 1: 'Arrhythmia', 2: 'Myocardial Infarction', 3: 'Other Abnormality'}
        
        response = {
            'success': True,
            'prediction': class_labels[predicted_class],
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'features': key_features,
            'processed_signal': ecg_signal.tolist()[:1000],  # Limit size
            'demo_mode': True,
            'note': 'This is a simulated prediction. Install TensorFlow to use real ML models.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    info = {
        'model_name': 'Demo Mode - No Model Loaded',
        'demo_mode': True,
        'message': 'Install TensorFlow to use real ML models',
        'classes': {
            0: 'Normal',
            1: 'Arrhythmia',
            2: 'Myocardial Infarction',
            3: 'Other Abnormality'
        }
    }
    
    return jsonify(info)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': False,
        'demo_mode': True
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Cardiac Abnormality Detection System - DEMO MODE")
    print("="*60 + "\n")
    print("Starting Flask server...")
    print("Navigate to: http://localhost:5000")
    print("\n[WARNING]   Running in DEMO mode - predictions are simulated")
    print("To use real ML models:")
    print("  1. Install Python 3.11 (TensorFlow not yet available for 3.14)")
    print("  2. Or wait for TensorFlow to support Python 3.14")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
