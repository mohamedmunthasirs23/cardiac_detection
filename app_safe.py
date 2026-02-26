import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

# --- Safe Mode Dashboard for Unstable Windows Environments ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'emergency_safe_key_123'

USERS = {
    'admin': {
        'password': 'admin',
        'role': 'Administrator',
        'name': 'Dr. Admin',
        'access_level': 'admin',
    },
    'doctor': {
        'password': 'password',
        'role': 'Cardiologist',
        'name': 'Dr. Smith',
        'access_level': 'admin',
    }
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        
        user = USERS.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            session['name'] = user['name']
            session['access_level'] = user['access_level']
            return redirect(url_for('index'))
        flash('Invalid credentials. (Try admin/admin)', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index_advanced.html')

@app.route('/api/me')
def me():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({
        'username': session.get('username'),
        'role': session.get('role'),
        'name': session.get('name'),
        'access_level': session.get('access_level')
    })

@app.route('/api/stats')
def stats():
    return jsonify({
        'success': True,
        'total_patients': 120,
        'total_analyses': 450,
        'risk_breakdown': {'Low': 300, 'Medium': 100, 'High': 50},
        'ml_model_active': False,
        'demo_mode': True,
        'db_backend': 'Mock (Safe Mode)'
    })

@app.route('/api/patients')
def patients():
    return jsonify({
        'success': True,
        'patients': [
            {'patient_id': 'PAT-001', 'name': 'John Doe', 'age': 45, 'gender': 'Male'},
            {'patient_id': 'PAT-002', 'name': 'Jane Smith', 'age': 32, 'gender': 'Female'}
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    # Heuristic Safe Prediction
    return jsonify({
        'success': True,
        'prediction': 'Normal',
        'predicted_class': 0,
        'probabilities': {'Normal': 0.85, 'Arrhythmia': 0.10, 'Myocardial Infarction': 0.05},
        'confidence': 0.85,
        'uncertainty': 0.15,
        'features': {'Heart Rate': 72, 'SDNN (HRV)': 45.2, 'Signal Energy': 1200.5},
        'gradcam_heatmap': [0.1] * 100,
        'shap_values': {'Heart Rate': 0.4, 'HRV Metrics': 0.3, 'QRS Morphology': 0.3},
        'processed_signal': [0.0] * 100,
        'demo_mode': True,
        'risk_level': 'Low',
        'recommendations': ["Maintain healthy lifestyle", "Schedule routine checkup in 6 months"]
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'mode': 'Windows Safe Mode Bypass'})


if __name__ == '__main__':
    print("="*60)
    print("STARTING CARDIAC MONITOR - WINDOWS SAFE MODE")
    print("Bypassing system DLL conflicts (OpenMP/SocketIO/Scipy)")
    print("="*60)
    print("Navigate to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
