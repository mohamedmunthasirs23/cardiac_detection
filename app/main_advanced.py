"""
Advanced Flask application with WebSocket support for real-time ECG monitoring.
Includes Explainable AI features and professional-grade capabilities.
"""

# ── eventlet monkey-patch MUST be first (required for gunicorn eventlet worker) ─
import eventlet
eventlet.monkey_patch()



import json
import io
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

sys.path.append(str(Path(__file__).parent.parent))

_USE_MONGO = False
try:
    from app.mongodb_database import (
        init_database as _mongo_init,
        get_all_patients, get_patient, create_patient,
        update_patient, delete_patient, save_analysis, get_patient_analyses,
        get_stats as mongo_get_stats,
        get_database, get_client, MONGO_URI,
        users_col
    )
    
    # Enable MongoDB if URI is provided, even if the first ping is slow
    if MONGO_URI:
        _USE_MONGO = True
        init_database = _mongo_init
        try:
            # Optional probe — we'll print a warning but not disable Mongo if it fails
            get_client().admin.command('ping')
            print("✅ MongoDB connected — data stored in MongoDB Atlas")
        except Exception as _probe_err:
            print(f"⚠️  MongoDB connected but ping failed ({_probe_err!s:.60}) — will retry on request")
    else:
        raise ValueError("No MONGO_URI provided")

except Exception as _mongo_err:
    print(f"⚠️  MongoDB unavailable ({_mongo_err!s:.80}) — falling back to SQLite")
    from app.database import init_database, get_db, get_session, Patient, ECGAnalysis

# ── ML model loading ──────────────────────────────────────────────────────────
DEMO_MODE = True
ml_model = None

try:
    from src.models.ensemble_model import AdvancedECGEnsemble

    ml_model = AdvancedECGEnsemble()
    model_path = Path(__file__).parent.parent / 'models' / 'ensemble_ecg_model.pkl'

    if model_path.exists():
        ml_model.load(model_path)
        DEMO_MODE = False
        print("✅ Advanced Ensemble model loaded!")
    else:
        raise FileNotFoundError("Ensemble model file not found")

except Exception:
    # Fallback: lightweight scikit-learn model
    try:
        from src.models.lightweight_model import LightweightECGClassifier

        ml_model = LightweightECGClassifier()
        lw_path = Path(__file__).parent.parent / 'models' / 'lightweight_ecg_model.pkl'

        if lw_path.exists():
            ml_model.load(lw_path)
            DEMO_MODE = False
            print("✅ Lightweight ML model loaded!")
        else:
            print("⚠️  No model file found — using heuristic predictions")
    except Exception as exc:
        print(f"⚠️  Could not load any ML model: {exc}")
        ml_model = None


# ── Flask / SocketIO setup ────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('CARDIAC_SECRET_KEY', 'cardiac-monitor-secret-2024')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Users (demo credentials — replace with DB-backed auth in production) ──────
# access_level: 'admin' = full edit/create/delete  |  'viewer' = read-only
USERS = {
    'admin':   {'password': 'admin123',   'role': 'Administrator', 'name': 'Admin User',    'access_level': 'admin'},
    'doctor':  {'password': 'doctor123',  'role': 'Cardiologist',  'name': 'Dr. Smith',     'access_level': 'admin'},
    'analyst': {'password': 'analyst123', 'role': 'ECG Analyst',   'name': 'Jane Analyst',  'access_level': 'admin'},
    'viewer':  {'password': 'viewer123',  'role': 'Viewer',        'name': 'View Only',     'access_level': 'viewer'},
}

def login_required(f):
    """Decorator that redirects unauthenticated requests to /login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Please sign in to access the dashboard.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Decorator: reject with 403 JSON if the current user is not an admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        if session.get('access_level') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied — admin role required'}), 403
        return f(*args, **kwargs)
    return decorated

print("\n" + "=" * 60)
print("🫀 ADVANCED CARDIAC ABNORMALITY DETECTION SYSTEM")
print("=" * 60)
print("✨ Features: Real-time WebSocket | Explainable AI | PDF Reports")
if DEMO_MODE:
    print("\n⚠️  Heuristic mode (ML model not loaded)")
else:
    print("\n✅ ML mode — Ensemble / scikit-learn model active")
print("=" * 60 + "\n")

# ── Initialising database ───────────────────────────────────────────────────
try:
    print("📊 Initialising database …")
    init_database()
except Exception as init_err:
    if _USE_MONGO:
        print(f"⚠️  MongoDB initialization failed: {init_err}")
        print("🔄 Falling back to local SQLite database...")
        _USE_MONGO = False
        # Re-import SQLite implementation
        from app.database import (
            init_database as sqlite_init,
            get_db, get_session, Patient, ECGAnalysis
        )
        init_database = sqlite_init
        init_database()
    else:
        print(f"❌ Database initialization failed: {init_err}")
        raise

# ── Load persisted users from MongoDB ────────────────────────────────────────
# ── Local User Persistence (Security Fallback) ────────────────────────────────
_USERS_FILE = Path(__file__).parent.parent / 'data' / 'local_users.json'

def _save_local_users():
    """Save the USERS dictionary to a local JSON file as a fallback."""
    try:
        _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_USERS_FILE, 'w') as f:
            json.dump(USERS, f, indent=4)
    except Exception as e:
        print(f"⚠️  Could not save local user fallback: {e}")

def _load_registered_users():
    """On startup, read users from MongoDB or local JSON fallback."""
    # 1. Load from local JSON first (ensures speed and baseline stability)
    if _USERS_FILE.exists():
        try:
            with open(_USERS_FILE, 'r') as f:
                data = json.load(f)
                USERS.update(data)
                print(f"👥 Loaded {len(data)} user(s) from local cache")
        except Exception as e:
            print(f"⚠️  Could not load local users: {e}")

    # 2. Try loading from MongoDB
    if not _USE_MONGO:
        return
    try:
        _db = get_database()
        cursor = _db.users.find({}, {'_id': 0})
        count = 0
        for doc in cursor:
            username = doc.get('username')
            if username:
                USERS[username] = {
                    'password':     doc.get('password', ''),
                    'role':         doc.get('role', 'User'),
                    'name':         doc.get('name', username),
                    'access_level': doc.get('access_level', 'viewer'),
                }
                count += 1
        
        # Ensure index on users
        _db.users.create_index([('username', 1)], unique=True)
        
        if count:
            print(f"👥 Sync'd {count} user(s) from MongoDB Atlas")
    except Exception as e:
        print(f"⚠️  Could not sync users from MongoDB: {e}")

_load_registered_users()

active_sessions: dict = {}

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_LABELS = {0: 'Normal', 1: 'Arrhythmia', 2: 'Myocardial Infarction', 3: 'Other Abnormality'}
MIN_SIGNAL_LENGTH = 100


# ── Helpers ───────────────────────────────────────────────────────────────────
def _compute_fft_band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Return integrated power in a frequency band using the FFT periodogram."""
    n = len(signal)
    if n < 2:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.sum(power[mask]))


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply a simple box-car smoothing window."""
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def generate_gradcam_simulation(ecg_signal: np.ndarray) -> np.ndarray:
    """
    Simulate Grad-CAM importance scores aligned with signal peaks.
    Smoothed with a Gaussian-like window for realistic visualisation.
    """
    importance = np.abs(ecg_signal)
    # Gaussian smoothing via convolution
    window = min(51, len(importance) // 4 * 2 + 1)
    if window > 2:
        x = np.linspace(-3, 3, window)
        kernel = np.exp(-0.5 * x ** 2)
        kernel /= kernel.sum()
        importance = np.convolve(importance, kernel, mode='same')
    rng = importance.max() - importance.min()
    if rng > 0:
        importance = (importance - importance.min()) / rng
    return importance


def _compute_risk_level(predicted_class: int, confidence: float) -> str:
    """Determine risk level from predicted class and model confidence."""
    if predicted_class == 0:
        return 'Low'
    if predicted_class == 2:           # MI → always High
        return 'High'
    # Arrhythmia or Other
    if confidence >= 0.75:
        return 'High'
    elif confidence >= 0.50:
        return 'Medium'
    return 'Low'


def generate_recommendations(predicted_class: int, features: dict) -> list[str]:
    """Generate clinical recommendations based on prediction and features."""
    hr = features.get('Heart Rate', 70)

    if predicted_class == 0:  # Normal
        recs = ["Continue regular monitoring", "Maintain healthy lifestyle"]
        if hr > 100:
            recs.append("Heart rate slightly elevated — consider stress evaluation")
        elif hr < 50:
            recs.append("Heart rate is low — consult a physician if symptomatic")
        recs.append("Schedule routine checkup in 6 months")
        return recs

    if predicted_class == 1:  # Arrhythmia
        return [
            "Consult a cardiologist promptly",
            "Consider Holter monitoring for 24–48 hours",
            "Avoid strenuous physical activity until assessed",
            "Monitor heart rate and rhythm regularly",
            "Review current medications with your physician",
        ]

    if predicted_class == 2:  # MI
        return [
            "⚠️ Seek emergency medical attention immediately",
            "Call emergency services (112/911) if chest pain is present",
            "Do NOT drive yourself — wait for medical assistance",
            "Avoid all physical exertion",
            "Administer aspirin if not contraindicated",
        ]

    # Other
    return [
        "Consult a cardiologist for further evaluation",
        "Perform additional diagnostic tests (echo, stress test)",
        "Consider hospitalisation if symptoms worsen",
    ]


def _extract_key_features(ecg_signal: np.ndarray) -> dict:
    """
    Derive key display features directly from the ECG signal using FFT
    (no random noise injected).
    """
    n = len(ecg_signal)
    estimated_fs = n / 10.0       # assume 10-second strip

    # ── R-peak detection (local maxima above adaptive threshold) ──────────
    # A true R-peak must:
    #   1. Exceed mean + 0.6*std (adaptive threshold)
    #   2. Be a local maximum in a ±min_sep window
    mean = float(np.mean(ecg_signal))
    std  = float(np.std(ecg_signal))
    min_sep = max(1, int(estimated_fs * 0.33))   # min 330 ms between peaks (≈ 180 bpm max)
    threshold = mean + 0.6 * std

    candidate_idx = np.where(ecg_signal > threshold)[0]
    r_peaks = []
    for idx in candidate_idx:
        lo = max(0, idx - min_sep)
        hi = min(n, idx + min_sep + 1)
        if ecg_signal[idx] == np.max(ecg_signal[lo:hi]):
            if not r_peaks or (idx - r_peaks[-1]) >= min_sep:
                r_peaks.append(idx)

    num_peaks = len(r_peaks)
    if num_peaks >= 2:
        # Mean RR interval across all detected peaks
        rr_intervals_samples = np.diff(r_peaks)
        mean_rr_samples = float(np.mean(rr_intervals_samples))
        mean_rr_seconds = mean_rr_samples / estimated_fs
        heart_rate = float(np.clip(60.0 / mean_rr_seconds, 30.0, 200.0))
        mean_rr = mean_rr_seconds
    elif num_peaks == 1:
        # Only one peak found — estimate from signal length
        heart_rate = float(np.clip(num_peaks / 10.0 * 60.0, 30.0, 200.0))
        mean_rr = 60.0 / heart_rate
    else:
        heart_rate = 70.0    # safe fallback
        mean_rr = 60.0 / 70.0

    # HRV time-domain
    diffs = np.diff(ecg_signal)
    sdnn = float(np.std(ecg_signal) * 1000)            # proxy in ms
    rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000) if len(diffs) > 0 else 0.0
    pnn50 = (float(np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100)
             if len(diffs) > 0 else 0.0)

    # Frequency-domain (real FFT)
    lf_power = _compute_fft_band_power(ecg_signal, estimated_fs, 0.04, 0.15)
    hf_power = _compute_fft_band_power(ecg_signal, estimated_fs, 0.15, 0.40)
    lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 1.0

    return {
        'Heart Rate': heart_rate,
        'Mean RR Interval': mean_rr,
        'SDNN (HRV)': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'LF Power': lf_power,
        'HF Power': hf_power,
        'LF/HF Ratio': float(lf_hf_ratio),
        'Signal Energy': float(np.sum(ecg_signal ** 2)),
        'QRS Width': float(0.08 + std * 0.01),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if 'username' in session:
        return redirect(url_for('index'))   # already logged in

    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = USERS.get(username)
        # ── Fallback: check MongoDB directly if not in memory (sync for Render workers) ──
        if not user and _USE_MONGO:
            try:
                doc = users_col().find_one({'username': username})
                if doc:
                    user = {
                        'password':     doc.get('password', ''),
                        'role':         doc.get('role', 'User'),
                        'name':         doc.get('name', username),
                        'access_level': doc.get('access_level', 'viewer'),
                    }
                    USERS[username] = user  # cache it
            except Exception as e:
                print(f"⚠️  Login DB lookup failed: {e}")

        if user and user['password'] == password:
            session['username']     = username
            session['role']         = user['role']
            session['name']         = user['name']
            session['access_level'] = user['access_level']
            if remember:
                app.permanent_session_lifetime = __import__('datetime').timedelta(days=7)
                session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    name = session.get('name', 'User')
    session.clear()
    flash(f'Goodbye, {name}! You have been signed out.', 'success')
    return redirect(url_for('login'))


# Role → display title and access level
_ROLE_MAP = {
    'administrator':  ('Administrator',  'admin'),
    'doctor':         ('Doctor',         'admin'),
    'cardiologist':   ('Cardiologist',   'admin'),
    'ecg analyst':    ('ECG Analyst',    'admin'),
    'analyst':        ('ECG Analyst',    'admin'),
    'nurse':          ('Nurse',          'admin'),
    'researcher':     ('Researcher',     'viewer'),
    'student':        ('Student',        'viewer'),
    'viewer':         ('Viewer',         'viewer'),
}

@app.route('/register', methods=['POST'])
def register():
    """Handle new user sign-up."""
    first     = request.form.get('first_name', '').strip()
    last      = request.form.get('last_name',  '').strip()
    role_raw  = request.form.get('role',       '').strip().lower()
    username  = request.form.get('username',   '').strip().lower()
    password  = request.form.get('password',   '')
    confirm   = request.form.get('confirm',    '')

    # Validate
    if not all([first, last, role_raw, username, password, confirm]):
        flash('All fields are required.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if password != confirm:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if len(password) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if username in USERS:
        flash(f'Username "{username}" is already taken. Please choose another.', 'error')
        return redirect(url_for('login') + '?tab=signup')

    role_title, access_level = _ROLE_MAP.get(role_raw, ('User', 'viewer'))
    full_name = f'{first} {last}'
    if role_title in ('Doctor', 'Cardiologist'):
        full_name = f'Dr. {last}'

    new_user = {
        'password':     password,
        'role':         role_title,
        'name':         full_name,
        'access_level': access_level,
    }
    USERS[username] = new_user
    _save_local_users()

    # Persist to MongoDB so the account survives restarts
    if _USE_MONGO:
        try:
            users_col().update_one(
                {'username': username},
                {'$set': {**new_user, 'username': username}},
                upsert=True,
            )
        except Exception as e:
            print(f'⚠️  Could not persist user to MongoDB: {e}')

    flash(
        f'Account created! Your username is <strong>{username}</strong>. '
        f'You can now sign in.',
        'success'
    )
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Render main dashboard."""
    return render_template('index_advanced.html')


@app.route('/api/me')
@login_required
def me():
    """Return current user info including access level."""
    return jsonify({
        'username':     session.get('username'),
        'role':         session.get('role'),
        'name':         session.get('name'),
        'access_level': session.get('access_level', 'viewer'),
    })


@app.route('/api/predict', methods=['POST'])
@login_required
@admin_required
def predict():
    """
    Advanced ECG prediction with explainability features.
    """
    try:
        data = request.get_json(silent=True) or {}

        if 'ecg_signal' not in data:
            return jsonify({'error': 'No ECG signal provided'}), 400

        ecg_signal = np.asarray(data['ecg_signal'], dtype=np.float64)

        if ecg_signal.ndim != 1:
            return jsonify({'error': 'ECG signal must be a 1-D array'}), 400
        if len(ecg_signal) < MIN_SIGNAL_LENGTH:
            return jsonify({
                'error': f'Signal too short — minimum {MIN_SIGNAL_LENGTH} samples required'
            }), 400

        # ── ML prediction ──────────────────────────────────────────────────
        if ml_model is not None and not DEMO_MODE:
            ml_result = ml_model.predict(ecg_signal)
            predicted_class = ml_result['predicted_class']
            probabilities = ml_result['probabilities']
            confidence = ml_result['confidence']
            uncertainty = ml_result['uncertainty']
            feature_importance = ml_result['feature_importance']
        else:
            # Heuristic fallback
            std = float(np.std(ecg_signal))
            mean = float(np.mean(ecg_signal))
            if std > 0.5 and abs(mean) < 0.3:
                predicted_class, confidence = 0, 0.85
                probabilities = {'Normal': 0.85, 'Arrhythmia': 0.10,
                                 'Myocardial Infarction': 0.03, 'Other Abnormality': 0.02}
            else:
                predicted_class, confidence = 1, 0.70
                probabilities = {'Normal': 0.15, 'Arrhythmia': 0.70,
                                 'Myocardial Infarction': 0.10, 'Other Abnormality': 0.05}
            uncertainty = 1.0 - confidence
            feature_importance = [0.1] * 10

        # ── Derived features ────────────────────────────────────────────────
        key_features = _extract_key_features(ecg_signal)

        # ── XAI ────────────────────────────────────────────────────────────
        gradcam = generate_gradcam_simulation(ecg_signal)

        feature_names = ['Heart Rate', 'HRV Metrics', 'QRS Morphology',
                         'Frequency Features', 'Signal Energy', 'Other']
        total_imp = sum(feature_importance[:6]) or 1.0
        shap_values = {
            name: float(feature_importance[i] / total_imp)
            for i, name in enumerate(feature_names)
        }

        risk_level = _compute_risk_level(predicted_class, confidence)
        recommendations = generate_recommendations(predicted_class, key_features)

        response = {
            'success': True,
            'prediction': CLASS_LABELS.get(predicted_class, 'Unknown'),
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'features': key_features,
            'gradcam_heatmap': gradcam.tolist(),
            'shap_values': shap_values,
            'processed_signal': ecg_signal.tolist()[:1000],
            'demo_mode': DEMO_MODE,
            'ml_model_active': ml_model is not None and not DEMO_MODE,
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level,
            'recommendations': recommendations,
        }

        # ── Persist to MongoDB ────────────────────────────────────────────
        try:
            doc = save_analysis({
                'patient_id':   'PATIENT_001',
                'prediction':   response['prediction'],
                'confidence':   float(confidence),
                'uncertainty':  float(uncertainty),
                'risk_level':   risk_level,
                'heart_rate':   key_features.get('Heart Rate'),
                'hrv_sdnn':     key_features.get('SDNN (HRV)'),
                'signal_length': len(ecg_signal),
                'probabilities': probabilities,
                'features':     key_features,
                'recommendations': recommendations,
            })
            response['analysis_id'] = doc['_id']
        except Exception as db_err:
            print(f'⚠️  MongoDB save error: {db_err}')

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ── WebSocket ──────────────────────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'session_id': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    active_sessions.pop(request.sid, None)


@socketio.on('start_monitoring')
def handle_start_monitoring(data):
    patient_id = data.get('patient_id', 'PATIENT_001')
    active_sessions[request.sid] = {
        'patient_id': patient_id,
        'start_time': datetime.now().isoformat(),
        'status': 'active',
    }
    emit('monitoring_started', {
        'patient_id': patient_id,
        'status': 'active',
        'message': 'Real-time monitoring initiated',
    })


@socketio.on('stream_ecg')
def handle_ecg_stream(data):
    """Process an ECG chunk and return a live prediction."""
    try:
        ecg_chunk = np.asarray(data['ecg_data'], dtype=np.float64)
        patient_id = data.get('patient_id', 'PATIENT_001')

        # Use the ML model when available
        if ml_model is not None and not DEMO_MODE and len(ecg_chunk) >= MIN_SIGNAL_LENGTH:
            result_ml = ml_model.predict(ecg_chunk)
            prediction = result_ml['prediction']
            confidence = float(result_ml['confidence'])
        else:
            chunk_std = float(np.std(ecg_chunk))
            chunk_mean = float(np.mean(ecg_chunk))
            if chunk_std > 0.5 and abs(chunk_mean) < 0.3:
                prediction, confidence = 'Normal', 0.85
            else:
                prediction, confidence = 'Abnormal', 0.70

        n = len(ecg_chunk)
        estimated_fs = max(n / 10.0, 1.0)
        threshold = np.mean(ecg_chunk) + 0.5 * np.std(ecg_chunk)
        peaks = int(np.sum(ecg_chunk > threshold))
        if peaks > 1:
            rr_seconds = (n / peaks) / estimated_fs
            heart_rate = float(np.clip(60.0 / rr_seconds, 30.0, 220.0))
        else:
            heart_rate = 70.0

        alert_level = 'normal' if prediction == 'Normal' else 'warning'

        result = {
            'patient_id': patient_id,
            'prediction': prediction,
            'confidence': confidence,
            'heart_rate': heart_rate,
            'alert_level': alert_level,
            'timestamp': datetime.now().isoformat(),
            'ecg_chunk': ecg_chunk.tolist(),
        }

        emit('prediction_update', result)

        if confidence < 0.4 or alert_level == 'critical':
            emit('critical_alert', {
                'patient_id': patient_id,
                'message': f'Critical condition detected for {patient_id}',
                'timestamp': datetime.now().isoformat(),
            })

    except Exception as exc:
        emit('error', {'message': str(exc)})



# ── Report generation ──────────────────────────────────────────────────────────
@app.route('/api/generate-report', methods=['POST'])
@login_required
@admin_required
def generate_report():
    """Generate a comprehensive PDF cardiac analysis report."""
    try:
        data = request.get_json(silent=True) or {}
        patient_id = data.get('patient_id', 'PATIENT_001')
        prediction = data.get('prediction', 'Unknown')
        confidence = float(data.get('confidence', 0))
        uncertainty = float(data.get('uncertainty', 0))
        probabilities: dict = data.get('probabilities', {})
        features: dict = data.get('features', {})
        recommendations: list = data.get('recommendations', [])
        risk_level = data.get('risk_level', 'Unknown')

        buf = io.BytesIO()
        p = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        # Header
        p.setFillColorRGB(0.05, 0.15, 0.45)
        p.rect(0, height - 1.6 * inch, width, 1.6 * inch, fill=True, stroke=False)
        p.setFillColorRGB(0, 0.83, 1)
        p.setFont("Helvetica-Bold", 22)
        p.drawString(1 * inch, height - 0.85 * inch, "CARDIAC ANALYSIS REPORT")
        p.setFillColorRGB(1, 1, 1)
        p.setFont("Helvetica", 10)
        p.drawString(1 * inch, height - 1.15 * inch,
                     f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        p.drawString(1 * inch, height - 1.35 * inch,
                     "Advanced Cardiac Abnormality Detection System v2.0")

        def section_title(title: str, y: float) -> None:
            p.setFillColorRGB(0.05, 0.15, 0.45)
            p.setFont("Helvetica-Bold", 13)
            p.drawString(1 * inch, y, title)
            p.setStrokeColorRGB(0, 0.83, 1)
            p.setLineWidth(1)
            p.line(1 * inch, y - 4, width - 1 * inch, y - 4)
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 11)

        # Patient info
        y = height - 2.1 * inch
        section_title("Patient Information", y)
        y -= 0.35 * inch
        p.drawString(1.2 * inch, y, f"Patient ID : {patient_id}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Report Date: {datetime.now().strftime('%B %d, %Y')}")

        # Diagnosis
        y -= 0.45 * inch
        section_title("Diagnosis", y)
        y -= 0.35 * inch
        p.drawString(1.2 * inch, y, f"Condition   : {prediction}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Risk Level  : {risk_level}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Confidence  : {confidence * 100:.1f}%")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Uncertainty : {uncertainty * 100:.1f}%")

        # Probabilities
        if probabilities:
            y -= 0.45 * inch
            section_title("Class Probabilities", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 10)
            for cls_name, prob in probabilities.items():
                p.drawString(1.2 * inch, y, f"{cls_name:<30} {prob * 100:.1f}%")
                y -= 0.20 * inch

        # Features table
        if features:
            y -= 0.35 * inch
            section_title("Cardiometric Features", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 9)
            for feature, value in list(features.items())[:10]:
                p.drawString(1.2 * inch, y, f"{feature:<30} {value:.3f}")
                y -= 0.18 * inch

        # Recommendations
        if recommendations and y > 1.8 * inch:
            y -= 0.35 * inch
            section_title("Clinical Recommendations", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 10)
            for i, rec in enumerate(recommendations[:6], 1):
                if y < 1.5 * inch:
                    break
                p.drawString(1.2 * inch, y, f"{i}. {rec}")
                y -= 0.24 * inch

        # Footer
        p.setStrokeColorRGB(0.8, 0.8, 0.8)
        p.line(1 * inch, 1.2 * inch, width - 1 * inch, 1.2 * inch)
        p.setFont("Helvetica-Oblique", 8)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawString(
            1 * inch, 1 * inch,
            "⚠ This report is AI-generated and must be reviewed by a qualified medical professional."
        )

        p.showPage()
        p.save()
        buf.seek(0)

        fname = f'cardiac_report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return send_file(buf, as_attachment=True, download_name=fname, mimetype='application/pdf')

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ── Database API ────────────────────────────────────────────────────────────────────
@app.route('/api/patients', methods=['GET', 'POST'])
@login_required
def manage_patients():
    """List all patients or create a new one."""
    # Write operations (POST) require admin access
    if request.method == 'POST' and session.get('access_level') != 'admin':
        return jsonify({'success': False, 'error': 'Access denied — admin role required'}), 403
    if _USE_MONGO:
        if request.method == 'GET':
            return jsonify({'success': True, 'patients': get_all_patients()})
        data = request.get_json(silent=True) or {}
        missing = [f for f in ('patient_id', 'name') if not data.get(f)]
        if missing:
            return jsonify({'success': False, 'error': f'Missing: {missing}'}), 400
        try:
            patient = create_patient({
                'patient_id':      data['patient_id'],
                'name':            data['name'],
                'age':             data.get('age'),
                'gender':          data.get('gender'),
                'contact_info':    data.get('contact_info'),
                'medical_history': data.get('medical_history', ''),
            })
            return jsonify({'success': True, 'patient': patient}), 201
        except Exception as exc:
            return jsonify({'success': False, 'error': str(exc)}), 500

    # ── SQLite fallback ────────────────────────────────────────────────
    sess = get_session()
    try:
        if request.method == 'GET':
            return jsonify({'success': True, 'patients': [p.to_dict() for p in sess.query(Patient).all()]})
        data = request.get_json(silent=True) or {}
        missing = [f for f in ('patient_id', 'name') if not data.get(f)]
        if missing:
            return jsonify({'success': False, 'error': f'Missing: {missing}'}), 400
        p = Patient(**{k: data.get(k) for k in ('patient_id','name','age','gender','contact_info','medical_history')})
        sess.add(p); sess.commit()
        return jsonify({'success': True, 'patient': p.to_dict()}), 201
    except Exception as exc:
        sess.rollback(); return jsonify({'success': False, 'error': str(exc)}), 500
    finally:
        sess.close()


@app.route('/api/patients/<patient_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def manage_patient(patient_id: str):
    """Get, update, or delete a specific patient."""
    # Write operations (PUT, DELETE) require admin access
    if request.method in ('PUT', 'DELETE') and session.get('access_level') != 'admin':
        return jsonify({'success': False, 'error': 'Access denied — admin role required'}), 403
    if _USE_MONGO:
        patient = get_patient(patient_id)
        if not patient:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        if request.method == 'GET':
            return jsonify({'success': True, 'patient': patient})
        if request.method == 'PUT':
            data = request.get_json(silent=True) or {}
            updates = {k: v for k, v in data.items() if k in ('name','age','gender','contact_info','medical_history')}
            return jsonify({'success': True, 'patient': update_patient(patient_id, updates)})
        delete_patient(patient_id)
        return jsonify({'success': True, 'message': f'Patient {patient_id} deleted'})

    # ── SQLite fallback
    sess = get_session()
    try:
        p = sess.query(Patient).filter_by(patient_id=patient_id).first()
        if not p: return jsonify({'success': False, 'error': 'Not found'}), 404
        if request.method == 'GET': return jsonify({'success': True, 'patient': p.to_dict()})
        if request.method == 'PUT':
            data = request.get_json(silent=True) or {}
            for f in ('name','age','gender','contact_info','medical_history'):
                if f in data: setattr(p, f, data[f])
            sess.commit(); return jsonify({'success': True, 'patient': p.to_dict()})
        sess.delete(p); sess.commit()
        return jsonify({'success': True, 'message': f'{patient_id} deleted'})
    except Exception as exc:
        sess.rollback(); return jsonify({'success': False, 'error': str(exc)}), 500
    finally:
        sess.close()


@app.route('/api/analyses/<patient_id>', methods=['GET'])
@login_required
def get_patient_analyses_route(patient_id: str):
    """Return all analyses for a given patient, newest first."""
    try:
        if _USE_MONGO:
            analyses = get_patient_analyses(patient_id)
        else:
            sess = get_session()
            try:
                from app.database import ECGAnalysis
                analyses = [a.to_dict() for a in sess.query(ECGAnalysis).filter_by(patient_id=patient_id).all()]
            finally:
                sess.close()
        return jsonify({'success': True, 'analyses': analyses})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats_route():
    """Overall system statistics."""
    try:
        if _USE_MONGO:
            stats = mongo_get_stats()
        else:
            sess = get_session()
            try:
                from app.database import Patient as P, ECGAnalysis as E
                stats = {
                    'total_patients':  sess.query(P).count(),
                    'total_analyses':  sess.query(E).count(),
                    'risk_breakdown': {r: sess.query(E).filter_by(risk_level=r).count() for r in ('Low','Medium','High')},
                }
            finally:
                sess.close()
        stats['ml_model_active'] = ml_model is not None and not DEMO_MODE
        stats['demo_mode']       = DEMO_MODE
        stats['success']         = True
        stats['db_backend']      = 'MongoDB' if _USE_MONGO else 'SQLite'
        return jsonify(stats)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health-check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_model is not None and not DEMO_MODE,
        'demo_mode': DEMO_MODE,
        'features': ['Real-time monitoring', 'Explainable AI',
                     'Advanced analytics', 'Report generation'],
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Advanced Cardiac Monitoring System")
    print("=" * 60)
    print("Navigate to: http://localhost:5000")
    print("=" * 60 + "\n")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
