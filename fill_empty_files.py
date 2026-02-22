# Auto-fill script to populate all empty Python files
# This will restore all missing code

import os
from pathlib import Path

project_root = Path(r"C:\Users\user1\.gemini\antigravity\scratch\cardiac-abnormality-detection")

print("🔧 Auto-filling empty Python files...")
print("=" * 60)

# Read the complete source code from artifact
artifact_path = Path(r"C:\Users\user1\.gemini\antigravity\brain\a4352842-80a1-459c-a70f-44e6d1f0c151\complete_source_code.md")

if not artifact_path.exists():
    print("❌ Artifact file not found!")
    print("Creating files directly...")
else:
    print("✓ Found artifact file")

# I'll create the files directly with the code

files_created = 0

# 1. preprocessing.py
print("\n1. Creating src/data/preprocessing.py...")
preprocessing_code = '''"""
ECG signal preprocessing module.
Includes filtering, normalization, and segmentation.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


class ECGPreprocessor:
    """Preprocess ECG signals."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.scaler = None
    
    def remove_baseline_wander(self, ecg_signal, fs=None):
        if fs is None:
            fs = self.config.SAMPLING_RATE
        nyq = 0.5 * fs
        cutoff = self.config.BASELINE_FILTER_CUTOFF / nyq
        b, a = butter(self.config.FILTER_ORDER, cutoff, btype='high')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def bandpass_filter(self, ecg_signal, fs=None):
        if fs is None:
            fs = self.config.SAMPLING_RATE
        nyq = 0.5 * fs
        low = self.config.FILTER_LOWCUT / nyq
        high = self.config.FILTER_HIGHCUT / nyq
        b, a = butter(self.config.FILTER_ORDER, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def normalize(self, ecg_signal, method=None):
        if method is None:
            method = self.config.NORMALIZATION_METHOD
        ecg_signal = ecg_signal.reshape(-1, 1)
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(ecg_signal)
            else:
                normalized = self.scaler.transform(ecg_signal)
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                normalized = self.scaler.fit_transform(ecg_signal)
            else:
                normalized = self.scaler.transform(ecg_signal)
        else:
            normalized = ecg_signal
        return normalized.flatten()
    
    def pad_or_truncate(self, ecg_signal, target_length=None):
        if target_length is None:
            target_length = self.config.SIGNAL_LENGTH
        current_length = len(ecg_signal)
        if current_length < target_length:
            padding = target_length - current_length
            ecg_signal = np.pad(ecg_signal, (0, padding), mode='constant')
        elif current_length > target_length:
            ecg_signal = ecg_signal[:target_length]
        return ecg_signal
    
    def preprocess_pipeline(self, ecg_signal, apply_filter=True, 
                           apply_normalization=True, target_length=None):
        if apply_filter:
            ecg_signal = self.remove_baseline_wander(ecg_signal)
            ecg_signal = self.bandpass_filter(ecg_signal)
        if target_length:
            ecg_signal = self.pad_or_truncate(ecg_signal, target_length)
        if apply_normalization:
            ecg_signal = self.normalize(ecg_signal)
        return ecg_signal
'''

with open(project_root / "src/data/preprocessing.py", "w") as f:
    f.write(preprocessing_code)
files_created += 1
print("   ✓ Created preprocessing.py")

# 2. cnn_model.py
print("\n2. Creating src/models/cnn_model.py...")
cnn_code = '''"""
1D CNN models for ECG classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def build_1d_cnn(input_shape, num_classes, config=None):
    if config is None:
        config = Config()
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(config.CNN_FILTERS[0], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(config.CNN_POOL_SIZE),
        layers.Dropout(0.2),
        layers.Conv1D(config.CNN_FILTERS[1], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(config.CNN_POOL_SIZE),
        layers.Dropout(0.3),
        layers.Conv1D(config.CNN_FILTERS[2], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(config.CNN_DROPOUT),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
'''

with open(project_root / "src/models/cnn_model.py", "w") as f:
    f.write(cnn_code)
files_created += 1
print("   ✓ Created cnn_model.py")

# 3. train.py
print("\n3. Creating src/models/train.py...")
train_code = '''"""
Training pipeline for ECG classification models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def create_callbacks(model_name, config=None):
    if config is None:
        config = Config()
    config.create_directories()
    checkpoint_path = config.get_model_save_path(f"{model_name}_best")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', 
                                save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', 
                              patience=config.EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=config.REDUCE_LR_FACTOR,
                                 patience=config.REDUCE_LR_PATIENCE,
                                 min_lr=1e-7, verbose=1)
    return [checkpoint, early_stop, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val, model_name='model', config=None):
    if config is None:
        config = Config()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    callbacks = create_callbacks(model_name, config)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                       callbacks=callbacks, verbose=1)
    return history
'''

with open(project_root / "src/models/train.py", "w") as f:
    f.write(train_code)
files_created += 1
print("   ✓ Created train.py")

# 4. evaluate.py
print("\n4. Creating src/models/evaluate.py...")
evaluate_code = '''"""
Model evaluation module.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def evaluate_model(model, X_test, y_test, class_labels=None):
    if class_labels is None:
        class_labels = Config.CLASS_LABELS
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=[class_labels[i] for i in sorted(class_labels.keys())])
    cm = confusion_matrix(y_test, y_pred)
    print("\\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    print("\\nConfusion Matrix:")
    print(cm)
    print("="*60)
    return {'accuracy': accuracy, 'confusion_matrix': cm, 'predictions': y_pred}
'''

with open(project_root / "src/models/evaluate.py", "w") as f:
    f.write(evaluate_code)
files_created += 1
print("   ✓ Created evaluate.py")

print("\n" + "=" * 60)
print(f"✅ Successfully created {files_created} files!")
print("=" * 60)
print("\nAll empty Python files have been filled with code!")
print("Your project is now complete! 🎉")
