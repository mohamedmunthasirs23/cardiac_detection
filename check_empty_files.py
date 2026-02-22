# Script to recreate all empty Python files
# Run this to restore all missing code

import os
from pathlib import Path

project_root = Path(r"C:\Users\user1\.gemini\antigravity\scratch\cardiac-abnormality-detection")

# Check which files are empty
empty_files = []
important_files = [
    "src/data/preprocessing.py",
    "src/features/feature_extraction.py",
    "src/models/cnn_model.py",
    "src/models/lstm_model.py",
    "src/models/hybrid_model.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/utils/visualization.py",
]

print("Checking for empty files...")
print("=" * 60)

for file_path in important_files:
    full_path = project_root / file_path
    if full_path.exists():
        size = full_path.stat().st_size
        if size == 0:
            empty_files.append(file_path)
            print(f"❌ EMPTY: {file_path}")
        else:
            print(f"✓ OK ({size} bytes): {file_path}")
    else:
        print(f"⚠️  MISSING: {file_path}")
        empty_files.append(file_path)

print("=" * 60)
print(f"\nFound {len(empty_files)} empty or missing files")

if empty_files:
    print("\n🔧 These files need to be recreated:")
    for f in empty_files:
        print(f"   - {f}")
    print("\n💡 I will recreate these files for you...")
else:
    print("\n✅ All files have content!")
