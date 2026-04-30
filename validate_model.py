"""
Validation script — patches sys.modules so joblib can unpickle
classes that were saved under __main__ (via python main.py).
"""

import sys

import joblib
import numpy as np

import main  # noqa: F401

sys.modules["__main__"] = main  # remap __main__ -> main before unpickling

b = joblib.load("best_model.pkl")
acc = b["accuracy"] * 100

print(f"Model     : {b['model_name']}")
print(f"Accuracy  : {acc:.2f}%")
print(f"Features  : {b['features']}")

if acc < 75.0:
    print("✗ FAILED — below 75% threshold")
    sys.exit(1)
print("✓ Accuracy PASSED")

if "Locally Weighted" not in b["model_name"]:
    X = np.array([[120, 5.0, 1500, 30, 500, 25, 1, 1]])
    X_scaled = b["scaler"].transform(X)
    pred = b["model"].predict(X_scaled)[0]
    print(f"✓ Prediction: class {pred} ({b['labels'][pred]})")
else:
    print("ℹ LWR — prediction test skipped")
