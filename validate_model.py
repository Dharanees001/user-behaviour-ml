"""
Validation script — loads best_model.pkl safely by remapping
__main__ classes to their actual module (main.py).
"""

import io
import pickle
import sys

import joblib
import numpy as np

import main  # noqa: F401


class _Remapper(pickle.Unpickler):
    """Remap __main__.ClassName -> main.ClassName so joblib can unpickle."""

    def find_class(self, module, name):
        if module == "__main__":
            module = "main"
        return super().find_class(module, name)


def safe_load(path):
    try:
        return joblib.load(path)
    except AttributeError:
        with open(path, "rb") as f:
            return _Remapper(io.BytesIO(f.read())).load()


b = safe_load("best_model.pkl")
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
