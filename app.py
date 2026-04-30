"""
BehaviourIQ — Flask Backend (Prediction Only)
Includes all model class definitions so joblib can load best_model.pkl
"""

import os
import warnings
import joblib
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── All model classes (needed by joblib to unpickle best_model.pkl) ──


class LocallyWeightedRegression:

    def __init__(self, tau=0.8, max_train_samples=1000):
        self.tau = tau
        self.max_train_samples = max_train_samples

    def _gaussian_weights(self, X_train, x_query):
        diff = X_train - x_query
        return np.exp(-np.sum(diff**2, axis=1) / (2 * self.tau**2))

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def predict(self, X_train, y_train, X_test):
        if len(X_train) > self.max_train_samples:
            idx = np.random.choice(len(X_train), self.max_train_samples, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        classes = np.unique(y_train)
        X_train_b = self._add_bias(X_train)
        X_test_b = self._add_bias(X_test)
        all_scores = np.zeros((X_test.shape[0], len(classes)))
        for i, x_q in enumerate(X_test):
            w = self._gaussian_weights(X_train, x_q)
            W = np.diag(w)
            for j, cls in enumerate(classes):
                y_bin = (y_train == cls).astype(float)
                theta = np.linalg.pinv(X_train_b.T @ W @ X_train_b) @ (X_train_b.T @ W @ y_bin)
                all_scores[i, j] = X_test_b[i] @ theta
        return classes[np.argmax(all_scores, axis=1)]


class WeightedRegression:

    def __init__(self, weighting="distance"):
        self.weighting = weighting
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_test):
        preds = []
        for x in X_test:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            dists = np.where(dists == 0, 1e-10, dists)
            weights = 1.0 / dists
            classes = np.unique(self.y_train)
            class_scores = {c: weights[self.y_train == c].sum() for c in classes}
            preds.append(max(class_scores, key=class_scores.get))
        return np.array(preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class SVMClassifier:

    def __init__(self):
        self.best_kernel = None
        self.best_model = None
        self.kernel_scores = {}

    def fit_best(self, X_train, y_train):
        from sklearn.model_selection import cross_val_score
        for kernel in ["rbf", "linear", "poly"]:
            clf = SVC(kernel=kernel, C=1.0, gamma="scale", probability=True, random_state=42)
            cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            self.kernel_scores[kernel] = cv.mean()
        self.best_kernel = max(self.kernel_scores, key=self.kernel_scores.get)
        self.best_model = SVC(
            kernel=self.best_kernel, C=1.0, gamma="scale",
            probability=True, random_state=42, decision_function_shape="ovr"
        )
        self.best_model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class ClusteringClassifier:

    def __init__(self, n_clusters=12):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_label = {}

    def fit(self, X, y):
        self.kmeans.fit(X)
        for c in range(self.n_clusters):
            mask = (self.kmeans.labels_ == c)
            if mask.sum() > 0:
                vals, counts = np.unique(y[mask], return_counts=True)
                self.cluster_label[c] = int(vals[np.argmax(counts)])
            else:
                self.cluster_label[c] = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        return np.array([self.cluster_label[c] for c in self.kmeans.predict(X)])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class DecisionTreeRuleLearner:

    def __init__(self, max_depth=6):
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_split=10,
            min_samples_leaf=5, criterion="gini", random_state=42
        )
        self.feature_names = None

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        return self.tree.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_rules(self):
        return export_text(self.tree, feature_names=self.feature_names)

    def feature_importance_dict(self):
        return dict(zip(self.feature_names, self.tree.feature_importances_))


class RuleBasedClassifier:

    def __init__(self):
        self.rules = []
        self.default = 0

    def _extract_rules(self, tree_clf):
        t, rules = tree_clf.tree_, []

        def recurse(node, conditions):
            if t.feature[node] == -2:
                rules.append((list(conditions), int(tree_clf.classes_[np.argmax(t.value[node])])))
                return
            feat, thr = t.feature[node], t.threshold[node]
            recurse(t.children_left[node], conditions + [(feat, thr, "left")])
            recurse(t.children_right[node], conditions + [(feat, thr, "right")])

        recurse(0, [])
        return rules

    def fit(self, X, y, feature_names):
        base = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
        base.fit(X, y)
        scored = []
        for conditions, label in self._extract_rules(base):
            mask = np.ones(X.shape[0], dtype=bool)
            for feat, thr, direction in conditions:
                if direction == "left":
                    mask &= X[:, feat] <= thr
                else:
                    mask &= X[:, feat] > thr
            n = mask.sum()
            if n >= 5:
                conf = (y[mask] == label).sum() / n
                scored.append((conf, n, conditions, label))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        self.rules = [(c, l) for _, _, c, l in scored[:20]]
        vals, counts = np.unique(y, return_counts=True)
        self.default = int(vals[np.argmax(counts)])
        self.feature_names = feature_names
        return self

    def _apply_rule(self, x, conditions):
        for feat, thr, direction in conditions:
            if direction == "left" and x[feat] > thr:
                return False
            if direction == "right" and x[feat] <= thr:
                return False
        return True

    def predict(self, X):
        preds = []
        for x in X:
            matched = False
            for conditions, label in self.rules:
                if self._apply_rule(x, conditions):
                    preds.append(label)
                    matched = True
                    break
            if not matched:
                preds.append(self.default)
        return np.array(preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ── Flask App ─────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

MODEL_PATH = "best_model.pkl"
LABEL_DESC = {
    0: "Minimal device engagement — healthy digital habits.",
    1: "Moderate usage — balanced screen time.",
    2: "Heavy usage — high digital dependency.",
}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    ready = os.path.exists(MODEL_PATH)
    return jsonify({"status": "ok", "model": MODEL_PATH if ready else None, "ready": ready})


@app.route("/predict", methods=["POST"])
def predict_route():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found. Place best_model.pkl in the app directory."}), 404
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    required = ["app_usage", "screen_on", "battery", "apps", "data_usage", "age", "gender", "os"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        features = np.array([[
            float(data["app_usage"]), float(data["screen_on"]),
            float(data["battery"]), float(data["apps"]),
            float(data["data_usage"]), float(data["age"]),
            int(data["gender"]), int(data["os"]),
        ]])
        scaled = bundle["scaler"].transform(features)
        pred = int(bundle["model"].predict(scaled)[0])
        label = bundle.get("labels", {}).get(pred, ["Low", "Medium", "High"][pred])
        return jsonify({
            "class": pred,
            "label": label,
            "description": LABEL_DESC.get(pred, ""),
            "model_name": bundle.get("model_name", "Unknown"),
            "accuracy": round(bundle.get("accuracy", 0) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  BehaviourIQ — Prediction Server")
    print("  Open  →  http://127.0.0.1:5000")
    print("═" * 52 + "\n")
    app.run(debug=True, port=5000)
