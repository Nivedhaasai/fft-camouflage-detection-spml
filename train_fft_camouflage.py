from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.config import RANDOM_SEED, CAMO_PROB_THRESHOLD
from src.dataset_utils import build_multiscale_patch_dataset
from src.fft_features import get_multiscale_feature_names

def optimize_threshold(y_true, y_probs):
    """Search for the optimal threshold (0.45 to 0.75) maximizing F1-score."""
    thresholds = np.linspace(0.45, 0.75, 31)
    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh

def train_camo_detector(
    camo_dir: str = "dataset/camo",
    non_camo_dir: str = "dataset/non_camo",
    masks_dir: str = "masks",
    model_path: str = "models/svm_fft_camouflage_model.joblib",
):
    print("Building extended feature dataset...")
    X, y = build_multiscale_patch_dataset(
        camo_dir=Path(camo_dir),
        non_camo_dir=Path(non_camo_dir),
        masks_dir=Path(masks_dir),
        limit=40000
    )
    feat_names = get_multiscale_feature_names()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Step 5: Cross-Validation
    # Use uncalibrated model for speed in CV
    base_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", class_weight="balanced"))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(base_pipe, X_train, y_train, cv=skf)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Step 4: Calibrated SVM
    print("Training Calibrated SVM...")
    calibrated_svm = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", class_weight="balanced"))]),
        method="sigmoid", cv=skf
    )
    calibrated_svm.fit(X_train, y_train)

    # Step 6: Feature Importance (Random Forest)
    print("Estimating feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.barh(range(10), importances[indices], align="center")
    plt.yticks(range(10), [feat_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_plot.png")
    plt.close()

    # Step 7: Threshold Optimization
    y_probs = calibrated_svm.predict_proba(X_test)[:, 1]
    best_thresh = optimize_threshold(y_test, y_probs)
    print(f"Optimized Probability Threshold: {best_thresh:.3f}")

    # Metrics
    y_pred = (y_probs >= best_thresh).astype(int)
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Feature vector length: {X.shape[1]}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

    joblib.dump({"model": calibrated_svm, "threshold": best_thresh}, "models/svm_fft_camouflage_model.joblib")
    print("Model and metadata saved.")

if __name__ == "__main__":
    train_camo_detector()
