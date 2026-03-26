from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.dataset_utils import build_multiscale_patch_dataset


def train_camo_detector(
    camo_dir: str = "dataset/camo",
    non_camo_dir: str = "dataset/non_camo",
    masks_dir: str = "masks",
    model_path: str = "models/svm_fft_camouflage_model.joblib",
):
    """Train an academic-quality SVM classifier (FFT+Gabor) with balanced sampling."""
    print("Building multi-scale feature dataset...")
    X, y = build_multiscale_patch_dataset(
        camo_dir=Path(camo_dir),
        non_camo_dir=Path(non_camo_dir),
        masks_dir=Path(masks_dir),
        limit=40000,
    )
    
    print(f"Dataset extracted: {X.shape[0]} patches, {X.shape[1]} features each.")
    
    # Split for internal validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Standard Pipeline: Scaler + SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=RANDOM_SEED))
    ])
    
    print("Training SVM classifier (this may take a few minutes)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    
    print("\n" + "="*40)
    print("EVALUATION METRICS")
    print("="*40)
    print(f"Training patch count: {len(X_train)}")
    print(f"Testing patch count:  {len(X_test)}")
    camo_count = int(np.sum(y))
    print(f"Class balance: {camo_count} camo / {len(y) - camo_count} background (ratio: {camo_count/len(y):.2f})")
    print("-" * 40)
    
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model and scaler saved to: {model_path}")


if __name__ == "__main__":
    train_camo_detector()
