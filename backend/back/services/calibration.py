import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os

CALIBRATION_MODEL_PATH = "calibrator.pkl"

def train_calibrator(raw_scores: list, true_labels: list):
    """Trains Platt scaling on validation set."""
    X = np.array(raw_scores).reshape(-1, 1)
    y = np.array(true_labels)
    lr = LogisticRegression()
    lr.fit(X, y)
    with open(CALIBRATION_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)

def calibrate_score(raw_score: float) -> float:
    """Applies Platt scaling."""
    if not os.path.exists(CALIBRATION_MODEL_PATH):
        return raw_score # Fallback
    with open(CALIBRATION_MODEL_PATH, "rb") as f:
        lr = pickle.load(f)
    return lr.predict_proba([[raw_score]])[0][1] # Return prob of class 1
