# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    candidates = [
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        SVC(C=1.0, kernel="rbf", random_state=42),
        RandomForestClassifier(n_estimators=200, random_state=42),
        HistGradientBoostingClassifier(random_state=42),
    ]

    best_model = None
    best_score = -1.0

    for model in candidates:
        # * for each candidate, go through the process of fit, pred, and score
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = accuracy_score(y_val, val_preds)
        if score > best_score:
            best_score = score
            best_model = model

    # refit model on combined train + val
    if X_val is not None and len(X_val) > 0:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        best_model.fit(X_combined, y_combined)

    def predict(X):
        preds = best_model.predict(np.asarray(X))
        return np.asarray(preds, dtype=int).ravel()

    return predict