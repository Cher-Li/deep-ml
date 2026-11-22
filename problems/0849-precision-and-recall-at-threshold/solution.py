import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """

    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)

    y_pred = (y_scores_arr >= threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    predicted_positives = int(np.sum(y_pred == 1))
    actual_positives = int(np.sum(y_true_arr == 1))

    precision = (tp / predicted_positives) if predicted_positives > 0 else 0.0
    recall = (tp / actual_positives) if actual_positives > 0 else 0.0

    return [float(round(precision, 4)), float(round(recall, 4))]