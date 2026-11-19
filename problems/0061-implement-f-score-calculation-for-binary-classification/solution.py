import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: Weight of recall relative to precision (beta > 1 favours recall)
    :return: F-Score rounded to three decimal places
    """
    # this is for recall: 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.sum(y_true == 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))

    if denominator == 0:
        recall = 0.0
    else:
        recall = float(tp / denominator)

    fp = np.sum((y_true == 0) & (y_pred == 1)) 
    denom = tp + fp
    if denom == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if beta ** 2 * precision + recall == 0:
        return 0.0 
    return round((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall), 3)