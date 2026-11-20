
from collections import Counter

def confusion_matrix(data):
    # Implement the function here
    counts = Counter((int(y_true), int(y_pred)) for y_true, y_pred in data) # * 

    # automatically 0 if no count 
    tp = counts[(1, 1)]
    fn = counts[(1, 0)]
    fp = counts[(0, 1)]
    tn = counts[(0, 0)]

    return [[tp, fn], [fp, tn]]