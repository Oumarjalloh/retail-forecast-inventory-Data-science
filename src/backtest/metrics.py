import numpy as np

def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return 200.0 * np.mean(np.abs(y_pred - y_true) / denom)

def wape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100.0 * (np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))