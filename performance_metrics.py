import numpy as np


def confusion_matrix(y_hat, y):
    """
    Confusion matrix

    Parameters
    ----------
    y_hat : numpy.ndarray
        Predicted values
    y : numpy.ndarray
        Actual values

    Returns
    -------
    numpy.ndarray
        Confusion matrix

    """
    n = len(np.unique(y))
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.sum((y == i) & (y_hat == j))
    return matrix

def calculate_metrics(y_hat, y):
    """
    Calculate performance metrics

    Parameters
    ----------
    y_hat : numpy.ndarray
        Predicted values
    y : numpy.ndarray
        Actual values

    Returns
    -------
    dict
        Performance metrics

    """
    matrix = confusion_matrix(y_hat, y)
    n = matrix.shape[0]
    metrics = {}
    metrics['accuracy'] = np.sum(np.diag(matrix)) / np.sum(matrix)
    metrics['precision'] = np.sum(np.diag(matrix) / np.sum(matrix, axis = 0)) / n
    metrics['recall'] = np.sum(np.diag(matrix) / np.sum(matrix, axis = 1)) / n
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    return metrics