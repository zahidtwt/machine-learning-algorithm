import numpy as np


def pretreat(X, method = 'standardization'):
    """
    Feature scaling for X. The default method is standardization. Takes a matrix X and returns the scaled matrix.

    Parameters
    ----------
    X : m x n matrix
        The matrix to be scaled. Each row is a sample and each column is a feature. All elements are numerical.
    method : str
        The method of scaling. The default is 'standardization'. Other option is 'minmax', 'pareto', 'center', 'unilength'.

    Returns
    -------
    X_scaled : m x n matrix
        The scaled matrix.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> pretreat(X)
    array([[-1.22474487, -1.22474487, -1.22474487],
           [ 0.        ,  0.        ,  0.        ],
           [ 1.22474487,  1.22474487,  1.22474487]])
    """

    if method == 'standardization':
        X_scaled = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    elif method == 'minmax':
        X_scaled = (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0))
    elif method == 'pareto':
        X_scaled = (X - np.mean(X, axis = 0)) / np.sqrt(np.std(X, axis = 0))
    elif method == 'center':
        X_scaled = X - np.mean(X, axis = 0)
    elif method == 'unilength':
        X_scaled = X / np.sqrt(np.sum(X ** 2, axis = 0))
    else:
        raise ValueError('The method is not supported.')
    return X_scaled

if __name__ == '__main__':
    print(pretreat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))