import numpy as np


def quad_fit(x):
    """Fits a quadratic curve to three points.

    params:
    -------
    x: np.ndarray
        has shape (n, 3)

    returns:
    --------
    np.ndarray of shape (n, 3) of respectively a, b, c in quadratic
    function ax^2 + bx + c for all n inputs.
    """
    c = x[:, 1:2]    
    a = (x[:, :1] + x[:, 2:]) / 2. - c
    b = x[:, 2:] - c - a

    return np.concatenate([a, b, c], axis=1)


def quad_opt(coeffs):
    """Given coefficients of quadratic function computes optimum.

    params:
    -------
    coefs: np.ndarray (n, 3)
        Respectively a, b, c in quadratic function ax^2 + bx + c

    returns:
    --------
    np.ndarray of (n, 2). First column contains the arg-optimum.
    While the second column contains arg-optimum.
    """
    a, b, c = coeffs[:, :1], coeffs[:, 1:2], coeffs[:, 2:]
    x = - b / (2 * a)
    opt = a * x * x + b * x + c
    return np.append(x, opt, axis=1)
