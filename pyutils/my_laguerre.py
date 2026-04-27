import numpy as np
from scipy.special import factorial

def my_laguerre(p, l, x):
    """
    Calculate Laguerre polynomial L_p^l(x).
    
    Parameters:
    -----------
    p : int
        Radial index
    l : int
        Azimuthal index
    x : ndarray
        Input values
    
    Returns:
    --------
    y : ndarray
        Polynomial values
    """
    if p == 0:
        return np.ones_like(x)
    
    coef = np.zeros(p + 1)
    for m in range(p + 1):
        coef[p - m] = ((-1)**m * factorial(p + l)) / (factorial(p - m) * factorial(l + m) * factorial(m))
    
    return np.polyval(coef, x)
