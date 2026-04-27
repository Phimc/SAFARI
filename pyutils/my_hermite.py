import numpy as np
from scipy.special import factorial

def my_hermite(n):
    """
    Calculate Hermite polynomial coefficients.
    
    Returns coefficients as a vector where the m-th element is the 
    coefficient of x^(n+1-m).
    np.polyval(my_hermite(n), x) evaluates H_n(x).
    
    Parameters:
    -----------
    n : int
        Order of Hermite polynomial
    
    Returns:
    --------
    hk : ndarray
        Coefficients
    """
    if n == 0:
        return np.array([1.0])
    elif n == 1:
        return np.array([2.0, 0.0])
    else:
        hkm2 = np.zeros(n + 1)
        hkm2[n] = 1
        hkm1 = np.zeros(n + 1)
        hkm1[n - 1] = 2
        
        for k in range(2, n + 1):
            hk = np.zeros(n + 1)
            
            for e in range(n - k + 1, n + 1, 2):
                hk[e - 1] = 2 * (hkm1[e] - (k - 1) * hkm2[e])
            
            hk[n] = -2 * (k - 1) * hkm2[n]
            
            if k < n:
                hkm2 = hkm1.copy()
                hkm1 = hk.copy()
        
        return hk
