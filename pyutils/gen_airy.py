import numpy as np
from scipy.special import airy as scipy_airy

def gen_airy(X, Y, w0, x0, y0, a):
    """
    Generate Airy beams.
    
    Parameters:
    -----------
    X, Y : ndarray
        2D coordinates (mm)
    w0 : float
        Scaling factor
    x0, y0 : float
        Center location (mm)
    a : float
        Exponential truncation factor
    
    Returns:
    --------
    U : ndarray
        Complex amplitude of the wavefront
    """
    ai_x, _, _, _ = scipy_airy((x0 - X) / w0)
    ai_y, _, _, _ = scipy_airy((y0 - Y) / w0)
    
    U = ai_x * ai_y * np.exp(a * (x0 - X) / w0) * np.exp(a * (y0 - Y) / w0)
    
    return U
