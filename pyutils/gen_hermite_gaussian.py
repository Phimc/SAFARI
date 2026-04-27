import numpy as np
from scipy.special import factorial
from .my_hermite import my_hermite

def gen_hermite_gaussian(X, Y, z, wavlen, w0, m, n):
    """
    Generate Hermite-Gaussian beams.
    
    Parameters:
    -----------
    X, Y : ndarray
        2D coordinates (mm)
    z : float
        Axial location (mm)
    wavlen : float
        Wavelength (mm)
    w0 : float
        Waist radius (mm)
    m, n : int
        Mode indices
    
    Returns:
    --------
    U : ndarray
        Complex amplitude of the wavefront
    """
    k = 2 * np.pi / wavlen
    rho = np.sqrt(X**2 + Y**2)
    zr = np.pi * w0**2 / wavlen
    w = w0 * np.sqrt(1 + (z / zr)**2)
    
    Hx = np.polyval(my_hermite(m), np.sqrt(2) * X / w)
    Hy = np.polyval(my_hermite(n), np.sqrt(2) * Y / w)
    rc = np.sqrt(2**(1 - n - m) / (np.pi * factorial(n) * factorial(m))) / w
    
    # Calculate R(z)
    if z == 0:
        Rz = np.inf
    else:
        Rz = z * (1 + (zr / z)**2)
    
    # Calculate the wavefront
    U = rc * Hx * Hy * np.exp(1j * (n + m + 1) * np.arctan(z / zr)) * np.exp(-rho**2 / w**2) * np.exp(-1j * k * rho**2 / (2 * Rz)) * np.exp(1j * k * z)
    
    # Normalization
    U = U / np.sqrt(np.sum(np.abs(U)**2))
    
    return U
