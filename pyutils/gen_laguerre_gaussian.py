import numpy as np
from scipy.special import factorial
from .my_laguerre import my_laguerre

def gen_laguerre_gaussian(X, Y, z, wavlen, w0, l, p):
    """
    Generate Laguerre-Gaussian beams.
    
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
    l : int
        Azimuthal index
    p : int
        Radial index
    
    Returns:
    --------
    U : ndarray
        Complex amplitude of the wavefront
    """
    zR = np.pi * w0**2 / wavlen
    N = abs(l) + 2 * p
    k = 2 * np.pi / wavlen
    
    w = lambda z: w0 * np.sqrt(1 + (z / zR)**2)
    psi = lambda z: (N + 1) * np.arctan(z / zR)
    
    # Convert to polar coordinates
    phi = np.arctan2(Y, X)
    rho = np.sqrt(X**2 + Y**2)
    
    # Define the normalizing constant
    C = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + abs(l))))
    
    # Calculate R(z)
    if z == 0:
        Rz = np.inf
    else:
        Rz = z * (1 + (zR / z)**2)
    
    # Calculate the wavefront
    U = (C * 1.0 / w(z) * (rho * np.sqrt(2) / w(z))**abs(l) * 
         np.exp(-rho**2 / w(z)**2) * 
         my_laguerre(p, abs(l), 2 * rho**2 / w(z)**2) *
         np.exp(-1j * k * rho**2 / (2 * Rz)) * 
         np.exp(-1j * l * phi) * 
         np.exp(1j * psi(z)))
    
    return U
