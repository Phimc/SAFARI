import numpy as np
from scipy.special import jv

def gen_bessel(X, Y, z, wavlen, n, theta):
    """
    Generate Bessel beams.
    
    Parameters:
    -----------
    X, Y : ndarray
        2D coordinates (mm)
    z : float
        Axial location (mm)
    wavlen : float
        Wavelength (mm)
    n : int
        Topological charge
    theta : float
        Axicon angle (rad)
    
    Returns:
    --------
    U : ndarray
        Complex amplitude of the wavefront
    """
    k = 2 * np.pi / wavlen
    phi = np.arctan2(Y, X)
    rho = np.sqrt(X**2 + Y**2)
    
    kz = k * np.cos(theta)
    kr = k * np.sin(theta)
    
    U = np.exp(1j * kz * z) * jv(n, kr * rho) * np.exp(1j * n * phi)
    
    return U
