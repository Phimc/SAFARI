import numpy as np

def aperture(n1, n2, c1, c2, radius):
    """
    Create a circular aperture.
    
    Parameters:
    -----------
    n1, n2 : int
        Image dimensions
    c1, c2 : float
        Center coordinates
    radius : float
        Aperture radius
    
    Returns:
    --------
    x : ndarray
        Binary aperture mask
    """
    y_idx, x_idx = np.ogrid[:n1, :n2]
    x = ((y_idx - c1)**2 + (x_idx - c2)**2) <= radius**2
    return x.astype(float)
