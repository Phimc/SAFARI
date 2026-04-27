import numpy as np
from scipy.special import factorial

def zernfun(n, m, r, theta, nflag=''):
    """
    Zernike functions of order N and frequency M on the unit circle.
    
    Parameters:
    -----------
    n : int or array
        Order(s)
    m : int or array
        Angular frequency(ies)
    r : ndarray
        Radial coordinates (0 to 1)
    theta : ndarray
        Angular coordinates
    nflag : str
        If 'norm', returns normalized Zernike functions
    
    Returns:
    --------
    z : ndarray
        Zernike function values
    """
    # Convert to arrays
    n = np.atleast_1d(n).flatten()
    m = np.atleast_1d(m).flatten()
    r = np.atleast_1d(r).flatten()
    theta = np.atleast_1d(theta).flatten()
    
    if len(n) != len(m):
        raise ValueError('N and M must be the same length.')
    
    if np.any((n - m) % 2 != 0):
        raise ValueError('All N and M must differ by multiples of 2.')
    
    if np.any(np.abs(m) > n):
        raise ValueError('Each M must be less than or equal to its corresponding N.')
    
    if np.any((r < 0) | (r > 1)):
        raise ValueError('All R must be between 0 and 1.')
    
    if len(r) != len(theta):
        raise ValueError('The number of R- and THETA-values must be equal.')
    
    # Check normalization
    isnorm = (nflag.lower() == 'norm') if isinstance(nflag, str) else False
    
    length_r = len(r)
    m_abs = np.abs(m)
    
    # Determine required powers of r
    rpowers = []
    for j in range(len(n)):
        rpowers.extend(range(m_abs[j], n[j] + 1, 2))
    rpowers = np.unique(rpowers)
    
    # Pre-compute r powers
    if rpowers[0] == 0:
        rpowern = np.column_stack([np.ones(length_r)] + 
                                   [r**p for p in rpowers[1:]])
    else:
        rpowern = np.column_stack([r**p for p in rpowers])
    
    # Compute polynomial values
    z = np.zeros((length_r, len(n)))
    for j in range(len(n)):
        s = np.arange(0, (n[j] - m_abs[j]) // 2 + 1)
        pows = np.arange(n[j], m_abs[j] - 1, -2)
        
        for k in range(len(s) - 1, -1, -1):
            p = ((-1)**(s[k] % 2) * 
                 factorial(n[j] - s[k]) / 
                 (factorial(s[k]) * 
                  factorial((n[j] - m_abs[j]) // 2 - s[k]) * 
                  factorial((n[j] + m_abs[j]) // 2 - s[k])))
            
            idx = np.where(rpowers == pows[k])[0][0]
            z[:, j] = z[:, j] + p * rpowern[:, idx]
        
        if isnorm:
            z[:, j] = z[:, j] * np.sqrt((1 + (m[j] != 0)) * (n[j] + 1) / np.pi)
    
    # Compute Zernike functions
    idx_pos = m > 0
    idx_neg = m < 0
    
    if np.any(idx_pos):
        z[:, idx_pos] = z[:, idx_pos] * np.cos(theta[:, None] * m_abs[idx_pos][None, :])
    
    if np.any(idx_neg):
        z[:, idx_neg] = z[:, idx_neg] * np.sin(theta[:, None] * m_abs[idx_neg][None, :])
    
    return z
