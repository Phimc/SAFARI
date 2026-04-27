import numpy as np
from scipy.special import gamma

def gen_ince_gaussian(L, N, parity, p, m, e, w0, k, z):
    """
    Calculate an Ince-Gaussian Beam at a given z plane.
    
    Parameters:
    -----------
    L : float
        Transverse physical size of the X-Y space [-L,L,-L,L]
    N : int
        Number of sampling points (must be ODD)
    parity : int
        Parity of the beam, 0 = EVEN, 1 = ODD
    p, m : int
        Order and degree of the Ince Gaussian beam
    e : float
        Ellipticity parameter
    w0 : float
        Beam width(waist) at z=0
    k : float
        2*pi/lambda, wavenumber
    z : float
        Propagation distance
    
    Returns:
    --------
    IGB : ndarray
        Ince Gaussian Beam
    X, Y : ndarray
        Space matrices
    """
    
    # Check input
    if N % 2 == 0:
        raise ValueError('ERROR: N must be ODD')
    
    if parity == 0:
        if m < 0 or m > p:
            raise ValueError('ERROR: Wrong range for "m", 0<=m<=p')
    else:
        if m < 1 or m > p:
            raise ValueError('ERROR: Wrong range for "m", 1<=m<=p')
    
    if (-1)**(m - p) != 1:
        raise ValueError('ERROR: (p,m) must have the same parity')
    
    # Parameters
    f0 = np.sqrt(e / 2) * w0
    
    # Ince Gaussian Beam
    if z == 0:
        xhi, etha, X, Y = _mesh_elliptic(f0, L, N)
        R = np.sqrt(X**2 + Y**2)
        if parity == 0:
            IGB = (_CInceIGB(p, m, e, etha)[0] * 
                   _CInceIGB(p, m, e, 1j * xhi)[0] * 
                   np.exp(-(R / w0)**2))
        else:
            IGB = (_SInceIGB(p, m, e, etha)[0] * 
                   _SInceIGB(p, m, e, 1j * xhi)[0] * 
                   np.exp(-(R / w0)**2))
    else:
        zr = 0.5 * k * w0**2
        wz = w0 * np.sqrt(1 + (z / zr)**2)
        Rz = z * (1 + (zr / z)**2)
        f = f0 * wz / w0
        xhi, etha, X, Y = _mesh_elliptic(f, L, N)
        R = np.sqrt(X**2 + Y**2)
        if parity == 0:
            IGB = ((w0 / wz) * 
                   (_CInceIGB(p, m, e, etha)[0] * 
                    _CInceIGB(p, m, e, 1j * xhi)[0]) * 
                   np.exp(-(R / wz)**2) * 
                   np.exp(1j * (k * z + k * R**2 / (2 * Rz) - (p + 1) * np.arctan(z / zr))))
        else:
            IGB = ((w0 / wz) * 
                   (_SInceIGB(p, m, e, etha)[0] * 
                    _SInceIGB(p, m, e, 1j * xhi)[0]) * 
                   np.exp(-(R / wz)**2) * 
                   np.exp(1j * (k * z + k * R**2 / (2 * Rz) - (p + 1) * np.arctan(z / zr))))
    
    # Compute normalization constants
    if parity == 0:
        if p % 2 == 0:
            C0, _, coef, _ = _CInceIGB(p, m, e, 0)
            Cp, _, _, _ = _CInceIGB(p, m, e, np.pi / 2)
            Norm = ((-1)**(m / 2) * np.sqrt(2) * gamma(p / 2 + 1) * coef[0] * 
                    np.sqrt(2 / np.pi) / w0 / C0 / Cp)
        else:
            C0, _, coef, _ = _CInceIGB(p, m, e, 0)
            _, _, _, DCp = _CInceIGB(p, m, e, np.pi / 2)
            Norm = ((-1)**((m + 1) / 2) * gamma((p + 1) / 2 + 1) * 
                    np.sqrt(4 * e / np.pi) * coef[0] / w0 / C0 / DCp)
    else:
        if p % 2 == 0:
            _, _, coef, dS0 = _SInceIGB(p, m, e, 0)
            _, _, _, dSp = _SInceIGB(p, m, e, np.pi / 2)
            Norm = ((-1)**(m / 2) * np.sqrt(2) * e * gamma((p + 2) / 2 + 1) * 
                    coef[0] * np.sqrt(2 / np.pi) / w0 / dS0 / dSp)
        else:
            Sp, _, coef, _ = _SInceIGB(p, m, e, np.pi / 2)
            _, _, _, dS0 = _SInceIGB(p, m, e, 0)
            Norm = ((-1)**((m - 1) / 2) * gamma((p + 1) / 2 + 1) * 
                    np.sqrt(4 * e / np.pi) * coef[0] / w0 / Sp / dS0)
    
    IGB = IGB * Norm
    
    return IGB, X, Y


def _mesh_elliptic(f, L, N):
    """Create elliptic coordinate mesh."""
    X, Y = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
    
    xi = np.zeros((N, N))
    eta = np.zeros((N, N))
    
    # Calculate First Quadrant
    en = np.arccosh((X[:(N + 1) // 2, (N - 1) // 2:N] + 
                     1j * Y[:(N + 1) // 2, (N - 1) // 2:N]) / f)
    ee = np.real(en)
    nn = np.imag(en)
    nn = nn + (nn < 0) * 2 * np.pi
    xi[:(N + 1) // 2, (N - 1) // 2:N] = ee
    eta[:(N + 1) // 2, (N - 1) // 2:N] = nn
    
    # Calculate other quadrants by symmetry
    xi[:(N + 1) // 2, :(N - 1) // 2] = np.fliplr(xi[:(N + 1) // 2, (N + 1) // 2:N])
    xi[(N + 1) // 2:N, :N] = np.flipud(xi[:(N - 1) // 2, :N])
    eta[:(N + 1) // 2, :(N - 1) // 2] = np.pi - np.fliplr(eta[:(N + 1) // 2, (N + 1) // 2:N])
    eta[(N + 1) // 2:N, :N] = np.pi + np.rot90(eta[:(N - 1) // 2, :N], 2)
    
    return xi, eta, X, Y


def _CInceIGB(p, m, q, z):
    """Even Ince Polynomial."""
    if m < 0 or m > p:
        raise ValueError('ERROR: Wrong range for "m", 0<=m<=p')
    if (-1)**(m - p) != 1:
        raise ValueError('ERROR: (p,m) must have the same parity')
    
    largo, ancho = z.shape if hasattr(z, 'shape') else (1, len(z))
    z = z.flatten()
    normalization = 1
    
    if p % 2 == 0:
        j = p // 2
        N = j + 1
        n = m // 2 + 1
        
        # Matrix
        M = np.diag(q * (j + np.arange(1, N)), 1) + \
            np.diag(np.concatenate([[2 * q * j], q * (j - np.arange(1, N - 1))]), -1) + \
            np.diag(np.concatenate([[0], 4 * (np.arange(N - 1) + 1)**2]))
        if p == 0:
            M = np.array([[0]])
        
        # Eigenvalues and Eigenvectors
        ets, A = np.linalg.eig(M)
        index = np.argsort(ets)
        ets = ets[index]
        A = A[:, index]
        
        # Normalization
        if normalization == 0:
            N2 = 2 * A[0, n - 1]**2 + np.sum(A[1:N, n - 1]**2)
            NS = np.sign(np.sum(A[:, n - 1]))
            A = A / np.sqrt(N2) * NS
        else:
            mv = np.arange(2, p + 1, 2).reshape(-1, 1)
            N2 = np.sqrt(A[0, n - 1]**2 * 2 * gamma(p / 2 + 1)**2 + 
                        np.sum((np.sqrt(gamma((p + mv) / 2 + 1) * gamma((p - mv) / 2 + 1)) * 
                               A[1:p // 2 + 1, n - 1].reshape(-1, 1))**2))
            NS = np.sign(np.sum(A[:, n - 1]))
            A = A / N2 * NS
        
        # Ince Polynomial
        r = np.arange(N)
        R, X = np.meshgrid(r, z)
        IP = np.cos(2 * X * R) @ A[:, n - 1]
        dIP = -2 * R * np.sin(2 * X * R) @ A[:, n - 1]
        eta = ets[n - 1]
    else:
        j = (p - 1) // 2
        N = j + 1
        n = (m + 1) // 2
        
        # Matrix
        M = (np.diag(q / 2 * (p + (2 * np.arange(N - 1) + 3)), 1) + 
             np.diag(q / 2 * (p - (2 * np.arange(1, N) - 1)), -1) + 
             np.diag(np.concatenate([[q / 2 + p * q / 2 + 1], (2 * np.arange(1, N) + 1)**2])))
        
        # Eigenvalues and Eigenvectors
        ets, A = np.linalg.eig(M)
        index = np.argsort(ets)
        ets = ets[index]
        A = A[:, index]
        
        # Normalization
        if normalization == 0:
            N2 = np.sum(A[:, n - 1]**2)
            NS = np.sign(np.sum(A[:, n - 1]))
            A = A / np.sqrt(N2) * NS
        else:
            mv = np.arange(1, p + 1, 2).reshape(-1, 1)
            N2 = np.sqrt(np.sum((np.sqrt(gamma((p + mv) / 2 + 1) * gamma((p - mv) / 2 + 1)) * 
                                A[:, n - 1].reshape(-1, 1))**2))
            NS = np.sign(np.sum(A[:, n - 1]))
            A = A / N2 * NS
        
        # Ince Polynomial
        r = 2 * np.arange(N) + 1
        R, X = np.meshgrid(r, z)
        IP = np.cos(X * R) @ A[:, n - 1]
        dIP = -R * np.sin(X * R) @ A[:, n - 1]
        eta = ets[n - 1]
    
    coef = A[:, n - 1]
    IP = IP.reshape(largo, ancho)
    dIP = dIP.reshape(largo, ancho)
    
    return IP, eta, coef, dIP


def _SInceIGB(p, m, q, z):
    """Odd Ince Polynomial."""
    if m < 1 or m > p:
        raise ValueError('ERROR: Wrong range for "m", 1<=m<=p')
    if (-1)**(m - p) != 1:
        raise ValueError('ERROR: (p,m) must have the same parity')
    
    largo, ancho = z.shape if hasattr(z, 'shape') else (1, len(z))
    z = z.flatten()
    normalization = 1
    
    if p % 2 == 0:
        j = p // 2
        N = j + 1
        n = m // 2
        
        # Matrix
        M = (np.diag(q * (j + np.arange(2, N)), 1) + 
             np.diag(q * (j - np.arange(1, N - 1)), -1) + 
             np.diag(4 * (np.arange(N - 1) + 1)**2))
        
        # Eigenvalues and Eigenvectors
        ets, A = np.linalg.eig(M)
        index = np.argsort(ets)
        ets = ets[index]
        A = A[:, index]
        
        # Normalization
        r = np.arange(1, N)
        if normalization == 0:
            N2 = np.sum(A[:, n - 1]**2)
            NS = np.sign(np.sum(r * A[:, n - 1]))
            A = A / np.sqrt(N2) * NS
        else:
            mv = np.arange(2, p + 1, 2).reshape(-1, 1)
            N2 = np.sqrt(np.sum((np.sqrt(gamma((p + mv) / 2 + 1) * gamma((p - mv) / 2 + 1)) * 
                                A[1:p // 2 + 1, n - 1].reshape(-1, 1))**2))
            NS = np.sign(np.sum(r * A[:, n - 1]))
            A = A / N2 * NS
        
        # Ince Polynomial
        R, X = np.meshgrid(r, z)
        IP = np.sin(2 * X * R) @ A[:, n - 1]
        dIP = 2 * R * np.cos(2 * X * R) @ A[:, n - 1]
        eta = ets[n - 1]
    else:
        j = (p - 1) // 2
        N = j + 1
        n = (m + 1) // 2
        
        # Matrix
        M = (np.diag(q / 2 * (p + (2 * np.arange(N - 1) + 3)), 1) + 
             np.diag(q / 2 * (p - (2 * np.arange(1, N) - 1)), -1) + 
             np.diag(np.concatenate([[-q / 2 - p * q / 2 + 1], (2 * np.arange(1, N) + 1)**2])))
        
        # Eigenvalues and Eigenvectors
        ets, A = np.linalg.eig(M)
        index = np.argsort(ets)
        ets = ets[index]
        A = A[:, index]
        
        # Normalization
        r = 2 * np.arange(N) + 1
        if normalization == 0:
            N2 = np.sum(A[:, n - 1]**2)
            NS = np.sign(np.sum(r * A[:, n - 1]))
            A = A / np.sqrt(N2) * NS
        else:
            mv = np.arange(1, p + 1, 2).reshape(-1, 1)
            N2 = np.sqrt(np.sum((np.sqrt(gamma((p + mv) / 2 + 1) * gamma((p - mv) / 2 + 1)) * 
                                A[:, n - 1].reshape(-1, 1))**2))
            NS = np.sign(np.sum(r * A[:, n - 1]))
            A = A / N2 * NS
        
        # Ince Polynomial
        R, X = np.meshgrid(r, z)
        IP = np.sin(X * R) @ A[:, n - 1]
        dIP = R * np.cos(X * R) @ A[:, n - 1]
        eta = ets[n - 1]
    
    coef = A[:, n - 1]
    IP = IP.reshape(largo, ancho)
    dIP = dIP.reshape(largo, ancho)
    
    return IP, eta, coef, dIP
