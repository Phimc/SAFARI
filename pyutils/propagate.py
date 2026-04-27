import numpy as np

def propagate(w_i, dist, pxsize, wavlen):
    """
    Propagate wavefield using angular spectrum method.
    
    Parameters:
    -----------
    w_i : ndarray
        Input wavefield
    dist : float
        Propagation distance
    pxsize : float
        Pixel size
    wavlen : float
        Wavelength
    
    Returns:
    --------
    w_o : ndarray
        Output wavefield
    """
    ny, nx = w_i.shape
    
    # sampling in the frequency domain
    kx = np.pi/pxsize * np.linspace(-1, 1-2/nx, nx)
    ky = np.pi/pxsize * np.linspace(-1, 1-2/ny, ny)
    KX, KY = np.meshgrid(kx, ky)
    
    # wave number
    k = 2*np.pi/wavlen
    
    # circular convolution via ffts
    inputFT = np.fft.fftshift(np.fft.fft2(w_i))
    H = np.exp(1j*dist*np.sqrt(k**2 - KX**2 - KY**2))
    w_o = np.fft.ifft2(np.fft.ifftshift(inputFT * H))
    
    return w_o
