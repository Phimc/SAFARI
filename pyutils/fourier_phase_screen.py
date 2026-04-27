import numpy as np

def fourier_phase_screen(N, dx, r0, rs, numsub=0):
    """
    Generate Fourier phase screen for turbulence simulation.
    
    Parameters:
    -----------
    N : int
        Number of pixels in real space
    dx : float
        Pixel size in real space
    r0 : float
        Fried parameter
    rs : int
        Random seed for repeatability
    numsub : int
        Number of subharmonics for subharmonic sampling
    
    Returns:
    --------
    screen : ndarray
        Phase screen
    """
    np.random.seed(rs)
    
    # Real Space Parameters
    Lx = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    X, Y = np.meshgrid(x, x)
    
    # Fourier Space Parameters
    df = 1 / Lx
    f = np.arange(-N / 2, N / 2) * df
    Fx, Fy = np.meshgrid(f, f)
    F = np.sqrt(Fx**2 + Fy**2)
    
    # High Frequency Screen
    PSD = 0.023 * r0**(-5 / 3) * F**(-11 / 3)
    PSD[N // 2, N // 2] = 0  # Set zero frequency to zero
    
    cnm_high = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * np.sqrt(PSD / (Lx * Lx))
    cnm_high[N // 2, N // 2] = 0
    
    screen_high = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(cnm_high))) * N**2
    screen = screen_high
    
    if numsub > 0:
        np.random.seed(rs)
        screen_low = np.zeros((N, N))
        
        for b in range(1, numsub + 1):
            dfs = (1 / (3**b)) * df
            fs = np.array([-1, 0, 1]) * dfs
            Fxs, Fys = np.meshgrid(fs, fs)
            Fs = np.sqrt(Fxs**2 + Fys**2)
            
            PSDs = 0.023 * r0**(-5 / 3) * Fs**(-11 / 3)
            cnm_low = ((np.random.randn(3, 3) + 1j * np.random.randn(3, 3)) * 
                      np.sqrt(PSDs / (Lx * Lx)) * (1 / 3**b))
            cnm_low[1, 1] = 0
            
            sh = np.zeros((N, N), dtype=complex)
            for n_idx in range(3):
                for m_idx in range(3):
                    sh = sh + cnm_low[n_idx, m_idx] * np.exp(1j * 2 * np.pi * 
                                                              (Fxs[n_idx, m_idx] * X + 
                                                               Fys[n_idx, m_idx] * Y))
            
            screen_low = screen_low + sh
        
        screen = screen_high + screen_low
    
    screen = np.real(screen) - np.mean(np.real(screen))
    
    return screen
