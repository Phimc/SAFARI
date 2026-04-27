import numpy as np

def visualize_complex(cimg, cmap, amp_range, mode='hsv', reverse=False):
    """
    Visualize complex image using amplitude-phase encoding.
    
    Parameters:
    -----------
    cimg : ndarray
        Complex image
    cmap : ndarray
        Colormap (N x 3)
    amp_range : list
        Amplitude range [min, max]
    mode : str
        'hsv' or 'hsl'
    reverse : bool
        Reverse amplitude
    
    Returns:
    --------
    img : ndarray
        RGB image
    cbarimg : ndarray
        Colorbar image
    """
    amp = np.abs(cimg)
    pha = np.angle(cimg)
    
    amin, amax = amp_range
    amp_norm = (amp - amin) / (amax - amin)
    amp_norm = np.clip(amp_norm, 0, 1)
    
    b = 1
    ncmap = len(cmap)
    
    # Create image
    img = np.zeros((cimg.shape[0], cimg.shape[1], 3))
    
    for i in range(cimg.shape[0]):
        for j in range(cimg.shape[1]):
            a = amp_norm[i, j]
            if reverse:
                a = 1 - a
            
            cmap_idx = int(round((ncmap - 1) / (2 * np.pi) * (pha[i, j] + np.pi)))
            cmap_idx = np.clip(cmap_idx, 0, ncmap - 1)
            color = cmap[cmap_idx]
            
            if mode.lower() == 'hsv':
                img[i, j, :] = color * a
            elif mode.lower() == 'hsl':
                if a > 1/2:
                    w = (2 * (a - 1/2))**b
                    img[i, j, :] = color * (1 - w) + np.array([1, 1, 1]) * w
                else:
                    w = (2 * (1/2 - a))**b
                    img[i, j, :] = color * (1 - w) + np.array([0, 0, 0]) * w
    
    # Create colorbar
    n = 256
    cbarimg = np.full((n, n, 3), np.nan)
    x = np.linspace(-1, 1, n)
    y = x
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    rho = np.sqrt(X**2 + Y**2)
    
    for i in range(n):
        for j in range(n):
            if rho[i, j] <= 1:
                a = rho[i, j]
                if reverse:
                    a = 1 - a
                
                cmap_idx = int(round((ncmap - 1) / (2 * np.pi) * (theta[i, j] + np.pi)))
                cmap_idx = np.clip(cmap_idx, 0, ncmap - 1)
                color = cmap[cmap_idx]
                
                if mode.lower() == 'hsv':
                    cbarimg[i, j, :] = color * a
                elif mode.lower() == 'hsl':
                    if a > 1/2:
                        w = (2 * (a - 1/2))**b
                        cbarimg[i, j, :] = color * (1 - w) + np.array([1, 1, 1]) * w
                    else:
                        w = (2 * (1/2 - a))**b
                        cbarimg[i, j, :] = color * (1 - w) + np.array([0, 0, 0]) * w
    
    return img, cbarimg
