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
    
    # Vectorized color index calculation
    cmap_idx = np.round((ncmap - 1) / (2 * np.pi) * (pha + np.pi)).astype(int)
    cmap_idx = np.clip(cmap_idx, 0, ncmap - 1)
    
    # Get colors for all pixels
    colors = cmap[cmap_idx]  # shape: (H, W, 3)
    
    a = amp_norm.copy()
    if reverse:
        a = 1 - a
    
    if mode.lower() == 'hsv':
        img = colors * a[..., None]
    elif mode.lower() == 'hsl':
        img = np.zeros_like(colors)
        mask_high = a > 0.5
        w_high = (2 * (a - 0.5))**b
        w_low = (2 * (0.5 - a))**b
        
        for c in range(3):
            img[..., c] = np.where(
                mask_high,
                colors[..., c] * (1 - w_high) + w_high,
                colors[..., c] * (1 - w_low) + 0
            )
    
    # Create colorbar
    n = 256
    x = np.linspace(-1, 1, n)
    X_cb, Y_cb = np.meshgrid(x, x)
    theta_cb = np.arctan2(Y_cb, X_cb)
    rho_cb = np.sqrt(X_cb**2 + Y_cb**2)
    
    mask = rho_cb <= 1
    cbarimg = np.full((n, n, 3), np.nan)
    
    cmap_idx_cb = np.round((ncmap - 1) / (2 * np.pi) * (theta_cb + np.pi)).astype(int)
    cmap_idx_cb = np.clip(cmap_idx_cb, 0, ncmap - 1)
    colors_cb = cmap[cmap_idx_cb]
    
    a_cb = rho_cb.copy()
    if reverse:
        a_cb = 1 - a_cb
    
    if mode.lower() == 'hsv':
        cbarimg[mask] = (colors_cb * a_cb[..., None])[mask]
    elif mode.lower() == 'hsl':
        mask_high = mask & (a_cb > 0.5)
        mask_low = mask & (a_cb <= 0.5)
        w_high = (2 * (a_cb - 0.5))**b
        w_low = (2 * (0.5 - a_cb))**b
        for c in range(3):
            cbarimg[..., c] = np.where(
                mask_high,
                colors_cb[..., c] * (1 - w_high) + w_high,
                np.where(mask_low, colors_cb[..., c] * (1 - w_low) + 0, np.nan)
            )
    
    return img, cbarimg
