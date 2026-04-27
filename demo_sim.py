"""
SAFARI: Spatial And Fourier-domAin Regularized Inversion
Python implementation of wavefront reconstruction demo.

Author: Converted from MATLAB implementation by Yunhui Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import time

# Import utils
from pyutils import (
    propagate, aperture, inferno, sinebow, visualize_complex,
    gen_laguerre_gaussian, gen_hermite_gaussian, 
    gen_airy, gen_bessel, gen_ince_gaussian, gen_vortex,
    fourier_phase_screen
)
from pyutils.zernfun import zernfun


# =========================================================================
# Auxiliary functions
# =========================================================================

def imgcrop(x, cropsize):
    """Crop the central part of the image."""
    return x[cropsize:-cropsize, cropsize:-cropsize]


def zeropad(x, padsize):
    """Zero-pad the image."""
    return np.pad(x, ((padsize, padsize), (padsize, padsize)), mode='constant')


def transfunc_propagate(n1, n2, dist, pxsize, wavlen):
    """Calculate the transfer function of the free-space diffraction."""
    k1 = np.pi / pxsize * np.linspace(-1, 1 - 2 / n1, n1)
    k2 = np.pi / pxsize * np.linspace(-1, 1 - 2 / n2, n2)
    K2, K1 = np.meshgrid(k2, k1)
    
    k = 2 * np.pi / wavlen
    
    ind = (K1**2 + K2**2 >= k**2)
    K1[ind] = 0
    K2[ind] = 0
    
    H = np.exp(1j * dist * np.sqrt(k**2 - K1**2 - K2**2))
    return H


def Df(x):
    """Calculate the 2D gradient (finite difference) of an input image."""
    w = np.stack([
        x - np.roll(x, -1, axis=0),
        x - np.roll(x, -1, axis=1)
    ], axis=-1)
    return w


def DTf(w):
    """Calculate the transpose of the gradient operator."""
    u1 = w[:, :, 0] - np.roll(w[:, :, 0], 1, axis=0)
    u1[0, :] = w[0, :, 0]
    u1[-1, :] = -w[-2, :, 0]
    
    u2 = w[:, :, 1] - np.roll(w[:, :, 1], 1, axis=1)
    u2[:, 0] = w[:, 0, 1]
    u2[:, -1] = -w[:, -2, 1]
    
    u = u1 + u2
    return u


# =========================================================================
# Main simulation
# =========================================================================

def main():
    # Key parameters
    params = {
        'pxsize': 2.4e-3,    # pixel size (mm)
        'wavlen': 0.532e-3,  # wavelength (mm)
    }

    n = 1000  # wavefront dimension

    # =========================================================================
    # Wavefront generation (uncomment the corresponding block)
    # =========================================================================

    # # speckle fields
    # np.random.seed(0)
    # grain_size = 8
    # m = round(n / grain_size)
    # m = round((m + 1) / 2) * 2
    # u = np.exp(1j * np.random.rand(m, m) * 2 * np.pi)
    # wavefront = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
    #     np.pad(u, ((n - m) // 2, (n - m) // 2), mode='constant'))))

    # # Laguerre Gaussian beams
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # z = 0
    # w0 = 0.2
    # l = 3
    # p = 3
    # wavefront = gen_laguerre_gaussian(X, Y, z, params['wavlen'], w0, l, p)

    # # Hermite Gaussian beams
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # z = 0
    # w0 = 0.2
    # mi = 3
    # ni = 3
    # wavefront = gen_hermite_gaussian(X, Y, z, params['wavlen'], w0, mi, ni)

    # # Ince Gaussian beams
    # z = 0
    # w0 = 0.2
    # p = 12
    # m = 8
    # e = 2
    # parity = 0
    # wavefront, _, _ = gen_ince_gaussian(
    #     params['pxsize'] * n / 2, n + 1, parity, p, m, e, w0, 
    #     2 * np.pi / params['wavlen'], z)
    # wavefront = wavefront[:n, :n]

    # # Airy beams
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # w0 = 0.2
    # x0 = -n * 0.4 * params['pxsize']
    # y0 = -n * 0.4 * params['pxsize']
    # a = 1e-3
    # wavefront = gen_airy(X, Y, w0, x0, y0, a)

    # # Bessel beams
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # z = 0
    # n_charge = 3
    # theta = np.pi / 3e3
    # wavefront = gen_bessel(X, Y, z, params['wavlen'], n_charge, theta)

    # # parabolic phase
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # f = 200
    # a = 0.2
    # k = 2 * np.pi / params['wavlen']
    # wavefront = np.exp(-a * (X**2 + Y**2)) * np.exp(-1j * k * (X**2 + Y**2) / (2 * f))

    # # Zernike aberrations
    # X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    # theta = np.arctan2(Y, X)
    # r = np.sqrt(X**2 + Y**2)
    # idx = r <= 1
    # n_max = 5
    # s_fac = 4
    # n_modes = (n_max + 2) * (n_max + 1) // 2
    # z_n = np.full(n_modes, np.nan)
    # z_m = np.full(n_modes, np.nan)
    # for i in range(n_max + 1):
    #     start = i * (i + 1) // 2
    #     end = (i + 1) * (i + 2) // 2
    #     z_n[start:end] = i
    #     z_m[start:end] = np.arange(-i, i + 1, 2)
    # np.random.seed(0)
    # coef = 2 * np.random.rand(n_modes) - 1
    # zer = np.zeros((n, n))
    # for i in range(n_modes):
    #     bfun = np.zeros((n, n))
    #     bfun[idx] = zernfun(z_n[i], z_m[i], r[idx], theta[idx]).flatten()
    #     zer[idx] = zer[idx] + coef[i] * bfun[idx]
    # phase = 2 * np.pi * s_fac * zer
    # # Crop central region
    # crop_start = int(np.ceil(n / 2 - np.sqrt(2) / 4 * n + 1))
    # crop_end = int(np.floor(n / 2 + np.sqrt(2) / 4 * n - 1))
    # phase = zoom(phase[crop_start:crop_end, crop_start:crop_end], 
    #              (n / (crop_end - crop_start), n / (crop_end - crop_start)))
    # a = 1
    # wavefront = np.exp(-a * (X**2 + Y**2)) * np.exp(1j * phase)

    # # turbulence (single phase screen)
    # seed = 0
    # numsub = 10
    # r0 = 0.1
    # phase = fourier_phase_screen(n, params['pxsize'], r0, seed, numsub)
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # a = 0.1
    # wavefront = np.exp(-a * (X**2 + Y**2)) * np.exp(1j * phase)

    # # turbulence (scintillated wavefront)
    # # NOTE: This requires gen_turbulence which is complex to implement in Python
    # # and requires the Optimization Toolbox in MATLAB.

    # # amplitude pattern
    # img = np.array(Image.open('data/thulogo.bmp').convert('L')).astype(np.float64) / 255.0
    # img = zoom(img, (n / 2 / img.shape[0], n / 2 / img.shape[1]))
    # img = np.pad(1 - img, ((n // 4, n // 4), (n // 4, n // 4)), mode='constant')
    # amp = img
    # pha = np.zeros((n, n))
    # wavefront = amp * np.exp(1j * pha)

    # phase pattern
    img = np.array(Image.open('data/cityulogo.bmp').convert('L')).astype(np.float64) / 255.0
    img = zoom(img, (n / 2 / img.shape[0], n / 2 / img.shape[1]))
    img = np.pad(1 - img, ((n // 4, n // 4), (n // 4, n // 4)), mode='constant')
    amp = np.ones((n, n))
    pha = img * np.pi
    wavefront = amp * np.exp(1j * pha)

    # # prism / tilted wavefront
    # X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
    #                    np.arange(-n / 2, n / 2) * params['pxsize'])
    # kmax = 2 * np.pi / params['pxsize'] / 4 / 2
    # kx = kmax * 0.5
    # ky = kmax * 0.0
    # pha = kx * X + ky * Y
    # amp = np.ones((n, n))
    # wavefront = amp * np.exp(1j * pha)

    # =========================================================================
    # Wavefront tilt / propagation
    # =========================================================================

    # wavefront tilt
    X, Y = np.meshgrid(np.arange(-n / 2, n / 2) * params['pxsize'],
                       np.arange(-n / 2, n / 2) * params['pxsize'])
    k_alpha_x = 0.0
    k_alpha_y = 0.0
    kx = 2 * np.pi / params['wavlen'] * np.sin(np.deg2rad(k_alpha_x))
    ky = 2 * np.pi / params['wavlen'] * np.sin(np.deg2rad(k_alpha_y))
    pha = kx * X + ky * Y
    wavefront = wavefront * np.exp(1j * pha)

    # wavefront propagation
    prop_dist = 0
    wavefront = propagate(wavefront, prop_dist, params['pxsize'], params['wavlen'])

    # =========================================================================
    # Measurement simulation
    # =========================================================================

    # normalization of the amplitude
    wavefront = wavefront / np.max(np.abs(wavefront))

    # physical parameters
    params['dist'] = 3.6  # diffuser-to-sensor distance (mm)
    cropsize = 50

    # define the diffuser profile (random binary phase modulation)
    np.random.seed(0)
    diff_feat_size = 5
    diffuser_small = np.random.rand(n // diff_feat_size, n // diff_feat_size)
    diffuser = zoom(diffuser_small, (diff_feat_size, diff_feat_size), order=0)

    index_1 = diffuser < 0.5
    index_2 = diffuser >= 0.5
    diffuser = np.zeros_like(diffuser, dtype=complex)
    diffuser[index_1] = np.exp(1j * 0)
    diffuser[index_2] = np.exp(1j * np.pi)

    # calculate the transfer function for diffraction modeling
    HQ = np.fft.fftshift(transfunc_propagate(n, n, params['dist'], 
                                              params['pxsize'], params['wavlen']))

    # define function handles for the measurement operators
    def M(x):
        return x * diffuser

    def MH(x):
        return x * np.conj(diffuser)

    def Q(x):
        return np.fft.ifft2(np.fft.fft2(x) * HQ)

    def QH(x):
        return np.fft.ifft2(np.fft.fft2(x) * np.conj(HQ))

    def C(x):
        return imgcrop(x, cropsize)

    def CH(x):
        return zeropad(x, cropsize)

    def A(x):
        return C(Q(M(x)))

    def AH(x):
        return MH(QH(CH(x)))

    # generate measurement data
    np.random.seed(0)
    x = wavefront
    y = np.abs(A(x))**2
    y = y / np.max(y)
    y = np.maximum(y + 1e-3 * np.random.randn(*y.shape), 0)

    # display the wavefront and measurement
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cmap_g = 'gray'
    cmap_i = inferno()
    cmap_s = sinebow()

    im1 = axes[0].imshow(np.abs(wavefront), cmap=cmap_g, vmin=0, vmax=1)
    axes[0].set_title('Amplitude of the wavefront')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.angle(wavefront), cmap='inferno', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase of the wavefront')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(y, cmap=cmap_g, vmin=0, vmax=1)
    axes[2].set_title('Encoded intensity image')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('measurement.png', dpi=150)
    plt.show()

    print("Measurement simulation complete.")

    # =========================================================================
    # Wavefront reconstruction algorithm
    # =========================================================================

    # set the regularization parameters
    lam_c = [1e-2, 1e-2]  # regularization parameter (complex amplitude) [start, end]
    lam_a = [1e-2, 1e-2]  # regularization parameter (amplitude) [start, end]
    alpha = 1e-2  # parameter tuning weight

    # set the support region in the Fourier domain
    R = n / 8
    support = aperture(n, n, n / 2, n / 2, R)

    # Lipschitz bound for the fidelity term
    LF = 1 * np.max(np.abs(diffuser))**2

    # algorithm settings
    gpu = False  # whether using GPU or not
    display = False  # whether display intermediate results
    verbose = True  # whether print status

    term_std = 1e-4  # termination criteria
    cache_iter = 10  # number of previous loss values stored
    cache_loss = np.full(cache_iter, np.inf)

    # optimization variables
    seed = 0
    np.random.seed(seed)
    x_est = np.ones((n, n)) * np.exp(1j * 2 * np.pi * np.random.rand(n, n))
    z_est = x_est.copy()

    # main loop
    timer = time.time()
    loss = np.inf
    iter_num = 1

    while True:
        # update regularization parameters
        lam_a_val = (lam_a[0] - lam_a[1]) * np.exp(-alpha * (iter_num - 1)) + lam_a[1]
        lam_c_val = (lam_c[0] - lam_c[1]) * np.exp(-alpha * (iter_num - 1)) + lam_c[1]
        
        # gradient calculation
        u = A(z_est)
        res = np.abs(u) - np.sqrt(y)
        u = u * (res / (np.abs(u) + 1e-10))  # avoid division by zero
        g = AH(u)
        g = g + lam_c_val * DTf(Df(z_est))
        g = g + lam_a_val * np.exp(1j * np.angle(z_est)) * DTf(Df(np.abs(z_est)))
        
        # proximal gradient update
        gamma = 1 / (LF + (lam_a_val + lam_c_val) * 8)
        u = z_est - gamma * g
        x_next = np.fft.ifft2(np.fft.fftshift(support * np.fft.fftshift(np.fft.fft2(u))))
        
        z_est = x_next + iter_num / (iter_num + 3) * (x_next - x_est)
        x_est = x_next.copy()
        
        loss = np.linalg.norm(res.flatten(), 2)**2
        
        # Compute criterion (handle initial iterations with inf values)
        valid_loss = cache_loss[np.isfinite(cache_loss)]
        if len(valid_loss) >= cache_iter // 2:
            criterion = np.std(valid_loss) / np.mean(valid_loss)
        else:
            criterion = np.inf
        
        # print status
        if verbose:
            print(f'iter: {iter_num:4d} | loss: {loss:5.2e} | '
                  f'loss std: {criterion:5.2e} | '
                  f'runtime: {time.time() - timer:5.1f} s')
        
        # terminate if criterion is met
        if criterion < term_std and iter_num > cache_iter:
            print('Terminated.')
            break
        
        # update loss cache
        idx = (iter_num - 1) % cache_iter
        cache_loss[idx] = loss
        
        iter_num += 1
        
        # safety break
        if iter_num > 10000:
            print('Max iterations reached.')
            break

    print(f'Total runtime: {time.time() - timer:.1f} s')

    # =========================================================================
    # Display results
    # =========================================================================

    a_max = np.percentile(np.abs(x_est), 99.99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(np.abs(C(x_est)), cmap=cmap_g)
    axes[0].set_title('Reconstructed amplitude')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.angle(C(x_est)), cmap='inferno', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Reconstructed phase')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150)
    plt.show()

    # Complex visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img_gt, _ = visualize_complex(C(x), cmap_s, [0, 1], 'hsv', False)
    axes[0].imshow(img_gt)
    axes[0].set_title('Ground-truth wavefront')

    img_rec, _ = visualize_complex(C(x_est), cmap_s, [0, a_max], 'hsv', False)
    axes[1].imshow(img_rec)
    axes[1].set_title('Reconstructed wavefront')

    img_conj, _ = visualize_complex(C(x_est * np.exp(-1j * np.angle(x))), 
                                     cmap_s, [0, a_max], 'hsv', False)
    axes[2].imshow(img_conj)
    axes[2].set_title('Phase-conjugated wavefront')

    plt.tight_layout()
    plt.savefig('complex_visualization.png', dpi=150)
    plt.show()

    print("Reconstruction complete!")


if __name__ == '__main__':
    main()
