"""
Quick test of SAFARI Python implementation with smaller dimensions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import time

from pyutils import (
    propagate, aperture, inferno, sinebow, visualize_complex,
    fourier_phase_screen
)

# Test with smaller dimensions
n = 200

params = {
    'pxsize': 2.4e-3,
    'wavlen': 0.532e-3,
}

# Test phase pattern wavefront generation
print("Testing wavefront generation...")
img = np.array(Image.open('data/cityulogo.bmp').convert('L')).astype(np.float64) / 255.0
img = zoom(img, (n / 2 / img.shape[0], n / 2 / img.shape[1]))
img = np.pad(1 - img, ((n // 4, n // 4), (n // 4, n // 4)), mode='constant')
amp = np.ones((n, n))
pha = img * np.pi
wavefront = amp * np.exp(1j * pha)

print(f"Wavefront shape: {wavefront.shape}")
print(f"Wavefront amplitude range: [{np.min(np.abs(wavefront)):.3f}, {np.max(np.abs(wavefront)):.3f}]")

# Test propagation
print("\nTesting propagation...")
wavefront = propagate(wavefront, 0, params['pxsize'], params['wavlen'])
print("Propagation OK")

# Test measurement simulation
print("\nTesting measurement simulation...")
wavefront = wavefront / np.max(np.abs(wavefront))
params['dist'] = 3.6
cropsize = 10

np.random.seed(0)
diff_feat_size = 5
diffuser_small = np.random.rand(n // diff_feat_size, n // diff_feat_size)
diffuser = zoom(diffuser_small, (diff_feat_size, diff_feat_size), order=0)

index_1 = diffuser < 0.5
index_2 = diffuser >= 0.5
diffuser = np.zeros_like(diffuser, dtype=complex)
diffuser[index_1] = np.exp(1j * 0)
diffuser[index_2] = np.exp(1j * np.pi)

# Test transfer function
from demo_sim import transfunc_propagate, imgcrop, zeropad, Df, DTf

HQ = np.fft.fftshift(transfunc_propagate(n, n, params['dist'], 
                                          params['pxsize'], params['wavlen']))

def M(x): return x * diffuser
def MH(x): return x * np.conj(diffuser)
def Q(x): return np.fft.ifft2(np.fft.fft2(x) * HQ)
def QH(x): return np.fft.ifft2(np.fft.fft2(x) * np.conj(HQ))
def C(x): return imgcrop(x, cropsize)
def CH(x): return zeropad(x, cropsize)
def A(x): return C(Q(M(x)))
def AH(x): return MH(QH(CH(x)))

np.random.seed(0)
x = wavefront
y = np.abs(A(x))**2
y = y / np.max(y)
y = np.maximum(y + 1e-3 * np.random.randn(*y.shape), 0)

print(f"Measurement shape: {y.shape}")
print("Measurement simulation OK")

# Test reconstruction algorithm (few iterations)
print("\nTesting reconstruction algorithm (10 iterations)...")
lam_c = [1e-2, 1e-2]
lam_a = [1e-2, 1e-2]
alpha = 1e-2

R = n / 8
support = aperture(n, n, n / 2, n / 2, R)

LF = 1 * np.max(np.abs(diffuser))**2

term_std = 1e-4
cache_iter = 10
cache_loss = np.full(cache_iter, np.inf)

np.random.seed(0)
x_est = np.ones((n, n)) * np.exp(1j * 2 * np.pi * np.random.rand(n, n))
z_est = x_est.copy()

timer = time.time()

for iter_num in range(1, 11):
    lam_a_val = (lam_a[0] - lam_a[1]) * np.exp(-alpha * (iter_num - 1)) + lam_a[1]
    lam_c_val = (lam_c[0] - lam_c[1]) * np.exp(-alpha * (iter_num - 1)) + lam_c[1]
    
    u = A(z_est)
    res = np.abs(u) - np.sqrt(y)
    u = u * (res / (np.abs(u) + 1e-10))
    g = AH(u)
    g = g + lam_c_val * DTf(Df(z_est))
    g = g + lam_a_val * np.exp(1j * np.angle(z_est)) * DTf(Df(np.abs(z_est)))
    
    gamma = 1 / (LF + (lam_a_val + lam_c_val) * 8)
    u = z_est - gamma * g
    x_next = np.fft.ifft2(np.fft.fftshift(support * np.fft.fftshift(np.fft.fft2(u))))
    
    z_est = x_next + iter_num / (iter_num + 3) * (x_next - x_est)
    x_est = x_next.copy()
    
    loss = np.linalg.norm(res.flatten(), 2)**2
    
    print(f'  iter: {iter_num:2d} | loss: {loss:5.2e} | time: {time.time() - timer:.2f}s')

print(f"\nReconstruction test complete in {time.time() - timer:.1f}s")

# Test visualization
print("\nTesting visualization...")
cmap_s = sinebow()
img_rec, _ = visualize_complex(C(x_est), cmap_s, [0, 1], 'hsv', False)
print(f"Visualization output shape: {img_rec.shape}")
print("Visualization OK")

print("\n=== All tests passed! ===")
