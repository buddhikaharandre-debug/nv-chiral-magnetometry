# ============================================================
# FULL PHYSICAL STRIPE MODEL (Bx + Bz + NV PROJECTION)
# + AUTOMATIC DOMAIN DETECTION + IMAGE DISPLAY
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, sobel, map_coordinates
from scipy.optimize import curve_fit

# ---------------------- LOAD DATA ----------------------
base_dir = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\test"
B_nv = np.load(os.path.join(base_dir, "B_NV_smoothed.npy"))  # in mT

# 🔥 Convert to Tesla
B_nv_T = B_nv * 1e-3

# ---------------------- CONSTANTS ----------------------
mu0 = 4*np.pi*1e-7
theta = np.deg2rad(54.7)
phi = np.deg2rad(45)

pixel_size = 0.325  # µm

# ---------------------- MATERIAL ----------------------
Mz = 1.0e6       # A/m (adjust if known)
t_film = 0.9e-9  # m (CoFeB thickness)

prefactor = mu0 * Mz * t_film / (2*np.pi)

# ---------------------- SMOOTH ----------------------
B_smooth = gaussian_filter(B_nv_T, sigma=1)

# ---------------------- GRADIENT ----------------------
gx = sobel(B_smooth, axis=1)
gy = sobel(B_smooth, axis=0)
grad_mag = np.sqrt(gx**2 + gy**2)

# ---------------------- FIND STRIPE EDGES ----------------------
N = 5
flat_idx = np.argsort(grad_mag.ravel())[::-1][:N]
points = [np.unravel_index(i, grad_mag.shape) for i in flat_idx]

best_score = -np.inf
best_pair = None

for i in range(len(points)):
    for j in range(i+1, len(points)):

        y1, x1 = points[i]
        y2, x2 = points[j]

        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2) * pixel_size

        if dist < 0.5 or dist > 5:
            continue

        g1 = np.array([gx[y1,x1], gy[y1,x1]])
        g2 = np.array([gx[y2,x2], gy[y2,x2]])

        g1 /= np.linalg.norm(g1) + 1e-12
        g2 /= np.linalg.norm(g2) + 1e-12

        alignment = np.dot(g1, g2)
        score = -alignment * grad_mag[y1,x1] * grad_mag[y2,x2]

        if score > best_score:
            best_score = score
            best_pair = (y1,x1,y2,x2)

# ---------------------- LINE EXTRACTION ----------------------
y1, x1, y2, x2 = best_pair

yc = (y1 + y2) / 2
xc = (x1 + x2) / 2

dx = gx[int(yc), int(xc)]
dy = gy[int(yc), int(xc)]

norm = np.sqrt(dx**2 + dy**2) + 1e-12
dx /= norm
dy /= norm

length_um = 4.0
n = 100

t = np.linspace(-length_um/2, length_um/2, n)

x_line = xc + (t / pixel_size) * dx
y_line = yc + (t / pixel_size) * dy

# ---------------------- PROFILE ----------------------
profile = map_coordinates(B_nv_T, [y_line, x_line], order=1)

distance = t * 1e-6  # meters

# ---------------------- SMOOTH ----------------------
profile_smooth = gaussian_filter(profile, sigma=3)

# ---------------------- 🔥 FULL PHYSICAL MODEL ----------------------
def stripe_model_full(x, x1, x2, d, S, offset):

    Bx = prefactor * (
        - d / ((x - x1)**2 + d**2)
        + d / ((x - x2)**2 + d**2)
    )

    Bz = prefactor * (
        (x - x1)/((x - x1)**2 + d**2)
        - (x - x2)/((x - x2)**2 + d**2)
    )

    Bnv = Bx * np.sin(theta) * np.cos(phi) + Bz * np.cos(theta)

    return S * Bnv + offset

# ---------------------- INITIAL GUESS ----------------------
grad_1d = np.gradient(profile_smooth)

i1 = np.argmax(grad_1d)
i2 = np.argmin(grad_1d)

x1_guess = distance[i1]
x2_guess = distance[i2]

if x1_guess > x2_guess:
    x1_guess, x2_guess = x2_guess, x1_guess

d_guess = 100e-9
S_guess = 1e3
offset_guess = np.mean(profile_smooth)

p0 = [x1_guess, x2_guess, d_guess, S_guess, offset_guess]

# ---------------------- FIT ----------------------
params, cov = curve_fit(
    stripe_model_full,
    distance,
    profile_smooth,
    p0=p0,
    maxfev=100000
)

x1_fit, x2_fit, d_fit, S_fit, offset_fit = params
d_err = np.sqrt(np.diag(cov))[2]

# ---------------------- OUTPUT ----------------------
print("\n========== FINAL PHYSICAL RESULT ==========")
print(f"Stripe width = {(x2_fit-x1_fit)*1e6:.2f} µm")
print(f"Standoff distance z = {d_fit*1e9:.2f} ± {d_err*1e9:.2f} nm")

# ============================================================
# 🔴 PLOT 1: IMAGE WITH COLORBAR
# ============================================================

plt.figure(figsize=(6,5))
im = plt.imshow(B_nv, cmap='RdBu', origin='lower')

plt.plot(x_line, y_line, 'k-', linewidth=2, label='Fit line')
plt.scatter([x1,x2], [y1,y2], c='yellow', s=60, label='Edges')

cbar = plt.colorbar(im)
cbar.set_label("B$_{NV}$ (mT)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.title("Domain wall selection", fontsize=17)
plt.xlabel("X (pixels)", fontsize=16)
plt.ylabel("Y (pixels)", fontsize=16)

plt.tick_params(labelsize=14)
plt.legend(fontsize=14)

plt.tight_layout()
plt.show()

# ============================================================
# 🔴 PLOT 2: PROFILE WITH Z ANNOTATION
# ============================================================

plt.figure(figsize=(5,3.5))

plt.plot(distance*1e6, profile_smooth*1e3, ".", ms=5, alpha=0.8, label="Data")

x_dense = np.linspace(distance.min(), distance.max(), 2000)
y_fit = stripe_model_full(x_dense, *params)

plt.plot(x_dense*1e6, y_fit*1e3, "r-", lw=1.5, label="Fit")

plt.xlabel("Distance (µm)", fontsize=16)
plt.ylabel("B$_{NV}$ (mT)", fontsize=16)
plt.title("Stripe model fit", fontsize=17)

# 🔥 ADD THIS PART (z annotation)
z_um = d_fit * 1e6
z_err_um = d_err * 1e6

plt.text(
    0.05, 0.95,
    f"d = {z_um:.2f} ± {z_err_um:.2f} µm",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tick_params(direction='in', top=True, right=True, labelsize=14)
plt.legend(fontsize=12, loc='lower right')

plt.tight_layout()
plt.show()