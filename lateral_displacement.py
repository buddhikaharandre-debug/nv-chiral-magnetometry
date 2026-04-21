# ============================================================
# B-field profile with centered axis (dip at x = 0)
# ============================================================

import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os

# ---------------------- Load data ----------------------
base_dir = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\1.6"
B_field = np.load(os.path.join(base_dir, "B_NV_smoothed.npy"))

# ---------------------- Spatial scale ----------------------

pixel_size = 6.5
magnification = 20
sample_pixel_size = pixel_size / magnification

ny, nx = B_field.shape
x_extent = nx * sample_pixel_size
y_extent = ny * sample_pixel_size

# ---------------------- DISPLAY IMAGE ----------------------

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(
    B_field,
    cmap="afmhot",
    extent=[0, x_extent, 0, y_extent],
    origin="lower"
)
plt.colorbar(im, ax=ax, label=r"B$_{NV}$ (mT)")

ax.set_title("Select TWO points for B-field profile")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

pts = plt.ginput(2)
plt.close(fig)

(x0, y0), (x1, y1) = pts

# ---------------------- Convert to pixels ----------------------

p0x = x0 / sample_pixel_size
p0y = y0 / sample_pixel_size
p1x = x1 / sample_pixel_size
p1y = y1 / sample_pixel_size

# ---------------------- Extract profile ----------------------

n = 80

x_pix = np.linspace(p0x, p1x, n)
y_pix = np.linspace(p0y, p1y, n)

profile_B = map_coordinates(B_field, [y_pix, x_pix], order=1)

# ---------------------- Distance axis ----------------------

distance_um = np.linspace(
    0,
    np.sqrt((x1 - x0)**2 + (y1 - y0)**2),
    n
)

# ---------------------- SMOOTHING ----------------------

window = int(n * 0.05)
if window % 2 == 0:
    window += 1

profile_smooth = savgol_filter(profile_B, window_length=window, polyorder=3)

# ---------------------- GAUSSIAN FIT ----------------------

def gaussian(x, A, x0, sigma, offset):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

A_guess = np.nanmin(profile_smooth) - np.nanmax(profile_smooth)
x0_guess = distance_um[np.nanargmin(profile_smooth)]
sigma_guess = (distance_um[-1] - distance_um[0]) / 10
offset_guess = np.nanmean(profile_smooth)

p0 = [A_guess, x0_guess, sigma_guess, offset_guess]

mask = ~np.isnan(profile_smooth)

try:
    popt, _ = curve_fit(
        gaussian,
        distance_um[mask],
        profile_smooth[mask],
        p0=p0
    )
    fit_success = True
except Exception:
    fit_success = False

# ---------------------- CENTER THE AXIS ----------------------

if fit_success:
    x0_fit = popt[1]

    # shift axis so dip center = 0
    distance_centered = distance_um - x0_fit
else:
    # fallback: use minimum point
    x0_min = distance_um[np.nanargmin(profile_smooth)]
    distance_centered = distance_um - x0_min

# ---------------------- Extract width ----------------------

if fit_success:
    sigma_fit = abs(popt[2])
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
    print(f"FWHM = {FWHM:.3f} µm")
else:
    print("Fit failed")

# ---------------------- PLOT ----------------------

plt.figure(figsize=(5, 3.5))

plt.plot(distance_centered, profile_B, ".", ms=3, alpha=0.5, label="Raw")

if fit_success:
    x_fit = np.linspace(distance_centered.min(), distance_centered.max(), 1000)
    y_fit = gaussian(x_fit + x0_fit, *popt)  # shift back for correct fit

    plt.plot(x_fit, y_fit, "k--", lw=2, label="Gaussian fit")
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)

# ---------------------- Axis limits ----------------------

plt.xlim(-2, 2)

# ---------------------- Labels ----------------------

plt.xlabel("Lateral displacement (µm)")
plt.ylabel(r"B$_{NV}$ (mT)")
plt.title("Magnetic field profile")

# ---------------------- CLEAN STYLE ----------------------

plt.tick_params(
    axis='both',
    direction='in',
    top=True,
    right=True,
    length=5,
    width=1.2
)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.subplots_adjust(left=0.15, bottom=0.15)

#plt.legend()
plt.show()

# ---------------------- SAVE ----------------------

np.save(
    os.path.join(base_dir, "B_line_profile_centered.npy"),
    np.vstack([distance_centered, profile_B, profile_smooth]).T
)