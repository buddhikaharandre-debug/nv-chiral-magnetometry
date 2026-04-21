import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import warnings
from datetime import datetime
from lmfit import Model
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

# Suppress OptimizeWarning from curve_fit covariance estimation
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize')

# ==============================
# PART SELECTION - CHOOSE WHICH PARTS TO RUN
# ==============================
# Set to True to run a part, False to skip
RUN_PART_1 = True   # Average Contrast vs Frequency
RUN_PART_2 = True   # 10-Peak Lorentzian Fit
RUN_PART_3 = True   # Image Binning & Processing
RUN_PART_4 = True   # Single Pixel Analysis (2-Peak)
RUN_PART_5 = True   # Multi-Pixel Fitting (SLOW - takes time)
RUN_PART_6 = True   # FWHM & Contrast Maps
RUN_PART_7 = True   # Magnetic Field Maps

# Set INTERACTIVE_MODE = True to pause between each part and choose to continue
INTERACTIVE_MODE = True

# ==============================
# SETUP & CONSTANTS
# ==============================
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Parameters
start_frequency = 2.835e9
end_frequency = 2.905e9
frequency_step = 0.0001e9

# Generate frequency list
flist = np.linspace(start_frequency, end_frequency, 701)

# Base directory
base_dir = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\New"
data_filename = "contrast_test1.npy"
data_path = os.path.join(base_dir, data_filename)

# ==============================
# LORENTZIAN FUNCTIONS
# ==============================
def lorentzian_10peak(x, off, x0, FWHM, a, a2, a3, hfs, x0b):
    """10-peak Lorentzian (hyperfine splitting) for two centers."""
    lorentz1 = a / ((x - x0 - 2 * hfs)**2 + (0.5 * FWHM)**2)
    lorentz2 = a2 / ((x - x0 - hfs)**2 + (0.5 * FWHM)**2)
    lorentz3 = a3 / ((x - x0)**2 + (0.5 * FWHM)**2)
    lorentz4 = a2 / ((x - x0 + hfs)**2 + (0.5 * FWHM)**2)
    lorentz5 = a / ((x - x0 + 2 * hfs)**2 + (0.5 * FWHM)**2)
    lorentz6 = a / ((x - x0b - 2 * hfs)**2 + (0.5 * FWHM)**2)
    lorentz7 = a2 / ((x - x0b - hfs)**2 + (0.5 * FWHM)**2)
    lorentz8 = a3 / ((x - x0b)**2 + (0.5 * FWHM)**2)
    lorentz9 = a2 / ((x - x0b + hfs)**2 + (0.5 * FWHM)**2)
    lorentz10 = a / ((x - x0b + 2 * hfs)**2 + (0.5 * FWHM)**2)
    return off + (lorentz1 + lorentz2 + lorentz3 + lorentz4 + lorentz5 + 
                  lorentz6 + lorentz7 + lorentz8 + lorentz9 + lorentz10)


def lorentzian_2peak(x, off, x0, x0b, FWHM, a, a2):
    """2-peak Lorentzian (simplified double Lorentzian)."""
    lorentz1 = a / ((x - x0)**2 + (0.5 * FWHM)**2)
    lorentz2 = a2 / ((x - x0b)**2 + (0.5 * FWHM)**2)
    return off + lorentz1 + lorentz2


# ==============================
# UTILITY FUNCTIONS
# ==============================
def save_binned_data(frequencies, contrast, base_dir, filename):
    """Save binned frequency and contrast data."""
    binned_data = np.column_stack((frequencies, contrast))
    path = os.path.join(base_dir, filename)
    np.save(path, binned_data)
    return path


def perform_binning(frequencies, contrast_data, bin_size=1):
    """Perform binning on frequency and contrast data."""
    num_bins = len(frequencies) // bin_size
    binned_freq = np.mean(
        frequencies[:num_bins * bin_size].reshape((num_bins, bin_size)), 
        axis=1
    )
    binned_contrast = np.mean(
        contrast_data[:num_bins * bin_size].reshape((num_bins, bin_size)), 
        axis=1
    )
    return binned_freq, binned_contrast


def plot_fit(frequencies, contrast, best_fit, title, xlabel='Frequency (Hz)', 
             ylabel='Contrast', label_data='Data', label_fit='Fit'):
    """Generic plot for data and fit."""
    plt.figure(figsize=(6, 4))
    plt.plot(frequencies, contrast, '.', label=label_data)
    plt.plot(frequencies, best_fit, 'r-', label=label_fit)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return plt


def fill_nan(arr):
    """Fill NaN values with median."""
    tmp = np.copy(arr)
    tmp[np.isnan(tmp)] = np.nanmedian(tmp)
    return tmp


def get_limits(data, vmin_manual, vmax_manual, color_mode='auto', p_low=5, p_high=98):
    """Get color limits based on mode."""
    if color_mode == "manual":
        return vmin_manual, vmax_manual
    else:
        return np.nanpercentile(data, p_low), np.nanpercentile(data, p_high)


def plot_map(ax, data, title, label, cmap, vmin, vmax, x_dim=None, y_dim=None):
    """Plot a map with colorbar."""
    extent = [0, x_dim, 0, y_dim] if x_dim and y_dim else None
    im = ax.imshow(data, cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=label)
    ax.set_title(title)
    ax.set_xlabel("X (µm)" if x_dim else "X (px)")
    ax.set_ylabel("Y (µm)" if y_dim else "Y (px)")
    return im


def breakpoint_handler(part_number, part_name):
    """Interactive breakpoint between parts."""
    if INTERACTIVE_MODE:
        print("\n" + "="*60)
        print(f"BREAKPOINT: About to run PART {part_number}: {part_name}")
        print("="*60)
        try:
            user_input = input("Press ENTER to continue, or press 'u' + ENTER to skip this part: ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting...")
            raise SystemExit(0)
        if user_input == 'u':
            return False
    return True


# ==============================
# LOAD DATA
# ==============================
print("Loading data...")
if not os.path.isfile(data_path):
    # Search subdirectories for the data file
    found = None
    for entry in os.listdir(base_dir):
        sub = os.path.join(base_dir, entry)
        if os.path.isdir(sub):
            candidate = os.path.join(sub, data_filename)
            if os.path.isfile(candidate):
                found = candidate
                break
    if found:
        data_path = found
        print(f"  File not in base_dir, found at: {data_path}")
    else:
        raise FileNotFoundError(f"Cannot find '{data_filename}' in {base_dir} or its subdirectories.")
averaged_contrast_data_3d = np.load(data_path)
print(f"Data shape: {averaged_contrast_data_3d.shape}")

# ==============================
# PART 1: AVERAGE CONTRAST ANALYSIS
# ==============================
if not RUN_PART_1:
    print("\nSkipping PART 1...")
else:
    if not breakpoint_handler(1, "Average Contrast vs Frequency"):
        RUN_PART_1 = False
        print("\nSkipping PART 1...")
    else:
        print("\n===== PART 1: Average Contrast vs Frequency =====")

if RUN_PART_1:
    average_pixel_contrast = np.mean(averaged_contrast_data_3d, axis=(1, 2))

    plt.figure(figsize=(6, 4))
    plt.plot(flist, average_pixel_contrast, label='Average Pixel Contrast', marker='.')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Average Contrast')
    plt.title('Average Contrast vs Frequency')
    plt.grid()
    plt.show()

    # Display sample image
    desired1 = averaged_contrast_data_3d[10, :, :]
    plt.imshow(desired1, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(f"Sample Image at Index 10, Shape: {desired1.shape}")
    plt.show()

# ==============================
# PART 2: BINNING & LORENTZIAN FIT (10-PEAK)
# ==============================
print("\n===== PART 2: 10-Peak Lorentzian Fit =====")
bin_size = 1
num_bins = len(flist) // bin_size
binned_frequencies, binned_average_contrast = perform_binning(flist, average_pixel_contrast, bin_size)

save_binned_data(binned_frequencies, binned_average_contrast, base_dir, 
                "binned_average_contrast_data.npy")

lmodel = Model(lorentzian_10peak)
params = lmodel.make_params(
    off=1, x0=2.85e9, FWHM=1e5, a=1e+5, a2=1e+8, a3=1e+5, 
    hfs=2.16e6, x0b=2.89e9
)

out = lmodel.fit(binned_average_contrast, params, x=binned_frequencies)

# Extract parameters
FWHM_Hz = abs(out.params['FWHM'].value)
FWHM_MHz = FWHM_Hz * 1e-6
FWHM_err = out.params['FWHM'].stderr
FWHM_err_MHz = FWHM_err * 1e-6 if FWHM_err else None

off = out.params['off'].value
baseline = np.max(out.best_fit)
dip_min = np.min(out.best_fit)
contrast_percent = abs((baseline - dip_min) / baseline) * 100

print(f"\n===== Fit Results (10-Peak) =====")
if FWHM_err_MHz:
    print(f"FWHM: {FWHM_MHz:.3f} ± {FWHM_err_MHz:.3f} MHz")
else:
    print(f"FWHM: {FWHM_MHz:.3f} MHz")
print(f"Contrast: {contrast_percent:.3f} %")

# Plot
plot_fit(binned_frequencies, binned_average_contrast, out.best_fit,
         'Fitting for Averaged Data (10-Peak)', label_fit='LMFit 10-Peak').show()

# Extract x0 values
x0_value = out.params['x0'].value
x0b_value = out.params['x0b'].value
difference = abs(x0b_value - x0_value)
BNV_field = (difference * 1e-6) / (2 * 28.024)
Bz_field = BNV_field / np.cos(np.deg2rad(54.75))

print(f"\nx0 value: {x0_value:.6e} Hz")
print(f"x0b value: {x0b_value:.6e} Hz")
print(f"Difference (x0b - x0): {difference:.6e} Hz")
print(f"Magnetic field (B_NV): {BNV_field:.3f} mT")
print(f"Magnetic field Bz: {Bz_field} mT")

# ==============================
# PART 3: IMAGE BINNING & PROCESSING
# ==============================
if not RUN_PART_3:
    print("\nSkipping PART 3...")
else:
    if not breakpoint_handler(3, "Image Binning & Processing"):
        RUN_PART_3 = False
        print("\nSkipping PART 3...")
    else:
        print("\n===== PART 3: Image Binning & Processing =====")

if RUN_PART_3:
    binning_factor = 1
    H, W = averaged_contrast_data_3d.shape[1], averaged_contrast_data_3d.shape[2]
    new_height = H // binning_factor
    new_width = W // binning_factor

    binned_images = np.empty((flist.size, new_height, new_width))
    for i in range(flist.size):
        img = averaged_contrast_data_3d[i, :new_height*binning_factor, :new_width*binning_factor]
        binned_images[i] = img.reshape(new_height, binning_factor, new_width, binning_factor).mean(axis=(1, 3))

    plt.imshow(binned_images[50], cmap='viridis')
    plt.colorbar()
    plt.title("Binned Image at Index 50")
    plt.show()

    print(f"Binned images shape: {binned_images.shape}")

# ==============================
# PART 4: SINGLE PIXEL ANALYSIS (2-PEAK LORENTZIAN)
# ==============================
if not RUN_PART_4:
    print("\nSkipping PART 4...")
else:
    if not breakpoint_handler(4, "Single Pixel Analysis (2-Peak)"):
        RUN_PART_4 = False
        print("\nSkipping PART 4...")
    else:
        print("\n===== PART 4: Single Pixel Analysis (2-Peak) =====")

if RUN_PART_4:
    imgData = binned_images  # (freq, x, y)
    avg_odmr = np.mean(averaged_contrast_data_3d, axis=(1, 2))

    # Reshape for pixel access
    check2 = np.transpose(imgData)
    reshaped = check2.reshape(-1, check2.shape[-1])

    i = 200
    pixel_val = reshaped[i]
    normalized_pixel_val = pixel_val / max(pixel_val)

    plt.plot(flist, normalized_pixel_val, "-")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fluorescence [a.u.]')
    plt.title(f"Pixel {i} ODMR")
    plt.show()

    data_array = np.column_stack((flist, normalized_pixel_val))
    np.save(os.path.join(base_dir, "pixel_data.npy"), data_array)

    # Smooth pixel data
    xData = flist
    yData = normalized_pixel_val
    smoothed_data = lowess(yData, xData, frac=0.16)
    smoothed_frequencies = smoothed_data[:, 0]
    smoothed_contrast = smoothed_data[:, 1]

    bin_size = 1
    binned_freq_pix, binned_contrast_pix = perform_binning(smoothed_frequencies, smoothed_contrast, bin_size)

    lmodel_2peak = Model(lorentzian_2peak)
    params_2peak = lmodel_2peak.make_params(
        off=1, x0=2.85e9, FWHM=0.01e7, a=-1e+8, a2=-1e+8, x0b=2.89e9
    )
    params_2peak['x0'].set(min=2.82e9, max=2.87e9)
    params_2peak['x0b'].set(min=2.87e9, max=2.92e9)

    out_2peak = lmodel_2peak.fit(binned_contrast_pix, params_2peak, x=binned_freq_pix)

    plot_fit(smoothed_frequencies, smoothed_contrast, out_2peak.best_fit,
             'Single Pixel Fit (2-Peak)', label_fit='2-Peak Lorentzian').show()

    save_path_pixel = os.path.join(base_dir, "smoothed_perpixel_lorentzian_fit.png")
    plt.savefig(save_path_pixel)

# ==============================
# PART 5: MULTI-PIXEL FITTING
# ==============================
if not RUN_PART_5:
    print("\nSkipping PART 5 (Multi-Pixel Fitting)...")
    print("Will load previously saved data files if available...\n")
else:
    if not breakpoint_handler(5, "Multi-Pixel Fitting (this may take a while)"):
        print("\nSkipping PART 5 (Multi-Pixel Fitting)...")
        print("Will load previously saved data files if available...\n")
    else:
        print("\n===== PART 5: Multi-Pixel Fitting =====\n")

        # Storage arrays
        pixel_coordinates = []
        x0_values = []
        x0b_values = []
        x0b_minus_x0_values = []
        FWHM_values = []
        contrast_values = []

        t1 = time.time()

        # Loop over pixels
        total_pixels = imgData.shape[1] * imgData.shape[2]
        processed_count = 0
        stop_requested = False
        paused = False
        print("Press Ctrl+C to stop processing early.")
        print("Press Ctrl+C once to pause, again to resume, third time to stop.\n")
        ctrl_c_count = 0
        
        for pixel_i in range(imgData.shape[1]):
            if stop_requested:
                break
            for pixel_j in range(imgData.shape[2]):
                try:
                    processed_count += 1
                    if processed_count % 100 == 0:
                        elapsed = time.time() - t1
                        mins = int(elapsed) // 60
                        secs = int(elapsed) % 60
                        print(f"Processed {processed_count}/{total_pixels} pixels... Elapsed: {mins:02d}:{secs:02d}")
                except KeyboardInterrupt:
                    ctrl_c_count += 1
                    if ctrl_c_count == 1:
                        print(f"\nPaused at pixel {processed_count}/{total_pixels}. Press Ctrl+C to resume or again to stop.")
                        try:
                            while True:
                                time.sleep(0.1)
                        except KeyboardInterrupt:
                            ctrl_c_count += 1
                            print("Resumed.")
                    else:
                        stop_requested = True
                        print(f"\nStop requested after {processed_count} pixels.")
                        break
                
                # Extract pixel data
                pixel_contrast_values = imgData[:, pixel_i, pixel_j]

                # Smoothing
                smoothed_frequencies = flist
                smoothed_contrast = savgol_filter(
                    pixel_contrast_values,
                    polyorder=2,
                    window_length=60
                )

                # Binning
                bin_size = 2 
                num_bins = len(smoothed_frequencies) // bin_size

                binned_frequencies = np.mean(
                    smoothed_frequencies[:num_bins * bin_size].reshape((num_bins, bin_size)),
                    axis=1
                )

                binned_contrast_values = np.mean(
                    smoothed_contrast[:num_bins * bin_size].reshape((num_bins, bin_size)),
                    axis=1
                )

                # Initial parameters
                initial_params = [1, 2.85e9, 2.89e9, 0.01e7, -1e8, -1e9]

                # Fit
                try:
                    popt, pcov = curve_fit(
                        lorentzian_2peak,
                        binned_frequencies,
                        binned_contrast_values,
                        p0=initial_params,
                        maxfev=40000
                    )
                except:
                    continue

                # Extract parameters
                x0_value = popt[1]
                x0b_value = popt[2]
                FWHM_value = popt[3]

                # Splitting
                x0b_minus_x0_value = abs(x0b_value - x0_value)

                # Contrast (dip strength)
                contrast_value = abs(popt[4]) + abs(popt[5])

                # Store
                pixel_coordinates.append((pixel_i, pixel_j))
                x0_values.append(x0_value)
                x0b_values.append(x0b_value)
                x0b_minus_x0_values.append(x0b_minus_x0_value)
                FWHM_values.append(FWHM_value)
                contrast_values.append(contrast_value)

        # Convert to arrays
        pixel_coordinates = np.array(pixel_coordinates)
        x0_values = np.array(x0_values)
        x0b_values = np.array(x0b_values)
        x0b_minus_x0_values = np.array(x0b_minus_x0_values)
        FWHM_values = np.array(FWHM_values)
        contrast_values = np.array(contrast_values)

        # Create directory
        os.makedirs(base_dir, exist_ok=True)

        # SAVE FILE 1 (for B-field map)
        save_path_diff = os.path.join(base_dir, "pixel_x0_x0b_differences.npy")

        np.save(
            save_path_diff,
            np.column_stack((
                pixel_coordinates,
                x0_values,
                x0b_values,
                x0b_minus_x0_values
            ))
        )
        print(f"Saved: {save_path_diff}")

        # SAVE FILE 2 (for contrast & FWHM maps)
        save_path_full = os.path.join(base_dir, "pixel_fit_parameters.npy")

        np.save(
            save_path_full,
            np.column_stack((
                pixel_coordinates,
                x0_values,
                x0b_values,
                x0b_minus_x0_values,
                FWHM_values,
                contrast_values
            ))
        )
        print(f"Saved: {save_path_full}")

        # Safety check
        assert os.path.isfile(save_path_diff), "Splitting file not saved!"
        assert os.path.isfile(save_path_full), "Full parameter file not saved!"

        # Timing
        t2 = time.time()
        elapsed_total = t2 - t1
        mins_total = int(elapsed_total) // 60
        secs_total = int(elapsed_total) % 60
        print(f"Total fitted pixels: {len(pixel_coordinates)}")
        print(f"Total time: {mins_total:02d}:{secs_total:02d} ({elapsed_total/60:.2f} minutes)\n")

# ==============================
# PART 6: FWHM & CONTRAST & SENSITIVITY MAPS
# ==============================
if not RUN_PART_6:
    print("Skipping PART 6...\n")
else:
    if not breakpoint_handler(6, "FWHM & Contrast & Sensitivity Maps"):
        print("Skipping PART 6...\n")
    else:
        print("===== PART 6: FWHM & Contrast & Sensitivity Maps =====\n")
        
        # Close any previous figures to avoid clutter
        plt.close('all')
        
        # Check if imgData is available (required for computing contrast)
        if imgData is None:
            print("ERROR: imgData not available. Please run PART 3 first to process the images.\n")
            RUN_PART_6 = False
        else:
            # Load saved data from Part 5
            save_path_full = os.path.join(base_dir, "pixel_fit_parameters.npy")
            if os.path.isfile(save_path_full):
                data = np.load(save_path_full)
                pixel_coordinates = data[:, :2].astype(int)
                x0_values = data[:, 2]
                x0b_values = data[:, 3]
                x0b_minus_x0_values = data[:, 4]
                FWHM_values = data[:, 5]
                contrast_values = data[:, 6]
                print(f"Loaded data from: {save_path_full}\n")
            else:
                print(f"ERROR: File not found: {save_path_full}")
                print("Please run PART 5 first to generate the data files.\n")
                RUN_PART_6 = False

            if RUN_PART_6:
                color_mode = "auto"
                p_low, p_high = 5, 98
                FWHM_vmin, FWHM_vmax = 3, 10
                contrast_vmin, contrast_vmax = 0.01, 0.1
                sens_vmin, sens_vmax = 1, 50

            # ==============================
            # Constants for Sensitivity Calculation
            # ==============================
            gamma_e = 28e9
            R = 1.85e8
            prefactor = 4 / (3 * np.sqrt(3))

            # Compute true contrast per pixel
            contrast_list = []
            for (i, j) in pixel_coordinates:
                spectrum = imgData[:, i, j]
                I_off = np.max(spectrum)
                I_on = np.min(spectrum)
                contrast = (I_off - I_on) / I_off if I_off > 0 else np.nan
                contrast_list.append(contrast)

            contrast_values_true = np.array(contrast_list)
            contrast_values_true[contrast_values_true < 1e-6] = 1e-6

            # ==============================
            # Calculate Sensitivity
            # ==============================
            sensitivity_T = prefactor * FWHM_values / (
                contrast_values_true * gamma_e * np.sqrt(R)
            )
            sensitivity_uT = sensitivity_T * 1e6

            FWHM_MHz = FWHM_values / 1e6

            # ==============================
            # Convert to grid
            # ==============================
            data_FWHM = np.array([(c[1], c[0], v) for c, v in zip(pixel_coordinates, FWHM_MHz)])
            data_contrast = np.array([(c[1], c[0], v) for c, v in zip(pixel_coordinates, contrast_values_true)])
            data_sens = np.array([(c[1], c[0], v) for c, v in zip(pixel_coordinates, sensitivity_uT)])

            x_min, x_max = int(data_FWHM[:, 0].min()), int(data_FWHM[:, 0].max())
            y_min, y_max = int(data_FWHM[:, 1].min()), int(data_FWHM[:, 1].max())

            xg, yg = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))

            FWHM_grid = griddata((data_FWHM[:, 0], data_FWHM[:, 1]), data_FWHM[:, 2], (xg, yg), method='cubic')
            contrast_grid = griddata((data_contrast[:, 0], data_contrast[:, 1]), data_contrast[:, 2], (xg, yg), method='cubic')
            sens_grid = griddata((data_sens[:, 0], data_sens[:, 1]), data_sens[:, 2], (xg, yg), method='cubic')

            # ==============================
            # Fill NaNs and smooth
            # ==============================
            FWHM_s = gaussian_filter(fill_nan(FWHM_grid), 1.5)
            contrast_s = gaussian_filter(fill_nan(contrast_grid), 1.5)
            sens_s = gaussian_filter(fill_nan(sens_grid), 1.5)

            # ==============================
            # Get color limits
            # ==============================
            FWHM_vmin, FWHM_vmax = get_limits(FWHM_s, FWHM_vmin, FWHM_vmax, color_mode, p_low, p_high)
            contrast_vmin, contrast_vmax = get_limits(contrast_s, contrast_vmin, contrast_vmax, color_mode, p_low, p_high)
            sens_vmin, sens_vmax = get_limits(sens_s, sens_vmin, sens_vmax, color_mode, p_low, p_high)

            # ==============================
            # Spatial scale (µm)
            # ==============================
            pixel_size = 6.5
            magnification = 20
            scale = pixel_size / magnification
            x_dim = FWHM_s.shape[1] * scale
            y_dim = FWHM_s.shape[0] * scale

            # ==============================
            # Plot 3 maps: FWHM, Contrast, Sensitivity
            # ==============================
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            plot_map(axs[0], FWHM_s, "FWHM Map", "FWHM (MHz)", "viridis", FWHM_vmin, FWHM_vmax, x_dim, y_dim)
            plot_map(axs[1], contrast_s, "Contrast Map", "Contrast (a.u.)", "inferno", contrast_vmin, contrast_vmax, x_dim, y_dim)
            plot_map(axs[2], sens_s, "Sensitivity Map", "Sensitivity (µT/√Hz)", "plasma", sens_vmin, sens_vmax, x_dim, y_dim)
            plt.tight_layout()
            plt.show()

            # ==============================
            # Mean values
            # ==============================
            mean_FWHM = np.nanmean(FWHM_values) / 1e6
            mean_contrast = np.nanmean(contrast_values_true)
            mean_sensitivity = np.nanmean(sensitivity_uT)

            print(f"\n===== MEAN VALUES =====")
            print(f"Mean FWHM: {mean_FWHM:.3f} MHz")
            print(f"Mean Contrast: {mean_contrast:.4f} ({mean_contrast*100:.2f} %)")
            print(f"Mean Sensitivity: {mean_sensitivity:.3f} µT/√Hz\n")
            print("✓ PART 6 Complete\n")

# ==============================
# PART 7: MAGNETIC FIELD MAPS
# ==============================
if not RUN_PART_7:
    print("Skipping PART 7...\n")
else:
    if not breakpoint_handler(7, "Magnetic Field Maps"):
        print("Skipping PART 7...\n")
    else:
        print("===== PART 7: Magnetic Field Maps =====\n")
        
        # Load saved data from Part 5
        save_path_diff = os.path.join(base_dir, "pixel_x0_x0b_differences_plot_data.npy")
        if os.path.isfile(save_path_diff):
            plot_data = np.load(save_path_diff)
            print(f"Loaded data from: {save_path_diff}\n")
        else:
            # Try to load from the raw file and reconstruct plot_data
            save_path_raw = os.path.join(base_dir, "pixel_x0_x0b_differences.npy")
            if os.path.isfile(save_path_raw):
                data = np.load(save_path_raw)
                pixel_coordinates = data[:, :2]
                x0b_minus_x0_values = data[:, 4]
                plot_data = np.column_stack((pixel_coordinates[:, 1], pixel_coordinates[:, 0], x0b_minus_x0_values))
                print(f"Loaded and reconstructed data from: {save_path_raw}\n")
            else:
                print(f"ERROR: Files not found")
                print("Please run PART 5 first to generate the data files.\n")
                RUN_PART_7 = False

        if RUN_PART_7:
            # Grid & interpolation
            x_min, x_max = int(plot_data[:, 0].min()), int(plot_data[:, 0].max())
            y_min, y_max = int(plot_data[:, 1].min()), int(plot_data[:, 1].max())

            x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

            frequency_grid = griddata(
                (plot_data[:, 0], plot_data[:, 1]),
                plot_data[:, 2] / 1e6,  # Convert to MHz
                (x_grid, y_grid),
                method='cubic',
                fill_value=np.nan
            )

            # Convert splitting → B-field (MHz to mT)
            B_values = frequency_grid / (2 * 28)

            valid_mask = ~np.isnan(B_values)
            p10, p90 = np.nanpercentile(B_values, (10, 95))
            print(f"Magnetic field range: {p10:.4f} - {p90:.4f} mT\n")

            # Clip & smooth
            B_contrast = np.full_like(B_values, np.nan)
            B_contrast[valid_mask] = np.clip(B_values[valid_mask], p10, p90)

            B_tmp = np.copy(B_contrast)
            B_tmp[~valid_mask] = np.nanmedian(B_contrast)
            B_smoothed = gaussian_filter(B_tmp, sigma=1)
            B_smoothed[~valid_mask] = np.nan

            # Spatial scale
            pixel_size = 6.5
            magnification = 20
            scale = pixel_size / magnification
            x_dim = B_values.shape[1] * scale
            y_dim = B_values.shape[0] * scale

            # Plot
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            plot_map(axs[0], B_contrast, "Magnetic Image (Contrast)", "B$_{NV}$ (mT)", "magma", p10, p90, x_dim, y_dim)
            plot_map(axs[1], B_smoothed, "Magnetic Image (Smoothed)", "B$_{NV}$ (mT)", "afmhot", p10, p90, x_dim, y_dim)
            plt.tight_layout()
            plt.show()

            # Save final results
            np.save(os.path.join(base_dir, "B_NV.npy"), B_values)
            np.save(os.path.join(base_dir, "B_NV_contrast.npy"), B_contrast)
            np.save(os.path.join(base_dir, "B_NV_smoothed.npy"), B_smoothed)
            print("Saved B_NV files\n")

            # ==============================
            # Normalized Magnetic Field Plot
            # ==============================
            # Calculate normalized B (in sigma units)
            B_mean = np.nanmean(B_smoothed)
            B_std = np.nanstd(B_smoothed)
            B_norm = (B_smoothed - B_mean) / B_std

            norm_lim = 3
            save_path_normalized = os.path.join(base_dir, "magnetic_image_normalized.png")

            plt.figure(figsize=(4.5, 3.5))
            plt.imshow(
                B_norm,
                cmap='afmhot',
                extent=[0, x_dim, 0, y_dim],
                origin='lower',
                vmin=-norm_lim,
                vmax=norm_lim
            )
            cbar = plt.colorbar()
            cbar.set_label("Normalized ΔB$_{NV}$ (σ units)")
            plt.xlabel("X (µm)")
            plt.ylabel("Y (µm)")
            plt.title("Normalized Magnetic Image")
            plt.tight_layout()
            plt.savefig(save_path_normalized, dpi=300)
            plt.show()
            print(f"Saved normalized plot: {save_path_normalized}\n")

            # Save normalized B_NV
            np.save(os.path.join(base_dir, "B_NV_normalized.npy"), B_norm)
            print("Saved B_NV_normalized.npy\n")
            
            print("✓ PART 7 Complete\n")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)
