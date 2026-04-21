# ============================================================
# sensitivity.py - ODMR Sensitivity Measurement
# ============================================================
# Step-by-step execution with interactive breakpoints.
# Press ENTER to run each part, 'skip' to skip, 'q' to quit.
# ============================================================

# ==============================
# IMPORTS
# ==============================
import pylablib as pll
pll.par["devices/dlls/andor_sdk3"] = "sCMOS"
from pylablib.devices import Andor

import sys
import usb.core
import usb.util
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import pickle
import os
import time
import cv2
from datetime import datetime
from lmfit import Model
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import detrend, savgol_filter

time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# ==============================
# PARAMETERS
# ==============================
start_frequency = 2.835e9    # Hz
end_frequency = 2.905e9      # Hz
frequency_step = 0.0001e9   # Hz

num_frequency_points = int((end_frequency - start_frequency) / frequency_step) + 1
flist = np.arange(start_frequency, end_frequency + frequency_step, frequency_step)

save_directory = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\test"
base_dir = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\test"
data_filename = "contrast_data_3d.npy"
roi_i, roi_j, roi_span = 225, 256, 220

# Set this to skip directly to a specific part (e.g., 5 to start from Part 5)
START_FROM_PART = 1

# ==============================
# FUNCTION DEFINITIONS
# ==============================
def breakpoint_handler(part_number, part_name):
    """Interactive breakpoint. Returns True to run, False to skip. 'q' exits."""
    if part_number < START_FROM_PART:
        print(f"  >> Auto-skipping Part {part_number}: {part_name}")
        return False
    print("\n" + "=" * 60)
    print(f"  PART {part_number}: {part_name}")
    print("=" * 60)
    user_input = input("  ENTER=run | u=skip | q=quit: ").strip().lower()
    if user_input == 'q':
        print("\nExiting script.")
        sys.exit(0)
    if user_input == 'u':
        print(f"  >> Skipping Part {part_number}.\n")
        return False
    return True


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


def camera_readout(exposure_time, nframes):
    """Read and average multiple camera frames."""
    frames = []
    for _ in range(nframes):
        time.sleep(exposure_time * 1.1)
        frame = cam.read_newest_image()
        if frame is not None:
            frames.append(frame)
    if len(frames) == 0:
        return None
    return np.mean(frames, axis=0)


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
    """Generic plot for data and fit. Returns the figure object."""
    fig = plt.figure(figsize=(6, 4))
    plt.plot(frequencies, contrast, '.', label=label_data)
    plt.plot(frequencies, best_fit, 'r-', label=label_fit)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig


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
    ax.set_xlabel("X (um)" if x_dim else "X (px)")
    ax.set_ylabel("Y (um)" if y_dim else "Y (px)")
    return im


# ============================================================
# PART 1: Hardware Setup (Camera + Signal Generator)
# ============================================================
if breakpoint_handler(1, "Hardware Setup (Camera + Signal Generator)"):
    cam = Andor.AndorSDK3Camera()
    cam.get_detector_size()

    rm = pyvisa.ResourceManager()
    rm.list_resources()

    srs = rm.open_resource('TCPIP0::10.106.9.106::inst0::INSTR')
    print(srs.query("*IDN?"))

    srs.write("ENBR" + str(1))   # RF on (1=on, 0=off)
    srs.write("AMPR" + str(0))
    srs.query("ENBR?")
    srs.query("AMPR?")
    print("Hardware initialized.")


# ============================================================
# PART 2: Live Camera View
# ============================================================
if breakpoint_handler(2, "Live Camera View"):
    cam.set_exposure(0.1)
    cam.set_roi(1000, 1500, 1000, 1500)

    cv2.namedWindow("Live Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Camera View", 640, 480)

    cam.start_acquisition()
    print("Live view running. Press 'q' or close the window.")

    try:
        while True:
            if cv2.getWindowProperty("Live Camera View", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break

            frame = cam.grab()[0]
            frame_disp = cv2.normalize(
                frame.astype(np.float32),
                None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imshow("Live Camera View", frame_disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested via keyboard.")
                break

    except Exception as e:
        print("Live view error:", e)

    finally:
        cam.stop_acquisition()
        cv2.destroyAllWindows()
        print("Camera stopped safely.")


# ============================================================
# PART 3: Image Capture & Preview
# ============================================================
if breakpoint_handler(3, "Image Capture & Preview"):
    cam.set_exposure(0.1)
    cam.set_roi(1000, 1500, 1000, 1500)

    images = np.array(cam.grab(2))
    x = images.mean(axis=0)
    pil_img = Image.fromarray(x)
    print(f"Max intensity: {np.max(x)}")
    print(f"Image shape: {x.shape}")
    print(f"Corner mean: {images[:, :1, :1].mean()}")

    plot = plt.pcolormesh(x, cmap='gray')
    cb = plt.colorbar(plot, orientation='vertical')
    cb.set_label('Intensity')
    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    plt.show()

    # Show ROI subset
    span = 220
    i_pos, j_pos = 225, 256
    subm = x[max(0, i_pos - span):i_pos + span, max(0, j_pos - span):j_pos + span]
    print(f"ROI mean: {np.mean(subm)}")

    plot = plt.pcolormesh(subm, cmap='viridis')
    cb = plt.colorbar(plot, orientation='vertical')
    cb.set_label('Intensity (a.u.)')
    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    plt.show()


# ============================================================
# PART 4: ODMR Frequency Sweep Acquisition
# ============================================================
if breakpoint_handler(4, "ODMR Frequency Sweep Acquisition"):
    contrast_data_3d = np.zeros((len(flist), 2 * roi_span, 2 * roi_span))
    frequencies = []
    average_pixel_contrast_acq = []

    # Real-time plot
    plt.ion()
    fig_realtime, ax_realtime = plt.subplots(figsize=(10, 6))
    fig_realtime.suptitle('ODMR Spectrum - Real-time Acquisition', fontsize=12, fontweight='bold')
    ax_realtime.set_xlabel('Frequency (GHz)', fontsize=11)
    ax_realtime.set_ylabel('Average Contrast', fontsize=11)
    ax_realtime.set_ylim([0.88, 1.02])
    ax_realtime.grid(True, alpha=0.3)
    line_plot, = ax_realtime.plot([], [], 'b.-', linewidth=2, markersize=6, label='ODMR Data')
    ax_realtime.legend(loc='upper right')
    plt.show(block=False)

    total_acquiring_time = 0
    cam.set_exposure(0.1)
    cam.set_roi(1000, 1500, 1000, 1500)
    cam.setup_acquisition(mode='sequence')

    print(f"\nStarting acquisition sweep over {len(flist)} frequency points...")
    print("Press 'q' in the plot window to stop acquisition early.\n")

    stop_acquisition_flag = [False]

    def on_key_event(event):
        if event.key == 'q':
            stop_acquisition_flag[0] = True
            print("\nStopping acquisition (q pressed)...")

    fig_realtime.canvas.mpl_connect('key_press_event', on_key_event)

    cam.start_acquisition()
    for idx, n in enumerate(flist):
        start_time_acq = time.time()

        if stop_acquisition_flag[0]:
            break

        progress = (idx + 1) / len(flist) * 100
        print(f"[{progress:5.1f}%] Freq: {n/1e9:.4f} GHz - Acquiring...", end='\r')

        # Set MW frequency ON
        srs.write("freq" + str(n))
        on_image = camera_readout(0.1, 5)
        on_image_roi = on_image[roi_i - roi_span:roi_i + roi_span,
                                roi_j - roi_span:roi_j + roi_span]

        # Set MW frequency OFF (detuned)
        srs.write("freq" + str(3.0e9))
        off_image = camera_readout(0.1, 5)
        off_image_roi = off_image[roi_i - roi_span:roi_i + roi_span,
                                  roi_j - roi_span:roi_j + roi_span]

        # Compute contrast map
        contrast_map = on_image_roi / off_image_roi
        contrast_map[np.isnan(contrast_map)] = 0
        contrast_data_3d[idx] = contrast_map

        # Average contrast
        avg_contrast = np.mean(contrast_map)
        frequencies.append(n / 1e9)
        average_pixel_contrast_acq.append(avg_contrast)

        # Update real-time plot
        line_plot.set_data(frequencies, average_pixel_contrast_acq)
        if len(frequencies) > 0:
            ax_realtime.set_xlim([start_frequency / 1e9 - 0.005, end_frequency / 1e9 + 0.005])
        plt.pause(0.01)
        fig_realtime.canvas.draw_idle()

        end_time_acq = time.time()
        total_acquiring_time += (end_time_acq - start_time_acq)

    print("\n" + "=" * 60)
    print("Acquisition complete!")
    print("=" * 60)

    cam.stop_acquisition()

    # Trim if stopped early
    if len(frequencies) < len(flist):
        contrast_data_3d = contrast_data_3d[:len(frequencies)]
        print(f"\nEarly stop. Data trimmed to {len(frequencies)} frequency points.\n")

    print(f"Total acquiring time: {total_acquiring_time / 60:.1f} min.\n")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, "contrast_data_3d_2.npy")
    np.save(save_path, contrast_data_3d)
    print(f"Data saved to: {save_path}\n")

    plt.ioff()

    # Final analysis figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(frequencies, average_pixel_contrast_acq, 'b.-', linewidth=2, markersize=8, label='Acquired Data')
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('Average Contrast', fontsize=11)
    ax1.set_title('Final ODMR Spectrum', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.88, 1.02])
    ax1.legend(fontsize=10)

    ax2.imshow(contrast_data_3d.mean(axis=0), cmap='viridis')
    ax2.set_title('Average Contrast Map (ROI)')
    ax2.set_xlabel('X [pixel]')
    ax2.set_ylabel('Y [pixel]')
    plt.colorbar(ax2.images[0], ax=ax2, label='Contrast')
    plt.tight_layout()
    plt.show()


# ============================================================
# PART 5: Load Data & Average Contrast Analysis
# ============================================================
if breakpoint_handler(5, "Load Data & Average Contrast Analysis"):
    data_path = os.path.join(base_dir, data_filename)
    print("Loading data...")
    averaged_contrast_data_3d = np.load(data_path)
    print(f"Data shape: {averaged_contrast_data_3d.shape}")

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


# ============================================================
# PART 6: 10-Peak Lorentzian Fit
# ============================================================
if breakpoint_handler(6, "10-Peak Lorentzian Fit"):
    print("\n===== PART 6: 10-Peak Lorentzian Fit =====")

    bin_size = 1
    binned_frequencies, binned_average_contrast = perform_binning(
        flist, average_pixel_contrast, bin_size
    )

    save_binned_data(binned_frequencies, binned_average_contrast, base_dir,
                     "binned_average_contrast_data.npy")

    lmodel = Model(lorentzian_10peak)
    params = lmodel.make_params(
        off=1, x0=2.845e9, FWHM=1e5, a=1e+5, a2=1e+8, a3=1e+5,
        hfs=2.16e6, x0b=2.892e9
    )

    out = lmodel.fit(binned_average_contrast, params, x=binned_frequencies)

    # Extract parameters
    FWHM_Hz = abs(out.params['FWHM'].value)
    FWHM_MHz = FWHM_Hz * 1e-6
    FWHM_err = out.params['FWHM'].stderr
    FWHM_err_MHz = FWHM_err * 1e-6 if FWHM_err else None

    off_val = out.params['off'].value
    baseline = np.max(out.best_fit)
    dip_min = np.min(out.best_fit)
    contrast_percent = abs((baseline - dip_min) / baseline) * 100

    print(f"\n===== Fit Results (10-Peak) =====")
    if FWHM_err_MHz:
        print(f"FWHM: {FWHM_MHz:.3f} +/- {FWHM_err_MHz:.3f} MHz")
    else:
        print(f"FWHM: {FWHM_MHz:.3f} MHz")
    print(f"Contrast: {contrast_percent:.3f} %")

    # Plot
    fig_10peak = plot_fit(binned_frequencies, binned_average_contrast, out.best_fit,
                          'Fitting for Averaged Data (10-Peak)', label_fit='LMFit 10-Peak')
    plt.show()

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


# ============================================================
# PART 7: Image Binning & Processing
# ============================================================
if breakpoint_handler(7, "Image Binning & Processing"):
    print("\n===== PART 7: Image Binning & Processing =====")

    binning_factor = 1
    H, W = averaged_contrast_data_3d.shape[1], averaged_contrast_data_3d.shape[2]
    new_height = H // binning_factor
    new_width = W // binning_factor

    binned_images = np.empty((flist.size, new_height, new_width))
    for i in range(flist.size):
        img = averaged_contrast_data_3d[i, :new_height * binning_factor, :new_width * binning_factor]
        binned_images[i] = img.reshape(new_height, binning_factor, new_width, binning_factor).mean(axis=(1, 3))

    plt.imshow(binned_images[50], cmap='viridis')
    plt.colorbar()
    plt.title("Binned Image at Index 50")
    plt.show()

    print(f"Binned images shape: {binned_images.shape}")


# ============================================================
# PART 8: Single Pixel 2-Peak Lorentzian Fit
# ============================================================
if breakpoint_handler(8, "Single Pixel 2-Peak Lorentzian Fit"):
    print("\n===== PART 8: Single Pixel 2-Peak Lorentzian Fit =====")

    imgData = binned_images   # (freq, x, y)
    avg_odmr = np.mean(averaged_contrast_data_3d, axis=(1, 2))

    # Reshape for pixel access
    check2 = np.transpose(imgData)
    reshaped = check2.reshape(-1, check2.shape[-1])

    pixel_idx = 200
    pixel_val = reshaped[pixel_idx]
    normalized_pixel_val = pixel_val / max(pixel_val)

    plt.plot(flist, normalized_pixel_val, "-")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fluorescence [a.u.]')
    plt.title(f"Pixel {pixel_idx} ODMR")
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
    binned_freq_pix, binned_contrast_pix = perform_binning(
        smoothed_frequencies, smoothed_contrast, bin_size
    )

    lmodel_2peak = Model(lorentzian_2peak)
    params_2peak = lmodel_2peak.make_params(
        off=1, x0=2.845e9, FWHM=0.01e7, a=-1e+8, a2=-1e+8, x0b=2.892e9
    )
    params_2peak['x0'].set(min=2.82e9, max=2.87e9)
    params_2peak['x0b'].set(min=2.87e9, max=2.92e9)

    out_2peak = lmodel_2peak.fit(binned_contrast_pix, params_2peak, x=binned_freq_pix)

    # Save plot BEFORE show
    fig_pix = plot_fit(smoothed_frequencies, smoothed_contrast, out_2peak.best_fit,
                       'Single Pixel Fit (2-Peak)', label_fit='2-Peak Lorentzian')
    save_path_pixel = os.path.join(base_dir, "smoothed_perpixel_lorentzian_fit.png")
    fig_pix.savefig(save_path_pixel)
    plt.show()
    print(f"Single pixel fit plot saved to: {save_path_pixel}")

    # ---- Extract fitted parameters (from 2-peak fit) ----
    best_vals = out_2peak.best_values

    off = best_vals['off']
    x0 = best_vals['x0']
    x0b = best_vals['x0b']
    FWHM = abs(best_vals['FWHM'])
    a = best_vals['a']
    a2 = best_vals['a2']

    print("Resonance 1 (Hz):", x0)
    print("Resonance 2 (Hz):", x0b)
    print("Linewidth FWHM (Hz):", FWHM)

    # ---- Compute fitted curve ----
    fit_curve = lorentzian_2peak(binned_freq_pix, **best_vals)

    # ---- Contrast estimation ----
    dip1 = lorentzian_2peak(x0, **best_vals)
    dip2 = lorentzian_2peak(x0b, **best_vals)

    C1 = off - dip1
    C2 = off - dip2

    print("Contrast dip 1:", C1)
    print("Contrast dip 2:", C2)


# ============================================================
# PART 9: Sensitivity Slope Calculation
# ============================================================
if breakpoint_handler(9, "Sensitivity Slope Calculation"):
    print("\n===== PART 9: Sensitivity Slope Calculation =====")

    # Define window around each dip (e.g., +/- 5 MHz)
    window = 5e6

    # Indices near each resonance
    mask1 = np.abs(binned_freq_pix - x0) < window
    mask2 = np.abs(binned_freq_pix - x0b) < window

    # Slopes
    dC_df = np.gradient(fit_curve, binned_freq_pix)

    # Max slope near each dip
    idx1 = np.argmax(np.abs(dC_df[mask1]))
    idx2 = np.argmax(np.abs(dC_df[mask2]))

    max_slope1 = dC_df[mask1][idx1]
    max_slope2 = dC_df[mask2][idx2]

    f_opt1 = binned_freq_pix[mask1][idx1]
    f_opt2 = binned_freq_pix[mask2][idx2]

    print("Max slope dip 1:", abs(max_slope1), "at", f_opt1)
    print("Max slope dip 2:", abs(max_slope2), "at", f_opt2)

    # ---- Smoothed slope calculation ----
    f_axis = binned_freq_pix
    smooth_curve = savgol_filter(fit_curve, window_length=11, polyorder=3)
    dC_df_smooth = np.gradient(smooth_curve, f_axis)

    window_Hz = 5e6
    mask1s = np.abs(f_axis - x0) < window_Hz
    mask2s = np.abs(f_axis - x0b) < window_Hz

    idx1_local = np.argmax(np.abs(dC_df_smooth[mask1s]))
    idx2_local = np.argmax(np.abs(dC_df_smooth[mask2s]))

    idx1_global = np.where(mask1s)[0][idx1_local]
    idx2_global = np.where(mask2s)[0][idx2_local]

    avg_window = 3

    def local_avg(idx):
        valid = slice(max(0, idx - avg_window), min(len(dC_df_smooth), idx + avg_window + 1))
        return np.mean(np.abs(dC_df_smooth[valid]))

    max_slope1 = local_avg(idx1_global)
    max_slope2 = local_avg(idx2_global)

    f_opt1 = f_axis[idx1_global]
    f_opt2 = f_axis[idx2_global]

    # Choose the stronger slope
    if max_slope1 > max_slope2:
        max_slope = max_slope1
        f_opt = f_opt1
        chosen_dip = 1
    else:
        max_slope = max_slope2
        f_opt = f_opt2
        chosen_dip = 2

    print("Chosen dip:", chosen_dip)
    print("Maximum slope dC/df:", max_slope)
    print("Optimal frequency (Hz):", f_opt)

    gamma_NV = 28.024e9   # Hz/T
    theta = np.deg2rad(54.75)

    dC_dB_NV = max_slope * gamma_NV
    dC_dBz = dC_dB_NV * np.cos(theta)

    print("Magnetic slope dC/dB_NV:", abs(dC_dB_NV))
    print("Magnetic slope dC/dBz:", abs(dC_dBz))


# ============================================================
# PART 10: Intensity Time Trace Acquisition
# ============================================================
if breakpoint_handler(10, "Intensity Time Trace Acquisition"):
    print("\n===== PART 10: Intensity Time Trace Acquisition =====")

    # Override optimal frequency if needed (uncomment next line):
    #f_opt = 2160000000.0  # for insensitive data

    srs.write("freq" + str(f_opt))
    print(f"Microwave set to optimal slope frequency: {f_opt:.2f} Hz")

    # Settings
    N_frames = 20000
    exposure = 0.01
    n_avg = 1

    cam.set_exposure(exposure)

    # --- ROI Selection ---
    # Option A: Large ROI (default)
    #cam.set_roi(1000, 1500, 1000, 1500)

    # Option B: Single pixel ROI (uncomment below and comment Option A)
    x_center = 1250
    y_center = 1250
    cam.set_roi(x_center, x_center + 1, y_center, y_center + 1)

    cam.setup_acquisition(mode='sequence')
    cam.start_acquisition()

    intensity_data = []
    time_data = []

    # Real-time matplotlib plot
    plt.ion()
    fig_trace, ax_trace = plt.subplots(figsize=(10, 4))
    line_trace, = ax_trace.plot([], [], 'b-', linewidth=0.5)
    ax_trace.set_xlabel('Time (s)')
    ax_trace.set_ylabel('Counts')
    ax_trace.set_title('Intensity vs Time (Camera)')
    ax_trace.grid(True, alpha=0.3)
    plt.show(block=False)

    start_time_trace = time.time()
    update_every = 50

    for i in range(N_frames):
        image = camera_readout(exposure, n_avg)

        if image is None:
            continue

        avg_intensity = np.mean(image)
        t_now = time.time() - start_time_trace

        intensity_data.append(avg_intensity)
        time_data.append(t_now)

        # Real-time update
        if i % update_every == 0:
            line_trace.set_data(time_data, intensity_data)
            ax_trace.relim()
            ax_trace.autoscale_view()
            plt.pause(0.01)

    cam.stop_acquisition()
    plt.ioff()
    print("Acquisition complete.")

    # Stats
    time_array = np.arange(len(intensity_data)) * exposure
    fps_real = 1 / exposure
    print(f"Real sampling rate: {fps_real:.2f} Hz")

    intensity_data = np.array(intensity_data)
    print(f"Mean intensity: {np.mean(intensity_data):.2f}")
    print(f"Std: {np.std(intensity_data):.2f}")

    # Save counts vs time
    data_to_save = np.column_stack((time_data, intensity_data))
    save_path_cvt = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\sensitivity\Elec_CvT_pix1.txt"
    np.savetxt(save_path_cvt, data_to_save, header="Time(s)\tCounts", delimiter="\t")
    print(f"Data saved to: {save_path_cvt}")


# ============================================================
# PART 11: Magnetic Field Conversion & Plot
# ============================================================
if breakpoint_handler(11, "Magnetic Field Conversion & Plot"):
    print("\n===== PART 11: Magnetic Field Conversion & Plot =====")

    file_path = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\sensitivity\Elec_CvT_pix1.txt"
    data = np.loadtxt(file_path)

    time_stamps = data[:, 0]
    time_trace = data[:, 1]

    # Remove DC + linear drift
    detrended_signal = detrend(time_trace, type='linear')

    # Normalization
    mean_signal = np.mean(time_trace)
    delta_C = detrended_signal / mean_signal   # delta_C / C

    # Convert to magnetic field
    if dC_dB_NV == 0:
        raise ValueError("dC_dB_NV is zero - cannot convert")

    delta_B = delta_C / dC_dB_NV   # Tesla
    delta_B_uT = delta_B

    # Remove residual offset
    delta_B_uT = delta_B_uT - np.mean(delta_B_uT)

    # Save magnetic field data
    data_to_save = np.column_stack((time_stamps, delta_B_uT))
    save_path_mvt = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\sensitivity\Elec_MvT_pix1.txt"
    np.savetxt(
        save_path_mvt,
        data_to_save,
        header="Time(s)\tMagnetic_Field(uT)",
        delimiter="\t"
    )
    print(f"Magnetic field data saved to: {save_path_mvt}")

    # Plot
    plt.figure(figsize=(8, 3))
    plt.plot(time_stamps, delta_B_uT, linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetic Field (uT)")
    plt.title("Magnetic Field vs Time (Single Pixel)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Stats
    print("Mean field (uT):", np.mean(delta_B_uT))
    print("Std field (uT):", np.std(delta_B_uT))


print("\n" + "=" * 60)
print("  Script finished.")
print("=" * 60)
