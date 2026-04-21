import pylablib as pll
pll.par["devices/dlls/andor_sdk3"] = "sCMOS"
from pylablib.devices import Andor

#%matplotlib inline
#%matplotlib notebook

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
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

cam = Andor.AndorSDK3Camera()  
cam.get_detector_size()

rm = pyvisa.ResourceManager()
rm.list_resources()

srs = rm.open_resource('TCPIP0::10.106.9.106::inst0::INSTR')
srs.timeout = 2000        # 2s timeout (default 10s is wasteful)
srs.chunk_size = 102400    # larger TCP chunks for faster I/O
print(srs.query("*IDN?"))

srs.write("ENBR"+str(1))#rf on 1=0 for off 

srs.write("AMPR"+str(0)) 
srs.query("ENBR?")#check if output is on
srs.query("AMPR?")#check if output is on

# ============= CAMERA LIVE VIEW SECTION =============
# Already imported above

# ---------------- Camera setup ----------------
cam.set_exposure(0.1)
cam.set_roi(1000,1500,1000,1500) #(0,2560,0,1920)

cv2.namedWindow("Live Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Camera View", 640, 480)  # Reduced window size: width x height

# Start continuous acquisition
cam.start_acquisition()

print("Live view running. Press 'q' or close the window.")

try:
    while True:
        # Check if window still exists
        if cv2.getWindowProperty("Live Camera View", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user.")
            break

        # Get latest frame
        frame = cam.grab()[0]

        # Convert uint32 → float32
        frame_disp = cv2.normalize(
            frame.astype(np.float32),
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        cv2.imshow("Live Camera View", frame_disp)

        # Keyboard exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested via keyboard.")
            break

except Exception as e:
    print("Live view error:", e)

finally:
    # -------- ALWAYS CLEAN UP --------
    cam.stop_acquisition()
    cv2.destroyAllWindows()
    print("Camera stopped safely.")


#bin = 1.0
cam.set_exposure(0.1)  
#cam.set_roi(1100,1400,1050,1350,int(bin),int(bin))  #(0,2560,0,1920)
cam.set_roi(1000,1500,1000,1500) 

images = np.array(cam.grab(2)) 
x = (images.mean(axis=0))
pil_img = Image.fromarray(x)
np.max(x)

np.shape(images)
images[:,:1,:1].mean()
print(x.shape)

np.shape(images)

plot = plt.pcolormesh(x, cmap='gray')#, vmin=Vmin, vmax=Vmax)
cb= plt.colorbar(plot, orientation='vertical')
cb.set_label('Intensity')
plt.xlabel('X [pixel]')
plt.ylabel('Y [pixel]')
plt.show()

#cam.get_full_status()

#show correct subset of the image
span=220
i,j= 225,256
subm = (x[max(0,i-span):i+span,max(0,j-span):j+span])
print(np.mean(subm))
plot = plt.pcolormesh(subm, cmap='viridis')#, vmin=Vmin, vmax=Vmax)
cb= plt.colorbar(plot, orientation='vertical')
#plt.draw()
cb.set_label('Intensity (a.u.)')
plt.xlabel('X [pixel]')
plt.ylabel('Y [pixel]')
plt.show()

# Parameters
start_frequency = 2.86e9 #2.835e9 
end_frequency = 2.88e9  #2.905e9
frequency_step = 0.0001e9  

# Calculate the number of frequency points based on the given range and step size
num_frequency_points = int((end_frequency - start_frequency) / frequency_step) + 1

flist = np.arange(start_frequency, end_frequency + frequency_step, frequency_step)

def camera_readout(exposure_time, nframes):
    """Fast bulk readout: flush stale frames, sleep once, read all at once."""
    cam.read_multiple_images()          # discard stale frames from previous freq
    time.sleep(exposure_time * nframes)  # single sleep for all frames
    frames = cam.read_multiple_images()
    if frames is not None and len(frames) >= nframes:
        return np.asarray(frames[-nframes:]).mean(axis=0)
    # If not enough frames yet, wait a bit more and retry
    time.sleep(exposure_time * 2)
    frames2 = cam.read_multiple_images()
    all_f = (list(frames) if frames else []) + (list(frames2) if frames2 else [])
    if len(all_f) >= 1:
        return np.asarray(all_f[-min(nframes, len(all_f)):]).mean(axis=0)
    # Last resort fallback
    return cam.read_newest_image()

# ================ PLOTLY SETUP (OPTIONAL) ================
# If running in Jupyter, uncomment the lines below:
# from IPython.display import display
# import plotly.graph_objs as go

# ------------------- Configuration -------------------
save_directory = r"C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\test"
roi_i, roi_j, roi_span = 225,256,220
#bin = 1.0

#roi_i, roi_j, roi_span = int(150/bin),int(150/bin),int(100/bin)
contrast_data_3d = np.zeros((len(flist), 2 * roi_span, 2 * roi_span))
roi_rows = slice(roi_i - roi_span, roi_i + roi_span)
roi_cols = slice(roi_j - roi_span, roi_j + roi_span)

# ------------------- Initialize Data Storage -------------------
frequencies = np.empty(len(flist), dtype=float)
average_pixel_contrast = np.empty(len(flist), dtype=float)

# Pre-format all SRS command strings (avoid per-step string formatting)
freq_on_cmds = ["freq" + str(n) for n in flist]
freq_off_cmd = "freq" + str(3.0e9)

# Setup real-time matplotlib plotting
plt.ion()  # Enable interactive mode
fig_realtime, ax_realtime = plt.subplots(figsize=(10, 6))
fig_realtime.suptitle('ODMR Spectrum - Real-time Acquisition', fontsize=12, fontweight='bold')
ax_realtime.set_xlabel('Frequency (GHz)', fontsize=11)
ax_realtime.set_ylabel('Average Contrast', fontsize=11)
ax_realtime.set_ylim([0.92, 1.02])
ax_realtime.grid(True, alpha=0.3)
line_plot, = ax_realtime.plot([], [], 'k.-', linewidth=1.5, markersize=4, markeredgewidth=0.5, label='ODMR Data')
ax_realtime.legend(loc='upper right')
ax_realtime.set_xlim([start_frequency/1e9 - 0.005, end_frequency/1e9 + 0.005])
plt.show(block=False)

# ------------------- Data Acquisition Loop -------------------
total_acquiring_time = 0
cam.set_exposure(0.1)
#cam.set_roi(1100, 1400, 1050, 1350,int(bin),int(bin))
cam.set_roi(1000,1500,1000,1500)
try:
    cam.set_frame_period(0.1)
except Exception:
    pass
cam.setup_acquisition(mode='sequence', nframes=max(32, 5 * 4))

print(f"\nStarting acquisition sweep over {len(flist)} frequency points...")
print("Press 'q' in the plot window to stop acquisition early.\n")

# Flag to stop acquisition
stop_acquisition_flag = [False]

def on_key_event(event):
    """Stop acquisition if 'q' is pressed"""
    if event.key == 'q':
        stop_acquisition_flag[0] = True
        print("\n⚠ Stopping acquisition (q pressed)...")

fig_realtime.canvas.mpl_connect('key_press_event', on_key_event)

acq_start_time = time.time()
cam.start_acquisition()
for idx, n in enumerate(flist):
    start_time = time.time()
    
    # Check if stop signal was received
    if stop_acquisition_flag[0]:
        break
    
    # Progress indicator with elapsed time
    progress = (idx + 1) / len(flist) * 100
    elapsed_sec = time.time() - acq_start_time
    e_m, e_s = int(elapsed_sec // 60), int(elapsed_sec % 60)
    print(f"[{progress:5.1f}%] Freq: {n/1e9:.4f} GHz | Elapsed: {e_m}m {e_s:02d}s  (Press 'q' to stop)", end='\r')
    
    # Set MW frequency ON (already pipelined from previous iteration on idx>0)
    if idx == 0:
        srs.write(freq_on_cmds[0])

    # Grab "on" image
    on_image = camera_readout(0.1, 5)
    on_image_roi = on_image[roi_rows, roi_cols]

    # Set MW frequency OFF (detuned)
    srs.write(freq_off_cmd)
    off_image = camera_readout(0.1, 5)
    off_image_roi = off_image[roi_rows, roi_cols]

    # Pipeline: set NEXT ON frequency now (during computation), saves one VISA round-trip
    if idx + 1 < len(flist):
        srs.write(freq_on_cmds[idx + 1])

    # Compute contrast map (safe division, no NaN)
    np.divide(on_image_roi, off_image_roi, out=contrast_data_3d[idx], where=off_image_roi != 0)
    contrast_data_3d[idx][~np.isfinite(contrast_data_3d[idx])] = 0

    # Average contrast
    avg_contrast = np.mean(contrast_data_3d[idx])
    frequencies[idx] = n / 1e9
    average_pixel_contrast[idx] = avg_contrast

    # Real-time per-point plot update (flush_events is much faster than plt.pause)
    line_plot.set_data(frequencies[:idx+1], average_pixel_contrast[:idx+1])
    fig_realtime.canvas.draw_idle()
    fig_realtime.canvas.flush_events()

    # Time accounting
    end_time = time.time()
    total_acquiring_time += (end_time - start_time)

print("\n" + "="*60)
print("Acquisition complete!")
print("="*60)

cam.stop_acquisition()

# Trim data array if acquisition was stopped early
point_count = idx if stop_acquisition_flag[0] else idx + 1
if point_count < len(flist):
    contrast_data_3d = contrast_data_3d[:point_count]
    frequencies = frequencies[:point_count]
    average_pixel_contrast = average_pixel_contrast[:point_count]
    print(f"\nNote: Early stop detected. Data trimmed to {point_count} frequency points.\n")

# ================ POST-ACQUISITION PROCESSING ================
t_m, t_s = int(total_acquiring_time // 60), int(total_acquiring_time % 60)
print(f"Total acquiring time for all frequency points: {t_m}m {t_s:02d}s\n")

# Save data
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save_path = os.path.join(save_directory, "contrast_data_3d_2.npy")
np.save(save_path, contrast_data_3d)
print(f"✓ Data saved to: {save_path}\n")

# Turn off interactive mode for final plots
plt.ioff()

# Plot final analysis figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ODMR Spectrum with fit
ax1.plot(frequencies, average_pixel_contrast, 'b.-', linewidth=2, markersize=8, label='Acquired Data')
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