import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
from lmfit import Model
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MW_VISA_ADDRESS     = 'TCPIP0::10.106.9.106::inst0::INSTR'
LOCKIN_VISA_ADDRESS = 'USB0::0xB506::0x2000::002907::INSTR'
SAVE_DIRECTORY      = r'C:\Users\Buddhika\OneDrive\Documents\Lab\rawdata_roi\lia'

os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# INSTRUMENT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def write_mw(commands, visa_address=MW_VISA_ADDRESS):
    """Send a list of write commands to the MW generator."""
    rm = pyvisa.ResourceManager()
    mw = rm.open_resource(visa_address)
    try:
        for cmd in commands:
            mw.write(cmd)
    finally:
        mw.close()


def query_mw(command, visa_address=MW_VISA_ADDRESS):
    """Query the MW generator and return the response string."""
    rm = pyvisa.ResourceManager()
    mw = rm.open_resource(visa_address)
    try:
        return mw.query(command)
    finally:
        mw.close()


def write_lockin(commands):
    """Send commands to lock-in; yield responses for each command."""
    rm = pyvisa.ResourceManager()
    lockin = rm.open_resource(LOCKIN_VISA_ADDRESS)
    try:
        if isinstance(commands, str):
            yield lockin.query(commands).strip()
        elif isinstance(commands, list):
            for cmd in commands:
                yield lockin.query(cmd).strip()
    finally:
        lockin.close()


def get_data_point_lockin(iterations, channel1_name, channel2_name, aux_channel_name):
    """Yield ``iterations`` snap readings from the lock-in amplifier."""
    rm = pyvisa.ResourceManager()
    lockin = rm.open_resource(LOCKIN_VISA_ADDRESS)
    try:
        for _ in range(iterations):
            raw = lockin.query(
                f'SNAP? {channel1_name}, {channel2_name}, {aux_channel_name}'
            ).strip()
            yield raw.split(",")
    finally:
        lockin.close()


# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — INITIALISE INSTRUMENTS
# ──────────────────────────────────────────────────────────────────────────────
print("Instrument ID:", query_mw("*IDN?").strip())
write_mw(["FREQ 2.87GHz", "POW 3dBm"])
write_mw(["FDEV 0.7 MHz", "AMPR 0 dBm"])

# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — ODMR FREQUENCY SWEEP
# ──────────────────────────────────────────────────────────────────────────────
start_frequency    = 2855   # MHz
end_frequency      = 2885   # MHz
steps              = 100    # kHz step size
number_of_averages = 10

mhz_steps, X, Y, AUX = [], [], [], []

for frequency_khz in range(
    int(start_frequency * 1000),
    int(end_frequency   * 1000),
    int(steps)
):
    freq_mhz = frequency_khz / 1000.0
    write_mw([f'FREQ {freq_mhz:.1f} MHz'])
    time.sleep(0.03)

    mhz_steps.append(freq_mhz)
    readings = np.array(
        list(get_data_point_lockin(number_of_averages, "X", "Y", "IN1")),
        dtype=float
    )
    x, y, aux = np.mean(readings, axis=0)
    X.append(x)
    Y.append(y)
    AUX.append(aux)

X   = np.array(X)
Y   = np.array(Y)
AUX = np.array(AUX)
R   = np.sqrt(X**2 + Y**2)

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(mhz_steps, X,   label="X",   linewidth=1)
ax[0].plot(mhz_steps, Y,   label="Y",   linewidth=1)
ax[0].plot(mhz_steps, R,   label="R",   linewidth=1)
ax[1].plot(mhz_steps, AUX, label="AUX", linewidth=1)
ax[0].set_ylabel("Signal (V)")
ax[1].set_xlabel("Microwave Frequency (MHz)")
ax[1].set_ylabel("AUX Signal (V)")
ax[0].legend()
plt.tight_layout()
plt.show()

r_file = os.path.join(SAVE_DIRECTORY, "LIA_signal_vs_frequency_R.txt")
np.savetxt(r_file,
           np.column_stack((mhz_steps, R)),
           header="Microwave Frequency [MHz], LIA Signal R [V]",
           fmt="%.6f", delimiter=", ")

aux_file = os.path.join(SAVE_DIRECTORY, "AUX_output_vs_frequency.txt")
np.savetxt(aux_file,
           np.column_stack((mhz_steps, AUX)),
           header="Microwave Frequency [MHz], AUX Signal [V]",
           fmt="%.6f", delimiter=", ")

print(f"Sweep data saved to {SAVE_DIRECTORY}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 3 — LORENTZIAN FIT OF AUX SIGNAL
# ──────────────────────────────────────────────────────────────────────────────
data = np.loadtxt(aux_file, delimiter=",", skiprows=1)
frequency       = data[:, 0]   # MHz
aux_signal      = data[:, 1]   # V
frequency_in_Hz = frequency * 1e6

def lorentzian_10peak(x, off, x0, FWHM, a, a2, a3, hfs, x0b):
    def lor(x, c, amp):
        return amp / ((x - c)**2 + (0.5 * FWHM)**2)
    return off + (
        lor(x, x0  - 2*hfs, a)  + lor(x, x0  - hfs, a2) +
        lor(x, x0,           a3) + lor(x, x0  + hfs, a2) +
        lor(x, x0  + 2*hfs, a)  +
        lor(x, x0b - 2*hfs, a)  + lor(x, x0b - hfs, a2) +
        lor(x, x0b,          a3) + lor(x, x0b + hfs, a2) +
        lor(x, x0b + 2*hfs, a)
    )

lmodel = Model(lorentzian_10peak)
params = lmodel.make_params(
    off=1, x0=2.8651e9, FWHM=0.001e8,
    a=-4e8, a2=-4e9, a3=-4e8,
    hfs=2.16e6, x0b=2.875e9
)

norm_aux = aux_signal / np.max(aux_signal)
result   = lmodel.fit(norm_aux, params, x=frequency_in_Hz)

fwhm_mhz = abs(result.params['FWHM'].value) / 1e6
print(f"FWHM: {fwhm_mhz:.2f} MHz")

plt.figure(figsize=(6, 3))
plt.plot(frequency, norm_aux,        '.', label='Normalised AUX data')
plt.plot(frequency, result.best_fit, '-', label='Lorentzian fit', linewidth=1.5)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Normalised AUX Signal")
plt.legend()
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# PART 4 — GAUSSIAN DERIVATIVE FIT (ZERO CROSSING)
# ──────────────────────────────────────────────────────────────────────────────
data_lia   = np.loadtxt(r_file, delimiter=",", skiprows=1)
frequency  = data_lia[:, 0]   # MHz
lia_signal = data_lia[:, 1]   # V

frequency_in_Hz = frequency * 1e6
grad_signal = np.gradient(lia_signal, frequency_in_Hz)

def gaussian_derivative(f, A, mu, sigma, offset):
    return A * (f - mu) * np.exp(-((f - mu)**2) / (2 * sigma**2)) + offset

p0   = [1.0, frequency[np.argmax(lia_signal)], 1.0, 0.0]
popt, _ = curve_fit(gaussian_derivative, frequency, lia_signal, p0=p0)
A_fit, mu_fit, sigma_fit, offset_fit = popt

fitted_signal           = gaussian_derivative(frequency, *popt)
zero_crossing_frequency = mu_fit
zc_idx                  = np.argmin(np.abs(frequency - zero_crossing_frequency))
zero_crossing_slope     = grad_signal[zc_idx]

print(f"Zero crossing at:  {zero_crossing_frequency:.3f} MHz")
print(f"Slope at crossing: {zero_crossing_slope:.3e} V/Hz")

mask = (frequency >= 2855) & (frequency <= 2885)
plt.figure(figsize=(7, 3))
plt.plot(frequency[mask], lia_signal[mask],    label="LIA Signal R",          linewidth=2)
plt.plot(frequency[mask], fitted_signal[mask], 'r--', label="Gaussian deriv. fit", linewidth=2)
plt.axvline(zero_crossing_frequency, color='green', linestyle='--',
            label=f"Zero crossing: {zero_crossing_frequency:.1f} MHz")
plt.xlabel("Frequency (MHz)", fontsize=14)
plt.ylabel("LIA Signal (V)", fontsize=14)
plt.tick_params(labelsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# PART 5 — SET MW TO ZERO-CROSSING FREQUENCY
# ──────────────────────────────────────────────────────────────────────────────
midpoint_hz = zero_crossing_frequency * 1e6
write_mw([f"FREQ {midpoint_hz:.0f}"])
time.sleep(1)
print(f"MW generator set to: {query_mw('FREQ?').strip()} Hz")

# ──────────────────────────────────────────────────────────────────────────────
# PART 6 — TIME-TRACE ACQUISITION (LOCK-IN CAPTURE)
# ──────────────────────────────────────────────────────────────────────────────
buffer_size   = 2500   # kilobytes
sampling_rate = 2      # samples per second

rm         = pyvisa.ResourceManager()
instrument = rm.open_resource(LOCKIN_VISA_ADDRESS)
instrument.timeout = 30000

try:
    for cmd in ['CAPTURECFG 0', f'CAPTURELEN {buffer_size}', f'CAPTURERATE {sampling_rate}']:
        instrument.write(cmd)

    instrument.write('CAPTURESTART ONE, IMM')
    real_rate = float(instrument.query('CAPTURERATE?'))
    time.sleep(1.5 * buffer_size * 1000 / (4 * real_rate))
    instrument.write('CAPTURESTOP')

    voltage_data = []
    for k in range((buffer_size - 1) // 64 + 1):
        start = k * 64
        count = min(64, buffer_size - start)
        voltage_data += instrument.query_binary_values(f'CAPTUREGET? {start}, {count}')
finally:
    instrument.close()

time_data = [i / real_rate for i in range(len(voltage_data))]
print(f"Acquired {len(voltage_data)} samples at {real_rate} S/s")

timetrace_file = os.path.join(SAVE_DIRECTORY, "LIA_signal_variation_ins.txt")
with open(timetrace_file, 'w') as f:
    f.write("# Time (s)\tVoltage (V)\n")
    for t_val, v in zip(time_data, voltage_data):
        f.write(f"{t_val:.8f}\t{v:.9f}\n")
print(f"Time-trace saved to {timetrace_file}")

plt.figure(figsize=(6, 3))
plt.plot(time_data, voltage_data, linewidth=0.5, label="LIA X Signal")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("X-Channel Voltage Variation Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# PART 7 — MAGNETIC SENSITIVITY CALCULATION
# ──────────────────────────────────────────────────────────────────────────────
sensitive_file = os.path.join(SAVE_DIRECTORY, "LIA_signal_variation_avg.txt")
t, amplitudes  = np.loadtxt(sensitive_file, delimiter="\t", unpack=True, skiprows=1)

plt.figure(figsize=(6, 3))
plt.plot(t, amplitudes, linewidth=0.4, label="LIA Signal")
plt.xlabel("Time (s)")
plt.ylabel("LIA Signal (V)")
plt.title("LIA Signal Variations")
plt.legend()
plt.tight_layout()
plt.show()

LI_output = np.mean(amplitudes)
print(f"Mean LI output: {LI_output:.2e} V")

cos_angle = np.cos(np.deg2rad(54.75))
factor    = (3.568e-11 / zero_crossing_slope) * cos_angle
mag_field = amplitudes * factor

mag_field_file = os.path.join(SAVE_DIRECTORY, "Magnetic_Field_Sensitive_avg.txt")
np.savetxt(mag_field_file,
           np.column_stack((t, mag_field)),
           delimiter="\t",
           header="Time (s)\tMagnetic Field (T)")

plt.figure(figsize=(6, 3))
plt.plot(t, mag_field, linewidth=0.4, label="Magnetic Field")
plt.xlabel("Time (s)")
plt.ylabel("Magnetic Field (T)")
plt.title("Magnetic Field Variations")
plt.legend()
plt.tight_layout()
plt.show()
