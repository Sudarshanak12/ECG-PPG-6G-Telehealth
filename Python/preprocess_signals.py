
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# File locations
DATA_FILE = ROOT / "Results" / "ECG_PPG_Data.mat"
RESULTS = ROOT / "Results"

RESULTS.mkdir(exist_ok=True)

# Load MATLAB-exported data
data = sio.loadmat(DATA_FILE)

ecg = data["ecg"].flatten().astype(float)
ppg = data["ppg"].flatten().astype(float)
fs = float(data["fs"].squeeze())

time = np.arange(len(ecg)) / fs

print("DATA LOADED SUCCESSFULLY")
print("Sampling frequency:", fs)
print("ECG samples:", len(ecg))
print("PPG samples:", len(ppg))

# Butterworth bandpass filter
def bandpass_filter(signal, low, high, fs, order=4):

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)

# Filter ECG and PPG
ecg_filtered = bandpass_filter(ecg, 0.5, 40, fs)
ppg_filtered = bandpass_filter(ppg, 0.5, 8, fs)

# Normalize signals using z-score
ecg_normalized = (
    ecg_filtered - np.mean(ecg_filtered)
) / np.std(ecg_filtered)

ppg_normalized = (
    ppg_filtered - np.mean(ppg_filtered)
) / np.std(ppg_filtered)

# Plot first 10 seconds
N = int(10 * fs)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(time[:N], ecg[:N])
axes[0, 0].set_title("Raw ECG")

axes[0, 1].plot(time[:N], ecg_normalized[:N])
axes[0, 1].set_title("Filtered and Normalized ECG")

axes[1, 0].plot(time[:N], ppg[:N])
axes[1, 0].set_title("Raw PPG")

axes[1, 1].plot(time[:N], ppg_normalized[:N])
axes[1, 1].set_title("Filtered and Normalized PPG")

for ax in axes.flat:
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS / "Preprocessing_Comparison.png",
    dpi=300
)

plt.show()

# Save processed signals
np.savez(
    RESULTS / "ECG_PPG_Preprocessed.npz",
    time=time,
    ecg_raw=ecg,
    ppg_raw=ppg,
    ecg_filtered=ecg_filtered,
    ppg_filtered=ppg_filtered,
    ecg_normalized=ecg_normalized,
    ppg_normalized=ppg_normalized,
    fs=fs
)

print("\nPREPROCESSING COMPLETED!")
print("Saved: Results/ECG_PPG_Preprocessed.npz")
