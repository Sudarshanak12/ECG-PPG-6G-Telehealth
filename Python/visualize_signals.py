
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset path
FILE = (
    ROOT / "Dataset" / "PPG" / "BIDMC" /
    "bidmc_csv" / "bidmc_01_Signals.csv"
)

# Load dataset
df = pd.read_csv(FILE)

# Remove hidden spaces from column names
df.columns = df.columns.str.strip()

# Display actual column names
print("AVAILABLE COLUMNS:")
for col in df.columns:
    print(repr(col))

# Sampling frequency
FS = 125

# Identify ECG and PPG columns
ecg_column = next(
    (c for c in df.columns if c.strip().upper() == "II"),
    None
)

ppg_column = next(
    (c for c in df.columns if c.strip().upper() == "PLETH"),
    None
)

time_column = next(
    (c for c in df.columns if c.strip().upper() == "TIME [S]"),
    None
)

# Check columns
if ecg_column is None or ppg_column is None:
    raise ValueError(
        f"Required signals not found. Available columns: {df.columns.tolist()}"
    )

# Extract signals
ecg = pd.to_numeric(df[ecg_column]).to_numpy()
ppg = pd.to_numeric(df[ppg_column]).to_numpy()

if time_column is not None:
    time = pd.to_numeric(df[time_column]).to_numpy()
else:
    time = np.arange(len(ecg)) / FS

print("\nSIGNAL EXTRACTION SUCCESSFUL!")
print("ECG column:", ecg_column)
print("PPG column:", ppg_column)
print("Sampling frequency:", FS, "Hz")
print("Total samples:", len(ecg))
print("Duration:", time[-1], "seconds")

# Create results directory
RESULTS = ROOT / "Results"
RESULTS.mkdir(exist_ok=True)

# Plot ECG
plt.figure(figsize=(14, 4))
plt.plot(time, ecg, linewidth=0.8)
plt.title("ECG Signal - Lead II")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "ECG_Lead_II.png", dpi=300)
plt.show()

# Plot PPG
plt.figure(figsize=(14, 4))
plt.plot(time, ppg, linewidth=0.8)
plt.title("PPG Signal - PLETH")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "PPG_PLETH.png", dpi=300)
plt.show()

# Save extracted signals
np.savez(
    RESULTS / "ECG_PPG_raw.npz",
    time=time,
    ecg=ecg,
    ppg=ppg,
    fs=FS
)

print("\nECG AND PPG SAVED SUCCESSFULLY!")
