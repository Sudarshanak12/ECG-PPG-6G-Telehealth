
import pandas as pd
import numpy as np
from scipy.io import savemat
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset path
FILE = ROOT / "Dataset" / "PPG" / "BIDMC" / "bidmc_csv" / "bidmc_01_Signals.csv"

# Results directory
RESULTS = ROOT / "Results"
RESULTS.mkdir(exist_ok=True)

# Load original dataset
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# Extract signals
fs = 125

time = df["Time [s]"].to_numpy()
ecg = df["II"].to_numpy()
ppg = df["PLETH"].to_numpy()

# Export to MATLAB format
output_file = RESULTS / "ECG_PPG_Data.mat"

savemat(output_file, {
    "time": time,
    "ecg": ecg,
    "ppg": ppg,
    "fs": fs
})

print("MATLAB export successful!")
print("ECG samples:", len(ecg))
print("PPG samples:", len(ppg))
print("Sampling frequency:", fs, "Hz")
print("Saved:", output_file)
