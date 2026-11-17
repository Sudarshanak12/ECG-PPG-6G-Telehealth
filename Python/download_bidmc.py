
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Dataset" / "PPG" / "BIDMC" / "bidmc_csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_01_Signals.csv"

FILE = DATA_DIR / "bidmc_01_Signals.csv"

print("Downloading BIDMC recording...")

if not FILE.exists():
    urlretrieve(URL, FILE)

print("Download complete!")

df = pd.read_csv(FILE)

print("\nDataset shape:", df.shape)
print("\nSignal columns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDATASET READY!")
