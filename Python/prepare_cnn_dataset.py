
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR / "Dataset" / "PPG" / "BIDMC" / "bidmc_csv"
)

OUTPUT_DIR = PROJECT_DIR / "Deep_Learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "BIDMC_CNN_Dataset.npz"

FS = 125
WINDOW_SIZE = 512
STRIDE = 256
RANDOM_SEED = 42

# =====================================================
# 2. FILTERING
# =====================================================

def filter_signal(signal, low, high):
    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        fs=FS,
        output="sos"
    )
    return sosfiltfilt(sos, signal)


def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return None

    return (signal - mean) / std


# =====================================================
# 3. LOAD RECORDINGS
# =====================================================

files = sorted(DATASET_DIR.glob("bidmc_*_Signals.csv"))

if not files:
    raise FileNotFoundError(
        f"No signal CSV files found in: {DATASET_DIR}"
    )

print(f"Found {len(files)} recordings.")

recordings = {}

for file in files:
    try:
        df = pd.read_csv(file)

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        ecg_column = next(
            (c for c in df.columns if c.strip().upper() == "II"),
            None
        )

        ppg_column = next(
            (c for c in df.columns if c.strip().upper() == "PLETH"),
            None
        )

        if ecg_column is None or ppg_column is None:
            print(f"Skipping {file.name}: required columns missing.")
            continue

        ecg = pd.to_numeric(
            df[ecg_column], errors="coerce"
        ).to_numpy(dtype=float)

        ppg = pd.to_numeric(
            df[ppg_column], errors="coerce"
        ).to_numpy(dtype=float)

        valid = np.isfinite(ecg) & np.isfinite(ppg)
        ecg = ecg[valid]
        ppg = ppg[valid]

        if len(ecg) < WINDOW_SIZE or len(ppg) < WINDOW_SIZE:
            print(f"Skipping {file.name}: insufficient samples.")
            continue

        ecg_filtered = filter_signal(ecg, 0.5, 40)
        ppg_filtered = filter_signal(ppg, 0.5, 8)

        ecg_normalized = normalize_signal(ecg_filtered)
        ppg_normalized = normalize_signal(ppg_filtered)

        if ecg_normalized is None or ppg_normalized is None:
            print(f"Skipping {file.name}: constant signal.")
            continue

        recording_id = file.stem.replace("_Signals", "")

        recordings[recording_id] = {
            "ecg": ecg_normalized.astype(np.float32),
            "ppg": ppg_normalized.astype(np.float32)
        }

        print(f"Processed {recording_id}: {len(ecg)} samples")

    except Exception as error:
        print(f"Error processing {file.name}: {error}")


# =====================================================
# 4. RECORDING-LEVEL TRAIN / VALIDATION / TEST SPLIT
# =====================================================

recording_ids = sorted(recordings.keys())

if len(recording_ids) < 3:
    raise RuntimeError("Not enough valid recordings for dataset split.")

rng = np.random.default_rng(RANDOM_SEED)
shuffled_ids = list(rng.permutation(recording_ids))

n_total = len(shuffled_ids)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

train_ids = shuffled_ids[:n_train]
val_ids = shuffled_ids[n_train:n_train + n_val]
test_ids = shuffled_ids[n_train + n_val:]

print("\nRecording split:")
print("Training:", train_ids)
print("Validation:", val_ids)
print("Testing:", test_ids)


# =====================================================
# 5. CREATE WINDOWS WITHOUT CROSSING RECORDINGS
# =====================================================

def create_windows(ids):
    ecg_windows = []
    ppg_windows = []
    window_recording_ids = []

    for recording_id in ids:
        ecg = recordings[recording_id]["ecg"]
        ppg = recordings[recording_id]["ppg"]

        for start in range(
            0,
            len(ecg) - WINDOW_SIZE + 1,
            STRIDE
        ):
            end = start + WINDOW_SIZE

            ecg_windows.append(ecg[start:end])
            ppg_windows.append(ppg[start:end])
            window_recording_ids.append(recording_id)

    return (
        np.asarray(ecg_windows, dtype=np.float32),
        np.asarray(ppg_windows, dtype=np.float32),
        np.asarray(window_recording_ids, dtype=str)
    )


train_ecg, train_ppg, train_recordings = create_windows(train_ids)
val_ecg, val_ppg, val_recordings = create_windows(val_ids)
test_ecg, test_ppg, test_recordings = create_windows(test_ids)


# =====================================================
# 6. SAVE COMPLETE DATASET
# =====================================================

np.savez_compressed(
    OUTPUT_FILE,

    train_ecg=train_ecg,
    train_ppg=train_ppg,

    val_ecg=val_ecg,
    val_ppg=val_ppg,

    test_ecg=test_ecg,
    test_ppg=test_ppg,

    train_recordings=train_recordings,
    val_recordings=val_recordings,
    test_recordings=test_recordings,

    fs=np.array(FS),
    window_size=np.array(WINDOW_SIZE),
    stride=np.array(STRIDE)
)

# =====================================================
# 7. REPORT
# =====================================================

print("\n======================================")
print("CNN DATASET PREPARATION COMPLETED")
print("======================================")

print("Valid recordings:", len(recording_ids))
print("Training windows:", len(train_ecg))
print("Validation windows:", len(val_ecg))
print("Testing windows:", len(test_ecg))

print("\nSaved dataset to:")
print(OUTPUT_FILE)
