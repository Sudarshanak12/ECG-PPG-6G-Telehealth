
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.fft import dct, idct

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "Results" / "ECG_PPG_Preprocessed.npz"

RESULTS = ROOT / "Results"
RESULTS.mkdir(exist_ok=True)

# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================

data = np.load(DATA_FILE)

ecg = data["ecg_normalized"]
ppg = data["ppg_normalized"]

fs = float(data["fs"])

print("DATA LOADED SUCCESSFULLY")
print("Sampling frequency:", fs)

# Use 512 samples for the first experiment
N = 512
K = 30

ecg_segment = ecg[:N]
ppg_segment = ppg[:N]

# ==========================================
# ORTHOGONAL MATCHING PURSUIT
# ==========================================

def omp(A, y, sparsity):

    residual = y.copy()

    support = []

    coefficients = np.array([])

    for iteration in range(sparsity):

        correlations = A.T @ residual

        selected = int(np.argmax(np.abs(correlations)))

        if selected in support:
            break

        support.append(selected)

        selected_matrix = A[:, support]

        coefficients = np.linalg.lstsq(
            selected_matrix,
            y,
            rcond=None
        )[0]

        residual = y - selected_matrix @ coefficients

        if np.linalg.norm(residual) < 1e-8:
            break

    alpha = np.zeros(A.shape[1])

    if len(support) > 0:
        alpha[support] = coefficients

    return alpha


# ==========================================
# COMPRESSIVE SENSING
# ==========================================

def compressive_sensing(signal, ratio):

    N = len(signal)

    M = max(1, int(round(N * ratio)))

    # DCT sparsifying basis
    Psi = idct(
        np.eye(N),
        axis=0,
        norm="ortho"
    )

    # Gaussian measurement matrix
    Phi = np.random.randn(M, N) / np.sqrt(M)

    # Combined sensing matrix
    A = Phi @ Psi

    # Normalize dictionary columns
    column_norms = np.linalg.norm(A, axis=0)

    A_normalized = A / np.maximum(column_norms, 1e-12)

    # Compressed measurements
    y = Phi @ signal

    # Recover sparse DCT coefficients
    alpha = omp(A_normalized, y, K)

    # Correct coefficient scaling
    alpha = alpha / np.maximum(column_norms, 1e-12)

    # Reconstruct original signal
    reconstructed = Psi @ alpha

    return reconstructed, M


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def calculate_metrics(original, reconstructed):

    mse = np.mean((original - reconstructed) ** 2)

    signal_power = np.sum(original ** 2)

    noise_power = np.sum((original - reconstructed) ** 2)

    snr = 10 * np.log10(
        signal_power / max(noise_power, 1e-12)
    )

    prd = 100 * np.sqrt(
        noise_power / max(signal_power, 1e-12)
    )

    return mse, snr, prd


# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

ratios = [0.10, 0.20, 0.30, 0.40, 0.50]

signals = {
    "ECG": ecg_segment,
    "PPG": ppg_segment
}

all_results = {}

# ==========================================
# RUN EXPERIMENT
# ==========================================

for signal_name, signal in signals.items():

    print("\n================================")
    print(signal_name, "COMPRESSIVE SENSING")
    print("================================")

    plt.figure(figsize=(14, 10))

    results = []

    for i, ratio in enumerate(ratios):

        # Reproducible random measurement matrix
        np.random.seed(42 + i)

        reconstructed, M = compressive_sensing(
            signal,
            ratio
        )

        mse, snr, prd = calculate_metrics(
            signal,
            reconstructed
        )

        results.append([ratio, M, mse, snr, prd])

        print(
            f"Ratio: {ratio*100:.0f}% | "
            f"Measurements: {M} | "
            f"MSE: {mse:.6f} | "
            f"SNR: {snr:.2f} dB | "
            f"PRD: {prd:.2f}%"
        )

        # Plot reconstruction
        plt.subplot(5, 1, i + 1)

        plt.plot(
            signal,
            label="Original",
            linewidth=1
        )

        plt.plot(
            reconstructed,
            "--",
            label="OMP Reconstruction",
            linewidth=1
        )

        plt.title(
            f"{signal_name} - Measurement Ratio: {ratio*100:.0f}%"
        )

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS / f"{signal_name}_OMP_Reconstruction.png",
        dpi=300
    )

    plt.show()

    all_results[signal_name] = np.array(results)


# ==========================================
# SAVE RESULTS
# ==========================================

np.savez(
    RESULTS / "OMP_Results.npz",
    ECG=all_results["ECG"],
    PPG=all_results["PPG"]
)

print("\nCOMPRESSIVE SENSING EXPERIMENT COMPLETED!")
print("Results saved in:", RESULTS)
