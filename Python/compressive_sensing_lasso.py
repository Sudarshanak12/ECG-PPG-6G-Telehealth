
import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path
from scipy.fft import idct
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "Results" /
    "01_Preprocessing" /
    "ECG_PPG_Preprocessed.npz"
)

OMP_DIR = (
    BASE_DIR / "Results" /
    "05_OMP_Reconstruction"
)

LASSO_DIR = (
    BASE_DIR / "Results" /
    "06_LASSO_Reconstruction"
)

COMPARISON_DIR = (
    BASE_DIR / "Results" /
    "07_Algorithm_Comparison"
)

OMP_DIR.mkdir(parents=True, exist_ok=True)
LASSO_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. EXPERIMENT SETTINGS
# ==========================================================

N = 512

MEASUREMENT_RATIOS = [
    0.10, 0.20, 0.30, 0.40, 0.50
]

OMP_SPARSITY = 30

# LASSO regularisation search range
ALPHA_FACTORS = np.logspace(-4, -0.3, 18)

RANDOM_SEED = 42

# ==========================================================
# 3. LOAD PREPROCESSED DATA
# ==========================================================

print("\n============================================")
print("PHASE 3: OMP vs LASSO COMPARISON")
print("============================================")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nPreprocessed dataset not found:\n{DATA_FILE}\n"
        "Check that ECG_PPG_Preprocessed.npz is inside "
        "Results/01_Preprocessing/"
    )

data = np.load(DATA_FILE)

signals = {
    "ECG": np.asarray(
        data["ecg_normalized"][:N],
        dtype=float
    ),
    "PPG": np.asarray(
        data["ppg_normalized"][:N],
        dtype=float
    )
}

for name, signal in signals.items():

    if len(signal) != N:
        raise ValueError(
            f"{name} contains fewer than {N} samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{name} contains invalid values."
        )

print("\nDataset loaded successfully.")
print("ECG samples:", len(signals["ECG"]))
print("PPG samples:", len(signals["PPG"]))

# ==========================================================
# 4. CREATE DCT SPARSIFYING BASIS
# ==========================================================

Psi = idct(
    np.eye(N),
    axis=0,
    norm="ortho"
)

print("\nDCT sparsifying basis created.")

# ==========================================================
# 5. PERFORMANCE METRICS
# ==========================================================

def calculate_metrics(original, reconstructed):

    error = original - reconstructed

    mse = np.mean(error ** 2)

    signal_power = np.sum(original ** 2)

    noise_power = np.sum(error ** 2)

    snr = 10 * np.log10(
        signal_power / (noise_power + 1e-12)
    )

    prd = 100 * np.sqrt(
        noise_power / (signal_power + 1e-12)
    )

    return mse, snr, prd


# ==========================================================
# 6. ORTHOGONAL MATCHING PURSUIT
# ==========================================================

def omp_reconstruction(A, y, sparsity):

    residual = y.copy()

    selected = []

    coefficients = np.zeros(A.shape[1])

    for iteration in range(sparsity):

        correlations = np.abs(A.T @ residual)

        if selected:
            correlations[selected] = 0

        index = np.argmax(correlations)

        selected.append(index)

        selected_A = A[:, selected]

        coefficients_selected = np.linalg.lstsq(
            selected_A,
            y,
            rcond=None
        )[0]

        residual = (
            y - selected_A @ coefficients_selected
        )

    coefficients[selected] = coefficients_selected

    return coefficients


# ==========================================================
# 7. LASSO WITH MEASUREMENT-BASED VALIDATION
# ==========================================================

def lasso_reconstruction(
    A,
    y,
    Psi,
    column_norms,
    random_state
):

    # Split measurement rows into training and validation
    indices = np.arange(len(y))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state
    )

    A_train = A[train_indices]
    y_train = y[train_indices]

    A_validation = A[validation_indices]
    y_validation = y[validation_indices]

    # Maximum alpha giving zero coefficients
    alpha_max = (
        np.max(np.abs(A_train.T @ y_train))
        / len(y_train)
    )

    best_alpha = None
    best_validation_error = float("inf")

    # Select alpha using held-out measurements
    for factor in ALPHA_FACTORS:

        alpha = max(alpha_max * factor, 1e-9)

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=30000,
            tol=1e-6,
            selection="cyclic"
        )

        model.fit(A_train, y_train)

        validation_prediction = (
            A_validation @ model.coef_
        )

        validation_error = np.mean(
            (y_validation - validation_prediction) ** 2
        )

        if validation_error < best_validation_error:

            best_validation_error = validation_error
            best_alpha = alpha

    # Refit using all available measurements
    final_model = Lasso(
        alpha=best_alpha,
        fit_intercept=False,
        max_iter=30000,
        tol=1e-6,
        selection="cyclic"
    )

    final_model.fit(A, y)

    beta = final_model.coef_

    # Undo dictionary column normalization
    sparse_coefficients = beta / column_norms

    reconstructed_signal = (
        Psi @ sparse_coefficients
    )

    return (
        reconstructed_signal,
        best_alpha,
        best_validation_error
    )


# ==========================================================
# 8. EXPERIMENT STORAGE
# ==========================================================

all_results = []

reconstruction_data = {}

metric_data = {
    "ECG": {"OMP": [], "LASSO": []},
    "PPG": {"OMP": [], "LASSO": []}
}

# ==========================================================
# 9. RUN EXPERIMENTS
# ==========================================================

for signal_name, original in signals.items():

    print("\n--------------------------------------------")
    print("Processing:", signal_name)
    print("--------------------------------------------")

    reconstruction_data[signal_name] = {}

    for ratio in MEASUREMENT_RATIOS:

        percentage = int(ratio * 100)

        M = int(round(N * ratio))

        print(
            f"\nMeasurement ratio: {percentage}% "
            f"({M} measurements)"
        )

        # ------------------------------------------
        # Generate reproducible measurement matrix
        # ------------------------------------------

        rng = np.random.default_rng(
            RANDOM_SEED + percentage
        )

        Phi = (
            rng.standard_normal((M, N))
            / np.sqrt(M)
        )

        # ------------------------------------------
        # Construct measurement dictionary
        # ------------------------------------------

        A_raw = Phi @ Psi

        column_norms = np.linalg.norm(
            A_raw,
            axis=0
        )

        column_norms[column_norms == 0] = 1

        A = A_raw / column_norms

        # Compressed measurements
        y = Phi @ original

        # ------------------------------------------
        # OMP reconstruction
        # ------------------------------------------

        omp_coefficients = omp_reconstruction(
            A,
            y,
            min(OMP_SPARSITY, M)
        )

        omp_signal = Psi @ (
            omp_coefficients / column_norms
        )

        # ------------------------------------------
        # LASSO reconstruction
        # ------------------------------------------

        (
            lasso_signal,
            best_alpha,
            validation_error
        ) = lasso_reconstruction(
            A,
            y,
            Psi,
            column_norms,
            RANDOM_SEED + percentage
        )

        # ------------------------------------------
        # Calculate metrics
        # ------------------------------------------

        omp_metrics = calculate_metrics(
            original,
            omp_signal
        )

        lasso_metrics = calculate_metrics(
            original,
            lasso_signal
        )

        metric_data[signal_name]["OMP"].append(
            omp_metrics
        )

        metric_data[signal_name]["LASSO"].append(
            lasso_metrics
        )

        # Save numerical results
        all_results.append([
            signal_name,
            percentage,
            M,
            "OMP",
            *omp_metrics,
            OMP_SPARSITY,
            ""
        ])

        all_results.append([
            signal_name,
            percentage,
            M,
            "LASSO",
            *lasso_metrics,
            "",
            best_alpha
        ])

        # Store reconstruction signals
        reconstruction_data[signal_name][percentage] = {
            "original": original,
            "omp": omp_signal,
            "lasso": lasso_signal
        }

        # ------------------------------------------
        # Print results
        # ------------------------------------------

        print(
            f"OMP   | MSE: {omp_metrics[0]:.6f} "
            f"| SNR: {omp_metrics[1]:.2f} dB "
            f"| PRD: {omp_metrics[2]:.2f}%"
        )

        print(
            f"LASSO | MSE: {lasso_metrics[0]:.6f} "
            f"| SNR: {lasso_metrics[1]:.2f} dB "
            f"| PRD: {lasso_metrics[2]:.2f}%"
        )

        print(
            f"Selected alpha: {best_alpha:.8f}"
        )

        # ------------------------------------------
        # Plot reconstruction comparison
        # ------------------------------------------

        plt.figure(figsize=(13, 4))

        plt.plot(
            original,
            label="Original",
            linewidth=1.5
        )

        plt.plot(
            omp_signal,
            "--",
            label="OMP Reconstruction"
        )

        plt.plot(
            lasso_signal,
            ":",
            label="LASSO Reconstruction",
            linewidth=1.8
        )

        plt.title(
            f"{signal_name}: OMP vs LASSO "
            f"({percentage}% Measurements)"
        )

        plt.xlabel("Sample")
        plt.ylabel("Normalized Amplitude")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"{signal_name}_OMP_vs_LASSO_{percentage}.png"
        )

        plt.savefig(
            LASSO_DIR / filename,
            dpi=200
        )

        plt.close()

        print("Graph saved:", filename)


# ==========================================================
# 10. SAVE RECONSTRUCTION DATA
# ==========================================================

npz_data = {}

for signal_name, ratios in reconstruction_data.items():

    for percentage, values in ratios.items():

        for method, signal in values.items():

            key = (
                f"{signal_name}_{percentage}_{method}"
            )

            npz_data[key] = signal

np.savez_compressed(
    LASSO_DIR / "OMP_LASSO_Reconstructions.npz",
    **npz_data
)

print("\nReconstruction data saved.")


# ==========================================================
# 11. EXPORT METRICS TO CSV
# ==========================================================

csv_path = (
    COMPARISON_DIR / "OMP_LASSO_Metrics.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Signal",
        "Measurement_Ratio_Percent",
        "Number_of_Measurements",
        "Algorithm",
        "MSE",
        "SNR_dB",
        "PRD_Percent",
        "OMP_Sparsity",
        "LASSO_Alpha"
    ])

    writer.writerows(all_results)

print("Metrics CSV saved.")


# ==========================================================
# 12. GENERATE METRIC COMPARISON GRAPHS
# ==========================================================

metric_names = [
    ("MSE", 0, "Mean Squared Error"),
    ("SNR", 1, "Signal-to-Noise Ratio (dB)"),
    ("PRD", 2, "Percentage Root Mean Square Difference (%)")
]

x_values = [
    int(ratio * 100)
    for ratio in MEASUREMENT_RATIOS
]

for signal_name in signals:

    for metric_name, metric_index, ylabel in metric_names:

        omp_values = [
            item[metric_index]
            for item in metric_data[signal_name]["OMP"]
        ]

        lasso_values = [
            item[metric_index]
            for item in metric_data[signal_name]["LASSO"]
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            x_values,
            omp_values,
            marker="o",
            label="OMP"
        )

        plt.plot(
            x_values,
            lasso_values,
            marker="s",
            label="LASSO"
        )

        plt.title(
            f"{signal_name}: {metric_name} Comparison"
        )

        plt.xlabel("Measurement Ratio (%)")
        plt.ylabel(ylabel)

        plt.xticks(x_values)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"{signal_name}_{metric_name}_Comparison.png"
        )

        plt.savefig(
            COMPARISON_DIR / filename,
            dpi=200
        )

        plt.close()

        print("Comparison graph saved:", filename)


# ==========================================================
# 13. FINAL SUMMARY
# ==========================================================

print("\n============================================")
print("PHASE 3 EXPERIMENT COMPLETED!")
print("============================================")

print("\nReconstruction graphs:")
print(LASSO_DIR)

print("\nMetrics and comparison graphs:")
print(COMPARISON_DIR)

print("\nGenerated files:")
print("1. ECG_OMP_vs_LASSO_*.png")
print("2. PPG_OMP_vs_LASSO_*.png")
print("3. OMP_LASSO_Reconstructions.npz")
print("4. OMP_LASSO_Metrics.csv")
print("5. ECG_MSE_Comparison.png")
print("6. ECG_SNR_Comparison.png")
print("7. ECG_PRD_Comparison.png")
print("8. PPG_MSE_Comparison.png")
print("9. PPG_SNR_Comparison.png")
print("10. PPG_PRD_Comparison.png")

print("\nAll experiments finished successfully!")
