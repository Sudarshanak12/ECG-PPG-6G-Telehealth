
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

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

OUTPUT_DIR = (
    BASE_DIR / "Results" /
    "07_Algorithm_Comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. EXPERIMENT SETTINGS
# ==========================================================

N = 512

SEGMENTS_PER_SIGNAL = 12

MEASUREMENT_RATIOS = [10, 20, 30, 40, 50]

RANDOM_SEEDS = [42, 123, 2026]

OMP_SPARSITY = 30

ALPHA_FACTORS = np.logspace(-3, -0.3, 12)

# ==========================================================
# 3. LOAD DATA
# ==========================================================

print("\n==========================================")
print("PHASE 3.5: ROBUSTNESS TESTING")
print("==========================================")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

data = np.load(DATA_FILE)

signals = {
    "ECG": np.asarray(data["ecg_normalized"], dtype=float),
    "PPG": np.asarray(data["ppg_normalized"], dtype=float)
}

for name, signal in signals.items():

    if len(signal) < N:
        raise ValueError(
            f"{name} has fewer than {N} samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

print("\nDataset loaded successfully.")

# ==========================================================
# 4. CREATE DCT BASIS
# ==========================================================

Psi = idct(
    np.eye(N),
    axis=0,
    norm="ortho"
)

print("DCT basis created.")

# ==========================================================
# 5. PERFORMANCE METRICS
# ==========================================================

def calculate_metrics(original, reconstructed):

    error = original - reconstructed

    mse = np.mean(error ** 2)

    signal_power = np.sum(original ** 2)

    error_power = np.sum(error ** 2)

    snr = 10 * np.log10(
        signal_power / (error_power + 1e-12)
    )

    prd = 100 * np.sqrt(
        error_power / (signal_power + 1e-12)
    )

    return mse, snr, prd


# ==========================================================
# 6. ORTHOGONAL MATCHING PURSUIT
# ==========================================================

def omp_reconstruction(A, y, sparsity):

    residual = y.copy()

    selected = []

    coefficients = np.zeros(A.shape[1])

    for _ in range(sparsity):

        correlations = np.abs(A.T @ residual)

        if selected:
            correlations[selected] = 0

        index = int(np.argmax(correlations))

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

        if np.linalg.norm(residual) < 1e-10:
            break

    coefficients[selected] = coefficients_selected

    return coefficients


# ==========================================================
# 7. LASSO WITH HELD-OUT MEASUREMENT VALIDATION
# ==========================================================

def lasso_reconstruction(
    A,
    y,
    Psi,
    column_norms,
    random_seed
):

    indices = np.arange(len(y))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_seed
    )

    A_train = A[train_indices]
    y_train = y[train_indices]

    A_validation = A[validation_indices]
    y_validation = y[validation_indices]

    alpha_max = (
        np.max(np.abs(A_train.T @ y_train))
        / len(y_train)
    )

    best_alpha = None
    best_validation_error = float("inf")

    for factor in ALPHA_FACTORS:

        alpha = max(alpha_max * factor, 1e-9)

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-5,
            selection="cyclic"
        )

        model.fit(A_train, y_train)

        predicted_measurements = (
            A_validation @ model.coef_
        )

        validation_error = np.mean(
            (y_validation - predicted_measurements) ** 2
        )

        if validation_error < best_validation_error:

            best_validation_error = validation_error
            best_alpha = alpha

    # Refit with all measurements using selected alpha
    final_model = Lasso(
        alpha=best_alpha,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-5,
        selection="cyclic"
    )

    final_model.fit(A, y)

    normalized_coefficients = final_model.coef_

    coefficients = (
        normalized_coefficients / column_norms
    )

    reconstructed_signal = Psi @ coefficients

    return (
        reconstructed_signal,
        best_alpha,
        best_validation_error
    )


# ==========================================================
# 8. SELECT MULTIPLE NON-OVERLAPPING SEGMENTS
# ==========================================================

def select_segments(signal, count):

    possible_starts = len(signal) - N + 1

    if possible_starts < count:
        raise ValueError(
            "Not enough samples for requested segments."
        )

    starts = np.linspace(
        0,
        possible_starts - 1,
        count,
        dtype=int
    )

    segments = []

    for start in starts:

        segment = signal[start:start + N].copy()

        segments.append(segment)

    return starts, segments


# ==========================================================
# 9. RUN ROBUSTNESS EXPERIMENT
# ==========================================================

trial_results = []

start_time = time.time()

total_trials = (
    len(signals)
    * SEGMENTS_PER_SIGNAL
    * len(MEASUREMENT_RATIOS)
    * len(RANDOM_SEEDS)
)

trial_number = 0

for signal_name, complete_signal in signals.items():

    print("\n==========================================")
    print("PROCESSING:", signal_name)
    print("==========================================")

    segment_starts, segments = select_segments(
        complete_signal,
        SEGMENTS_PER_SIGNAL
    )

    for segment_number, (start, original) in enumerate(
        zip(segment_starts, segments),
        start=1
    ):

        print(
            f"\n{signal_name}: Segment "
            f"{segment_number}/{SEGMENTS_PER_SIGNAL} "
            f"(starting sample {start})"
        )

        for percentage in MEASUREMENT_RATIOS:

            M = int(round(N * percentage / 100))

            for seed in RANDOM_SEEDS:

                trial_number += 1

                print(
                    f"\nTrial {trial_number}/{total_trials} | "
                    f"{percentage}% | Seed {seed}"
                )

                # --------------------------------------
                # Measurement matrix
                # --------------------------------------

                rng = np.random.default_rng(
                    seed + segment_number * 10000 + percentage
                )

                Phi = (
                    rng.standard_normal((M, N))
                    / np.sqrt(M)
                )

                # --------------------------------------
                # Sensing dictionary
                # --------------------------------------

                A_raw = Phi @ Psi

                column_norms = np.linalg.norm(
                    A_raw,
                    axis=0
                )

                column_norms[column_norms == 0] = 1

                A = A_raw / column_norms

                # --------------------------------------
                # Compressed measurements
                # --------------------------------------

                y = Phi @ original

                # --------------------------------------
                # OMP
                # --------------------------------------

                omp_coefficients = omp_reconstruction(
                    A,
                    y,
                    min(OMP_SPARSITY, M)
                )

                omp_signal = Psi @ (
                    omp_coefficients / column_norms
                )

                # --------------------------------------
                # LASSO
                # --------------------------------------

                (
                    lasso_signal,
                    best_alpha,
                    validation_error
                ) = lasso_reconstruction(
                    A,
                    y,
                    Psi,
                    column_norms,
                    seed + segment_number
                )

                # --------------------------------------
                # Metrics
                # --------------------------------------

                omp_metrics = calculate_metrics(
                    original,
                    omp_signal
                )

                lasso_metrics = calculate_metrics(
                    original,
                    lasso_signal
                )

                trial_results.append([
                    signal_name,
                    segment_number,
                    int(start),
                    percentage,
                    seed,
                    "OMP",
                    *omp_metrics,
                    OMP_SPARSITY,
                    ""
                ])

                trial_results.append([
                    signal_name,
                    segment_number,
                    int(start),
                    percentage,
                    seed,
                    "LASSO",
                    *lasso_metrics,
                    "",
                    best_alpha
                ])

                print(
                    f"OMP   MSE: {omp_metrics[0]:.6f} | "
                    f"SNR: {omp_metrics[1]:.2f} dB"
                )

                print(
                    f"LASSO MSE: {lasso_metrics[0]:.6f} | "
                    f"SNR: {lasso_metrics[1]:.2f} dB"
                )


# ==========================================================
# 10. SAVE INDIVIDUAL TRIAL RESULTS
# ==========================================================

trial_csv = OUTPUT_DIR / "Robustness_Individual_Trials.csv"

with open(trial_csv, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Signal",
        "Segment",
        "Starting_Sample",
        "Measurement_Ratio_Percent",
        "Random_Seed",
        "Algorithm",
        "MSE",
        "SNR_dB",
        "PRD_Percent",
        "OMP_Sparsity",
        "LASSO_Alpha"
    ])

    writer.writerows(trial_results)


# ==========================================================
# 11. CALCULATE MEAN AND STANDARD DEVIATION
# ==========================================================

summary_results = []

metric_columns = {
    "MSE": 6,
    "SNR": 7,
    "PRD": 8
}

for signal_name in signals:

    for percentage in MEASUREMENT_RATIOS:

        for algorithm in ["OMP", "LASSO"]:

            matching_rows = [
                row for row in trial_results
                if row[0] == signal_name
                and row[3] == percentage
                and row[5] == algorithm
            ]

            summary = []

            for metric_name, column in metric_columns.items():

                values = np.array([
                    row[column] for row in matching_rows
                ], dtype=float)

                mean_value = np.mean(values)

                std_value = np.std(values, ddof=1)

                summary.extend([
                    mean_value,
                    std_value
                ])

            summary_results.append([
                signal_name,
                percentage,
                algorithm,
                len(matching_rows),
                *summary
            ])


# ==========================================================
# 12. SAVE SUMMARY CSV
# ==========================================================

summary_csv = OUTPUT_DIR / "Robustness_Summary.csv"

with open(summary_csv, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Signal",
        "Measurement_Ratio_Percent",
        "Algorithm",
        "Number_of_Trials",
        "MSE_Mean",
        "MSE_Std",
        "SNR_Mean_dB",
        "SNR_Std_dB",
        "PRD_Mean_Percent",
        "PRD_Std"
    ])

    writer.writerows(summary_results)


# ==========================================================
# 13. GENERATE ROBUSTNESS COMPARISON GRAPHS
# ==========================================================

plot_metrics = [
    ("MSE", 4, 5, "Mean Squared Error"),
    ("SNR", 6, 7, "Signal-to-Noise Ratio (dB)"),
    ("PRD", 8, 9, "Percentage Root Mean Square Difference (%)")
]

for signal_name in signals:

    for metric_name, mean_col, std_col, ylabel in plot_metrics:

        plt.figure(figsize=(9, 5))

        for algorithm in ["OMP", "LASSO"]:

            rows = [
                row for row in summary_results
                if row[0] == signal_name
                and row[2] == algorithm
            ]

            means = [row[mean_col] for row in rows]

            stds = [row[std_col] for row in rows]

            plt.errorbar(
                MEASUREMENT_RATIOS,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                label=algorithm
            )

        plt.title(
            f"{signal_name}: Robustness {metric_name}"
        )

        plt.xlabel("Measurement Ratio (%)")
        plt.ylabel(ylabel)

        plt.xticks(MEASUREMENT_RATIOS)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"{signal_name}_Robustness_{metric_name}.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=200
        )

        plt.close()


# ==========================================================
# 14. FINAL REPORT
# ==========================================================

elapsed = time.time() - start_time

print("\n==========================================")
print("ROBUSTNESS TESTING COMPLETED!")
print("==========================================")

print("\nTotal reconstruction trials:", total_trials)

print("Total algorithm evaluations:", len(trial_results))

print(f"Execution time: {elapsed / 60:.2f} minutes")

print("\nGenerated files:")

print("1. Robustness_Individual_Trials.csv")
print("2. Robustness_Summary.csv")
print("3. ECG_Robustness_MSE.png")
print("4. ECG_Robustness_SNR.png")
print("5. ECG_Robustness_PRD.png")
print("6. PPG_Robustness_MSE.png")
print("7. PPG_Robustness_SNR.png")
print("8. PPG_Robustness_PRD.png")

print("\nSaved in:")
print(OUTPUT_DIR)
