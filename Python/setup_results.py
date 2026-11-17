
from pathlib import Path

# Project root = parent of the Python folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Main Results directory
RESULTS_DIR = PROJECT_DIR / "Results"

# Folder structure
folders = [
    "01_Preprocessing",
    "02_Sampling_Theorem",
    "03_Quantization",
    "04_PCM_DPCM",
    "05_OMP_Reconstruction",
    "06_LASSO_Reconstruction",
    "07_Algorithm_Comparison",
    "08_Final_Results"
]

def create_results_structure():

    print("\nCreating project Results structure...\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for folder in folders:

        folder_path = RESULTS_DIR / folder

        folder_path.mkdir(exist_ok=True)

        print(f"[CREATED] {folder}")

    print("\n================================")
    print("RESULTS FOLDER SETUP COMPLETED!")
    print("================================")

    print(f"\nLocation: {RESULTS_DIR}")

    print("\nYour folder structure:")

    for folder in folders:
        print(f"Results/{folder}/")


if __name__ == "__main__":
    create_results_structure()
