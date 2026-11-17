# ECG-PPG-6G-Telehealth



# Dataset Information

## Dataset Used
This project uses the BIDMC (Beth Israel Deaconess Medical Center) physiological signal dataset.

## Signals
- ECG: Lead II
- PPG: PLETH

## Sampling Information
- Sampling frequency: 125 Hz
- Window length: 512 samples
- Window duration: 4.096 seconds
- Window stride: 256 samples
- Overlap: 50%

## Dataset Split

| Split | Windows per signal |
|---|---:|
| Training | 8,621 |
| Validation | 1,631 |
| Testing | 2,097 |

## Preprocessing
The ECG and PPG signals are segmented into fixed-length windows. The prepared data is used for compressive sensing, reconstruction, CNN training and testing.

## Files
- `ECG_PPG_Data.mat`: Prepared ECG and PPG data arrays.
- `Measurement_Matrices.mat`: Measurement matrices used for compressive sensing.

## Data Source
BIDMC physiological signal dataset. Refer to the original dataset provider for access, licensing and citation information.

**Note:** The dataset itself should be obtained from its authorised source. Do not redistribute files unless the applicable licence permits it.
