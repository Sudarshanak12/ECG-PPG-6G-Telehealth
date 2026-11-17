clc;
clear;
close all;

%% 1. Set project directory
project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';

cd(project_path);

%% 2. Load dataset
load('ECG_PPG_Data.mat');

%% 3. Verify dataset dimensions
fprintf('\n========== DATASET VERIFICATION ==========\n');

fprintf('Sampling Frequency: %d Hz\n', fs);
fprintf('Window Length: %d samples\n', window_length);
fprintf('Window Stride: %d samples\n', window_stride);

fprintf('\nECG Training: %d x %d\n', size(train_ecg));
fprintf('ECG Validation: %d x %d\n', size(val_ecg));
fprintf('ECG Testing: %d x %d\n', size(test_ecg));

fprintf('\nPPG Training: %d x %d\n', size(train_ppg));
fprintf('PPG Validation: %d x %d\n', size(val_ppg));
fprintf('PPG Testing: %d x %d\n', size(test_ppg));

%% 4. Create time axis
t = (0:window_length-1)/fs;

%% 5. Plot ECG test signal
figure;
plot(t, test_ecg(1,:), 'b', 'LineWidth', 1.2);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original ECG Test Signal');
grid on;

%% 6. Plot PPG test signal
figure;
plot(t, test_ppg(1,:), 'r', 'LineWidth', 1.2);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original PPG Test Signal');
grid on;

%% 7. Create output folders
if ~exist('results', 'dir')
    mkdir('results');
end

if ~exist('functions', 'dir')
    mkdir('functions');
end

fprintf('\n==========================================\n');
disp('DATASET LOADED AND VERIFIED SUCCESSFULLY!');
disp('Project directory initialized.');