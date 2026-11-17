
clc;
clear;
close all;

%% 1. LOAD DATASET AND MEASUREMENT MATRICES

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

%% 2. CONVERT DATA TYPES

train_ecg = double(train_ecg);
val_ecg   = double(val_ecg);
test_ecg  = double(test_ecg);

train_ppg = double(train_ppg);
val_ppg   = double(val_ppg);
test_ppg  = double(test_ppg);

fs = double(fs);
window_length = double(window_length);

%% 3. PARAMETERS

N = window_length;

sampling_ratios = [0.10 0.20 0.30 0.40 0.50];

num_ratios = length(sampling_ratios);

t = (0:N-1)' / fs;

% Select first test window
x_ecg = test_ecg(1,:)';
x_ppg = test_ppg(1,:)';

%% 4. INITIALIZE RESULT STORAGE

ECG_MSE = zeros(num_ratios,1);
ECG_SNR = zeros(num_ratios,1);
ECG_PRD = zeros(num_ratios,1);

PPG_MSE = zeros(num_ratios,1);
PPG_SNR = zeros(num_ratios,1);
PPG_PRD = zeros(num_ratios,1);

%% 5. ECG BACK PROJECTION

figure('Name','ECG Back Projection','Color','w');

for i = 1:num_ratios

    % Measurement matrix
    measurement_matrix = double(Phi{i});

    % Compressed measurements
    y = measurement_matrix * x_ecg;

    % Back Projection reconstruction
    x_reconstructed = measurement_matrix' * y;

    % Performance metrics
    error_signal = x_ecg - x_reconstructed;

    ECG_MSE(i) = mean(error_signal.^2);

    ECG_SNR(i) = 10*log10( ...
        sum(x_ecg.^2) / sum(error_signal.^2));

    ECG_PRD(i) = 100 * sqrt( ...
        sum(error_signal.^2) / sum(x_ecg.^2));

    % Plot
    subplot(3,2,i);

    plot(t,x_ecg,'b','LineWidth',1.2);
    hold on;

    plot(t,x_reconstructed,'r--','LineWidth',1.1);

    title(sprintf('ECG - %.0f%% Sampling', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','Reconstructed');
    grid on;

end

%% 6. PPG BACK PROJECTION

figure('Name','PPG Back Projection','Color','w');

for i = 1:num_ratios

    measurement_matrix = double(Phi{i});

    % Compressed measurements
    y = measurement_matrix * x_ppg;

    % Back Projection reconstruction
    x_reconstructed = measurement_matrix' * y;

    % Performance metrics
    error_signal = x_ppg - x_reconstructed;

    PPG_MSE(i) = mean(error_signal.^2);

    PPG_SNR(i) = 10*log10( ...
        sum(x_ppg.^2) / sum(error_signal.^2));

    PPG_PRD(i) = 100 * sqrt( ...
        sum(error_signal.^2) / sum(x_ppg.^2));

    % Plot
    subplot(3,2,i);

    plot(t,x_ppg,'b','LineWidth',1.2);
    hold on;

    plot(t,x_reconstructed,'r--','LineWidth',1.1);

    title(sprintf('PPG - %.0f%% Sampling', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','Reconstructed');
    grid on;

end

%% 7. DISPLAY RESULTS

fprintf('\n============================================\n');
fprintf('       BACK PROJECTION RESULTS\n');
fprintf('============================================\n');

fprintf('\nECG RESULTS\n');

fprintf('Ratio\tMSE\t\tSNR (dB)\tPRD (%%)\n');

for i = 1:num_ratios

    fprintf('%.0f%%\t%.6f\t%.4f\t\t%.4f\n', ...
        sampling_ratios(i)*100, ...
        ECG_MSE(i), ECG_SNR(i), ECG_PRD(i));

end

fprintf('\nPPG RESULTS\n');

fprintf('Ratio\tMSE\t\tSNR (dB)\tPRD (%%)\n');

for i = 1:num_ratios

    fprintf('%.0f%%\t%.6f\t%.4f\t\t%.4f\n', ...
        sampling_ratios(i)*100, ...
        PPG_MSE(i), PPG_SNR(i), PPG_PRD(i));

end

%% 8. SAVE RESULTS

ECG_Results = table( ...
    sampling_ratios'*100, ECG_MSE, ECG_SNR, ECG_PRD, ...
    'VariableNames', ...
    {'Sampling_Ratio_Percent','MSE','SNR_dB','PRD_Percent'});

PPG_Results = table( ...
    sampling_ratios'*100, PPG_MSE, PPG_SNR, PPG_PRD, ...
    'VariableNames', ...
    {'Sampling_Ratio_Percent','MSE','SNR_dB','PRD_Percent'});

if ~exist('results','dir')
    mkdir('results');
end

writetable(ECG_Results,'results/ECG_BackProjection.csv');
writetable(PPG_Results,'results/PPG_BackProjection.csv');

save('results/BackProjection_Results.mat', ...
    'ECG_Results','PPG_Results');

fprintf('\n============================================\n');
disp('BACK PROJECTION COMPLETED SUCCESSFULLY!');
disp('Results saved in the results folder.');
fprintf('============================================\n');
