
clc;
clear;
close all;

%% 1. LOAD DATA AND MEASUREMENT MATRICES

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

%% 2. CONVERT DATA TYPES

test_ecg = double(test_ecg);
test_ppg = double(test_ppg);

fs = double(fs);
N = double(window_length);

%% 3. PARAMETERS

sampling_ratios = [0.10 0.20 0.30 0.40 0.50];

num_ratios = length(sampling_ratios);

% Sparsity level for OMP
K = 40;

t = (0:N-1)' / fs;

% First test window
x_ecg = test_ecg(1,:)';
x_ppg = test_ppg(1,:)';

%% 4. INITIALIZE RESULT ARRAYS

ECG_MSE = zeros(num_ratios,1);
ECG_SNR = zeros(num_ratios,1);
ECG_PRD = zeros(num_ratios,1);

PPG_MSE = zeros(num_ratios,1);
PPG_SNR = zeros(num_ratios,1);
PPG_PRD = zeros(num_ratios,1);

%% 5. ECG OMP RECONSTRUCTION

figure('Name','ECG OMP Reconstruction','Color','w');

for i = 1:num_ratios

    Phi_current = double(Phi{i});

    % Compressed measurements
    y = Phi_current * x_ecg;

    % OMP reconstruction
    x_reconstructed = omp_solver(Phi_current,y,K);

    % Calculate error
    error_signal = x_ecg - x_reconstructed;

    % Performance metrics
    ECG_MSE(i) = mean(error_signal.^2);

    ECG_SNR(i) = 10*log10( ...
        sum(x_ecg.^2) / max(sum(error_signal.^2),eps));

    ECG_PRD(i) = 100*sqrt( ...
        sum(error_signal.^2) / max(sum(x_ecg.^2),eps));

    % Plot
    subplot(3,2,i);

    plot(t,x_ecg,'b','LineWidth',1.2);
    hold on;

    plot(t,x_reconstructed,'r--','LineWidth',1.1);

    title(sprintf('ECG OMP - %.0f%% Sampling', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','OMP Reconstructed');
    grid on;

end

%% 6. PPG OMP RECONSTRUCTION

figure('Name','PPG OMP Reconstruction','Color','w');

for i = 1:num_ratios

    Phi_current = double(Phi{i});

    % Compressed measurements
    y = Phi_current * x_ppg;

    % OMP reconstruction
    x_reconstructed = omp_solver(Phi_current,y,K);

    % Calculate error
    error_signal = x_ppg - x_reconstructed;

    % Performance metrics
    PPG_MSE(i) = mean(error_signal.^2);

    PPG_SNR(i) = 10*log10( ...
        sum(x_ppg.^2) / max(sum(error_signal.^2),eps));

    PPG_PRD(i) = 100*sqrt( ...
        sum(error_signal.^2) / max(sum(x_ppg.^2),eps));

    % Plot
    subplot(3,2,i);

    plot(t,x_ppg,'b','LineWidth',1.2);
    hold on;

    plot(t,x_reconstructed,'r--','LineWidth',1.1);

    title(sprintf('PPG OMP - %.0f%% Sampling', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','OMP Reconstructed');
    grid on;

end

%% 7. DISPLAY RESULTS

fprintf('\n============================================\n');
fprintf('             OMP RESULTS\n');
fprintf('============================================\n');

fprintf('\nECG RESULTS\n');
fprintf('Ratio\tMSE\t\tSNR (dB)\tPRD (%%)\n');

for i = 1:num_ratios

    fprintf('%.0f%%\t%.6f\t%.4f\t\t%.4f\n', ...
        sampling_ratios(i)*100, ...
        ECG_MSE(i),ECG_SNR(i),ECG_PRD(i));

end

fprintf('\nPPG RESULTS\n');
fprintf('Ratio\tMSE\t\tSNR (dB)\tPRD (%%)\n');

for i = 1:num_ratios

    fprintf('%.0f%%\t%.6f\t%.4f\t\t%.4f\n', ...
        sampling_ratios(i)*100, ...
        PPG_MSE(i),PPG_SNR(i),PPG_PRD(i));

end

%% 8. SAVE RESULTS

ECG_Results = table( ...
    sampling_ratios'*100,ECG_MSE,ECG_SNR,ECG_PRD, ...
    'VariableNames', ...
    {'Sampling_Ratio_Percent','MSE','SNR_dB','PRD_Percent'});

PPG_Results = table( ...
    sampling_ratios'*100,PPG_MSE,PPG_SNR,PPG_PRD, ...
    'VariableNames', ...
    {'Sampling_Ratio_Percent','MSE','SNR_dB','PRD_Percent'});

if ~exist('results','dir')
    mkdir('results');
end

writetable(ECG_Results,'results/ECG_OMP.csv');
writetable(PPG_Results,'results/PPG_OMP.csv');

save('results/OMP_Results.mat', ...
    'ECG_Results','PPG_Results','K');

fprintf('\n============================================\n');
disp('OMP RECONSTRUCTION COMPLETED!');
disp('Results saved in results folder.');
fprintf('============================================\n');

%% 9. OMP FUNCTION

function x_hat = omp_solver(A,y,K)

    % Dimensions
    [M,N] = size(A);

    % Normalize dictionary columns
    column_norms = sqrt(sum(A.^2,1));
    column_norms(column_norms < eps) = 1;

    D = A ./ column_norms;

    % Initialize
    residual = y;
    selected_atoms = [];
    coefficients = [];

    K = min([K,M,N]);

    % OMP iterations
    for iteration = 1:K

        % Correlation with residual
        correlations = abs(D' * residual);

        % Avoid selecting an atom twice
        correlations(selected_atoms) = 0;

        % Select most correlated atom
        [~,index] = max(correlations);

        selected_atoms = [selected_atoms,index];

        % Selected dictionary
        D_selected = D(:,selected_atoms);

        % Least squares coefficient estimation
        coefficients = D_selected \ y;

        % Update residual
        residual = y - D_selected * coefficients;

        % Stop if residual is sufficiently small
        if norm(residual) < 1e-8
            break;
        end

    end

    % Reconstruct original signal
    sparse_coefficients = zeros(N,1);

    sparse_coefficients(selected_atoms) = ...
        coefficients ./ column_norms(selected_atoms)';

    x_hat = sparse_coefficients;

end
