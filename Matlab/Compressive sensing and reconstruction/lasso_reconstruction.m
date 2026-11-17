
clc;
clear;
close all;

%% 1. LOAD DATA

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

lambda = 0.01;
max_iterations = 2000;

t = (0:N-1)' / fs;

x_ecg = test_ecg(1,:)';
x_ppg = test_ppg(1,:)';

%% 4. INITIALIZE RESULTS

ECG_MSE = zeros(num_ratios,1);
ECG_SNR = zeros(num_ratios,1);
ECG_PRD = zeros(num_ratios,1);

PPG_MSE = zeros(num_ratios,1);
PPG_SNR = zeros(num_ratios,1);
PPG_PRD = zeros(num_ratios,1);

%% 5. ECG LASSO

figure('Name','ECG LASSO Reconstruction','Color','w');

for i = 1:num_ratios

    A = double(Phi{i});
    y = A * x_ecg;

    % LASSO reconstruction
    x_rec = lasso_solver(A,y,lambda,max_iterations);

    error_signal = x_ecg - x_rec;

    ECG_MSE(i) = mean(error_signal.^2);

    ECG_SNR(i) = 10*log10( ...
        sum(x_ecg.^2)/max(sum(error_signal.^2),eps));

    ECG_PRD(i) = 100*sqrt( ...
        sum(error_signal.^2)/max(sum(x_ecg.^2),eps));

    subplot(3,2,i);

    plot(t,x_ecg,'b','LineWidth',1.2);
    hold on;
    plot(t,x_rec,'r--','LineWidth',1.1);

    title(sprintf('ECG LASSO - %.0f%%', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');
    legend('Original','Reconstructed');
    grid on;

end

%% 6. PPG LASSO

figure('Name','PPG LASSO Reconstruction','Color','w');

for i = 1:num_ratios

    A = double(Phi{i});
    y = A * x_ppg;

    x_rec = lasso_solver(A,y,lambda,max_iterations);

    error_signal = x_ppg - x_rec;

    PPG_MSE(i) = mean(error_signal.^2);

    PPG_SNR(i) = 10*log10( ...
        sum(x_ppg.^2)/max(sum(error_signal.^2),eps));

    PPG_PRD(i) = 100*sqrt( ...
        sum(error_signal.^2)/max(sum(x_ppg.^2),eps));

    subplot(3,2,i);

    plot(t,x_ppg,'b','LineWidth',1.2);
    hold on;
    plot(t,x_rec,'r--','LineWidth',1.1);

    title(sprintf('PPG LASSO - %.0f%%', ...
        sampling_ratios(i)*100));

    xlabel('Time (seconds)');
    ylabel('Amplitude');
    legend('Original','Reconstructed');
    grid on;

end

%% 7. DISPLAY RESULTS

fprintf('\n============================================\n');
fprintf('          LASSO RESULTS\n');
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

writetable(ECG_Results,'results/ECG_LASSO.csv');
writetable(PPG_Results,'results/PPG_LASSO.csv');

save('results/LASSO_Results.mat', ...
    'ECG_Results','PPG_Results','lambda');

fprintf('\nLASSO RECONSTRUCTION COMPLETED!\n');

%% 9. LASSO SOLVER - PROXIMAL GRADIENT

function x_hat = lasso_solver(A,y,lambda,max_iter)

    % Normalize dictionary columns
    column_norms = sqrt(sum(A.^2,1));
    column_norms(column_norms < eps) = 1;

    D = A ./ column_norms;

    % Lipschitz constant
    L = norm(D,2)^2;

    % Initialize
    alpha = zeros(size(D,2),1);
    z = alpha;
    momentum = 1;

    for k = 1:max_iter

        gradient = D'*(D*z-y);

        % Soft thresholding
        alpha_new = soft_threshold( ...
            z-gradient/L,lambda/L);

        momentum_new = ...
            (1+sqrt(1+4*momentum^2))/2;

        z = alpha_new + ...
            ((momentum-1)/momentum_new)*(alpha_new-alpha);

        alpha = alpha_new;
        momentum = momentum_new;

    end

    % Recover signal
    x_hat = (alpha ./ column_norms') ;

end

%% 10. SOFT THRESHOLDING

function result = soft_threshold(x,threshold)

    result = sign(x).*max(abs(x)-threshold,0);

end
