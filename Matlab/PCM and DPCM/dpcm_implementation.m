
clc;
clear;
close all;

%% LOAD DATA

projectRoot = fileparts(fileparts(mfilename('fullpath')));

filePath = fullfile(projectRoot, ...
    'Results', 'ECG_PPG_Data.mat');

data = load(filePath);

ecg = double(data.ecg(:));
ppg = double(data.ppg(:));
fs = double(data.fs);
time = double(data.time(:));

%% SELECT FIRST 2 SECONDS

duration = 2;
N = min(round(duration*fs),length(ecg));

ecg = ecg(1:N);
ppg = ppg(1:N);
time = time(1:N);

%% DPCM SETTINGS

bits = 4;
L = 2^bits;

%% ECG DPCM

ecg_min = min(ecg);
ecg_max = max(ecg);

delta = (ecg_max-ecg_min)/(L-1);

ecg_reconstructed = zeros(N,1);
ecg_error = zeros(N,1);
ecg_encoded = zeros(N,1);

% Initial predictor
previous = ecg(1);

for n = 1:N

    % Prediction
    predicted = previous;

    % Prediction error
    error_signal = ecg(n)-predicted;

    % Quantize prediction error
    q_index = round(error_signal/delta);

    % Limit quantizer index
    q_index = max(-L/2,min(L/2-1,q_index));

    % Reconstruct error
    quantized_error = q_index*delta;

    % Reconstruct signal
    ecg_reconstructed(n) = predicted+quantized_error;

    % Store error and encoded index
    ecg_error(n) = error_signal;
    ecg_encoded(n) = q_index;

    % Update predictor
    previous = ecg_reconstructed(n);

end

ecg_mse = mean((ecg-ecg_reconstructed).^2);

%% PLOT ECG

figure('Name','ECG DPCM');

subplot(2,1,1);

plot(time,ecg,'b','LineWidth',1.2);
hold on;

plot(time,ecg_reconstructed,'r--','LineWidth',1.2);

title('ECG DPCM Reconstruction');
xlabel('Time (seconds)');
ylabel('Amplitude');

legend('Original ECG','DPCM Reconstructed');
grid on;

subplot(2,1,2);

plot(time,ecg_error,'k');

title('ECG Prediction Error');
xlabel('Time (seconds)');
ylabel('Prediction Error');

grid on;

%% PPG DPCM

ppg_min = min(ppg);
ppg_max = max(ppg);

delta = (ppg_max-ppg_min)/(L-1);

ppg_reconstructed = zeros(N,1);
ppg_error = zeros(N,1);
ppg_encoded = zeros(N,1);

previous = ppg(1);

for n = 1:N

    predicted = previous;

    error_signal = ppg(n)-predicted;

    q_index = round(error_signal/delta);

    q_index = max(-L/2,min(L/2-1,q_index));

    quantized_error = q_index*delta;

    ppg_reconstructed(n) = predicted+quantized_error;

    ppg_error(n) = error_signal;
    ppg_encoded(n) = q_index;

    previous = ppg_reconstructed(n);

end

ppg_mse = mean((ppg-ppg_reconstructed).^2);

%% PLOT PPG

figure('Name','PPG DPCM');

subplot(2,1,1);

plot(time,ppg,'b','LineWidth',1.2);
hold on;

plot(time,ppg_reconstructed,'r--','LineWidth',1.2);

title('PPG DPCM Reconstruction');
xlabel('Time (seconds)');
ylabel('Amplitude');

legend('Original PPG','DPCM Reconstructed');
grid on;

subplot(2,1,2);

plot(time,ppg_error,'k');

title('PPG Prediction Error');
xlabel('Time (seconds)');
ylabel('Prediction Error');

grid on;

%% DISPLAY RESULTS

fprintf('\nDPCM RESULTS\n');
fprintf('---------------------------\n');

fprintf('Sampling Frequency: %.2f Hz\n',fs);
fprintf('Quantization Bits: %d\n',bits);

fprintf('\nECG MSE: %.8f\n',ecg_mse);
fprintf('PPG MSE: %.8f\n',ppg_mse);

fprintf('\nECG Bit Rate: %.2f bps\n',fs*bits);
fprintf('PPG Bit Rate: %.2f bps\n',fs*bits);

%% SAVE RESULTS

resultsFolder = fullfile(projectRoot,'Results');

save(fullfile(resultsFolder,'DPCM_Results.mat'), ...
    'ecg_reconstructed','ppg_reconstructed', ...
    'ecg_error','ppg_error','ecg_mse','ppg_mse', ...
    'ecg_encoded','ppg_encoded','bits','fs');

disp('DPCM EXPERIMENT COMPLETED!');
