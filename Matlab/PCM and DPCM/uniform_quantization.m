
clc;
clear;
close all;

%% LOAD ECG AND PPG DATA

projectRoot = fileparts(fileparts(mfilename('fullpath')));

filePath = fullfile(projectRoot, ...
    'Results', 'ECG_PPG_Data.mat');

data = load(filePath);

ecg = double(data.ecg(:));
ppg = double(data.ppg(:));
time = double(data.time(:));
fs = double(data.fs);

%% SELECT FIRST 2 SECONDS

duration = 2;

N = min(round(duration * fs), length(ecg));

ecg = ecg(1:N);
ppg = ppg(1:N);
time = time(1:N);

%% QUANTIZATION SETTINGS

bits = [2 4 8 16];

%% ECG QUANTIZATION

ecg_min = min(ecg);
ecg_max = max(ecg);

ecg_mse = zeros(length(bits),1);
ecg_sqnr = zeros(length(bits),1);

figure('Name','ECG Uniform Quantization');

for k = 1:length(bits)

    B = bits(k);

    L = 2^B;

    % Calculate quantization step
    delta = (ecg_max - ecg_min)/L;

    % Quantization index
    index = floor((ecg - ecg_min)/delta);

    % Prevent index overflow
    index = min(index,L-1);

    % Reconstructed quantized signal
    ecg_quantized = ecg_min + (index + 0.5)*delta;

    % Calculate error
    error = ecg - ecg_quantized;

    ecg_mse(k) = mean(error.^2);

    ecg_sqnr(k) = 10*log10(var(ecg)/ecg_mse(k));

    % Plot
    subplot(2,2,k);

    plot(time,ecg,'b','LineWidth',1.2);
    hold on;

    stairs(time,ecg_quantized,'r','LineWidth',1.1);

    title(sprintf('%d-bit Quantization (%d Levels)',B,L));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original ECG','Quantized ECG');
    grid on;

end

sgtitle('ECG Uniform Quantization Analysis');

%% DISPLAY ECG RESULTS

fprintf('\nECG QUANTIZATION RESULTS\n');
fprintf('-------------------------------------\n');

for k = 1:length(bits)

    fprintf('%2d-bit | Levels: %5d | MSE: %.8f | SQNR: %.3f dB\n', ...
        bits(k),2^bits(k),ecg_mse(k),ecg_sqnr(k));

end

%% PPG QUANTIZATION

ppg_min = min(ppg);
ppg_max = max(ppg);

ppg_mse = zeros(length(bits),1);

figure('Name','PPG Uniform Quantization');

for k = 1:length(bits)

    B = bits(k);
    L = 2^B;

    delta = (ppg_max - ppg_min)/L;

    index = floor((ppg - ppg_min)/delta);
    index = min(index,L-1);

    ppg_quantized = ppg_min + (index + 0.5)*delta;

    ppg_mse(k) = mean((ppg - ppg_quantized).^2);

    subplot(2,2,k);

    plot(time,ppg,'b','LineWidth',1.2);
    hold on;

    stairs(time,ppg_quantized,'r','LineWidth',1.1);

    title(sprintf('%d-bit Quantization',B));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original PPG','Quantized PPG');
    grid on;

end

sgtitle('PPG Uniform Quantization Analysis');

%% SAVE RESULTS

resultsFolder = fullfile(projectRoot,'Results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

resultsTable = table(bits',2.^bits',ecg_mse,ecg_sqnr,ppg_mse, ...
    'VariableNames',{'Bits','Levels','ECG_MSE','ECG_SQNR_dB','PPG_MSE'});

writetable(resultsTable, ...
    fullfile(resultsFolder,'Quantization_Results.csv'));

save(fullfile(resultsFolder,'Quantization_Data.mat'), ...
    'bits','ecg_mse','ecg_sqnr','ppg_mse');

disp('QUANTIZATION EXPERIMENT COMPLETED!');
