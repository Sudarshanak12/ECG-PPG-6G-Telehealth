
clc;
clear;
close all;

% Locate project folder
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% Load exported data
filePath = fullfile(projectRoot, ...
    'Results', 'ECG_PPG_Data.mat');

data = load(filePath);

% Extract signals
ecg = data.ecg(:);
ppg = data.ppg(:);
time = data.time(:);
fs = double(data.fs);

% Display information
fprintf('\nECG AND PPG DATA LOADED\n');
fprintf('ECG samples: %d\n', length(ecg));
fprintf('PPG samples: %d\n', length(ppg));
fprintf('Sampling frequency: %.2f Hz\n', fs);

% Plot first 10 seconds
N = min(10*fs, length(ecg));

figure;

subplot(2,1,1);
plot(time(1:N), ecg(1:N));
title('ECG Lead II');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(time(1:N), ppg(1:N));
title('PPG PLETH');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

sgtitle('Original ECG and PPG Signals');
