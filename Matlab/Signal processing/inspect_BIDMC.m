
clc;
clear;
close all;

% Load BIDMC CSV
filePath = fullfile('Dataset', 'PPG', 'BIDMC', ...
    'bidmc_csv', 'bidmc_01_Signals.csv');

data = readtable(filePath);

% Display dataset information
disp('Dataset dimensions:');
disp(size(data));

disp('Column names:');
disp(data.Properties.VariableNames);

disp('First five rows:');
disp(head(data));

% Plot numerical columns
figure;
plot(data{:,1});
title('First Dataset Column');
xlabel('Sample Number');
ylabel('Amplitude');
grid on;
