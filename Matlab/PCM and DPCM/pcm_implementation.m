
clc;
clear;
close all;

%% LOAD ECG AND PPG

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

%% PCM SETTINGS

bits = [2 4 8];

ecg_mse = zeros(length(bits),1);
ppg_mse = zeros(length(bits),1);

%% ECG PCM ENCODING AND DECODING

figure('Name','ECG PCM Analysis');

for k = 1:length(bits)

    B = bits(k);
    L = 2^B;

    % Quantization range
    xmin = min(ecg);
    xmax = max(ecg);

    delta = (xmax-xmin)/L;

    % Quantization
    index = floor((ecg-xmin)/delta);
    index = min(max(index,0),L-1);

    % Binary PCM encoding
    binary_matrix = dec2bin(index,B)-'0';

    % Serial bitstream
    bitstream = reshape(binary_matrix.',1,[]);

    % PCM decoding
    decoded_index = bin2dec(char(binary_matrix+'0'));

    % Signal reconstruction
    reconstructed = xmin + (decoded_index+0.5)*delta;

    % Error calculation
    ecg_mse(k) = mean((ecg-reconstructed).^2);

    % Plot
    subplot(3,1,k);

    plot(time,ecg,'b','LineWidth',1.2);
    hold on;

    stairs(time,reconstructed,'r--','LineWidth',1.1);

    title(sprintf('ECG PCM - %d Bits',B));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original ECG','PCM Reconstructed');
    grid on;

    fprintf('\nECG PCM: %d bits\n',B);
    fprintf('Quantization levels: %d\n',L);
    fprintf('Bitstream length: %d bits\n',length(bitstream));
    fprintf('Bit rate: %.0f bits/second\n',fs*B);
    fprintf('MSE: %.8f\n',ecg_mse(k));

end

sgtitle('ECG Pulse Code Modulation');

%% PPG PCM ENCODING AND DECODING

figure('Name','PPG PCM Analysis');

for k = 1:length(bits)

    B = bits(k);
    L = 2^B;

    xmin = min(ppg);
    xmax = max(ppg);

    delta = (xmax-xmin)/L;

    index = floor((ppg-xmin)/delta);
    index = min(max(index,0),L-1);

    % Binary encoding
    binary_matrix = dec2bin(index,B)-'0';

    % Serial bitstream
    bitstream = reshape(binary_matrix.',1,[]);

    % Decoding
    decoded_index = bin2dec(char(binary_matrix+'0'));

    % Reconstruction
    reconstructed = xmin + (decoded_index+0.5)*delta;

    ppg_mse(k) = mean((ppg-reconstructed).^2);

    subplot(3,1,k);

    plot(time,ppg,'b','LineWidth',1.2);
    hold on;

    stairs(time,reconstructed,'r--','LineWidth',1.1);

    title(sprintf('PPG PCM - %d Bits',B));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original PPG','PCM Reconstructed');
    grid on;

    fprintf('\nPPG PCM: %d bits\n',B);
    fprintf('Quantization levels: %d\n',L);
    fprintf('Bitstream length: %d bits\n',length(bitstream));
    fprintf('Bit rate: %.0f bits/second\n',fs*B);
    fprintf('MSE: %.8f\n',ppg_mse(k));

end

sgtitle('PPG Pulse Code Modulation');

%% SAVE RESULTS

resultsFolder = fullfile(projectRoot,'Results');

resultsTable = table(bits',2.^bits', ...
    ecg_mse,ppg_mse, ...
    (fs*bits)', ...
    'VariableNames', ...
    {'Bits','Levels','ECG_MSE','PPG_MSE','BitRate'});

writetable(resultsTable, ...
    fullfile(resultsFolder,'PCM_Results.csv'));

disp('PCM ENCODING AND DECODING COMPLETED!');
