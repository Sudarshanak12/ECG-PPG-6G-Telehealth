
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

%% SELECT FIRST 10 SECONDS

duration = 10;
N = min(round(duration*fs),length(ecg));

ecg = ecg(1:N);
ppg = ppg(1:N);
time = time(1:N);

%% SETTINGS

B = 4;
L = 2^B;

%% SIGNALS

signals = {ecg, ppg};
signal_names = {'ECG Lead II','PPG PLETH'};

results = zeros(2,4);

for s = 1:2

    x = signals{s};

    %% PCM ENCODING

    xmin = min(x);
    xmax = max(x);

    delta_pcm = (xmax-xmin)/L;

    pcm_index = floor((x-xmin)/delta_pcm);
    pcm_index = min(max(pcm_index,0),L-1);

    pcm_reconstructed = ...
        xmin+(pcm_index+0.5)*delta_pcm;

    %% DPCM ENCODING

    predictor_residual = [0; diff(x)];

    residual_range = max(abs(predictor_residual));

    delta_dpcm = 2*residual_range/L;

    dpcm_reconstructed = zeros(N,1);
    dpcm_index = zeros(N,1);

    previous = x(1);

    for n = 1:N

        predicted = previous;

        error_signal = x(n)-predicted;

        q_index = floor(error_signal/delta_dpcm ...
            + L/2);

        q_index = min(max(q_index,0),L-1);

        quantized_error = ...
            (q_index-L/2+0.5)*delta_dpcm;

        dpcm_reconstructed(n) = ...
            predicted+quantized_error;

        dpcm_index(n) = q_index;

        previous = dpcm_reconstructed(n);

    end

    %% PERFORMANCE METRICS

    pcm_mse = mean((x-pcm_reconstructed).^2);

    dpcm_mse = mean((x-dpcm_reconstructed).^2);

    signal_variance = var(x);

    pcm_sqnr = 10*log10(signal_variance/pcm_mse);

    dpcm_sqnr = 10*log10(signal_variance/dpcm_mse);

    pcm_rate = fs*B;

    dpcm_rate = fs*B;

    %% STORE RESULTS

    results(s,:) = [pcm_mse,dpcm_mse, ...
        pcm_sqnr,dpcm_sqnr];

    %% PLOT

    figure('Name',signal_names{s});

    subplot(2,1,1);

    plot(time,x,'b','LineWidth',1.2);
    hold on;

    plot(time,pcm_reconstructed,'r--');

    title([signal_names{s},' - PCM Reconstruction']);

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','PCM');
    grid on;

    subplot(2,1,2);

    plot(time,x,'b','LineWidth',1.2);
    hold on;

    plot(time,dpcm_reconstructed,'m--');

    title([signal_names{s},' - DPCM Reconstruction']);

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original','DPCM');
    grid on;

    %% PRINT RESULTS

    fprintf('\n%s RESULTS\n',signal_names{s});

    fprintf('PCM MSE: %.8f\n',pcm_mse);
    fprintf('DPCM MSE: %.8f\n',dpcm_mse);

    fprintf('PCM SQNR: %.4f dB\n',pcm_sqnr);
    fprintf('DPCM SQNR: %.4f dB\n',dpcm_sqnr);

    fprintf('PCM Bit Rate: %.2f bps\n',pcm_rate);
    fprintf('DPCM Bit Rate: %.2f bps\n',dpcm_rate);

end

%% SAVE COMPARISON TABLE

comparison = table( ...
    signal_names', ...
    results(:,1), ...
    results(:,2), ...
    results(:,3), ...
    results(:,4), ...
    'VariableNames', ...
    {'Signal','PCM_MSE','DPCM_MSE', ...
    'PCM_SQNR_dB','DPCM_SQNR_dB'});

resultsFolder = fullfile(projectRoot,'Results');

writetable(comparison, ...
    fullfile(resultsFolder,'PCM_DPCM_Comparison.csv'));

disp('PCM VS DPCM COMPARISON COMPLETED!');
disp(comparison);
