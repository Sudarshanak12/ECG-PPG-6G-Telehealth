
clc;
clear;
close all;

%% 1. LOAD DATA

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');

%% 2. PARAMETERS

bit_depths = [8 10 12];
num_bits = length(bit_depths);

fs = double(fs);
N = double(window_length);

signals = {'ECG','PPG'};

train_data = {double(train_ecg), double(train_ppg)};
test_data  = {double(test_ecg), double(test_ppg)};

%% 3. INITIALIZE RESULTS

results = [];

%% 4. PCM PROCESSING

for s = 1:2

    fprintf('\nProcessing %s...\n',signals{s});

    Xtrain = train_data{s};
    Xtest = test_data{s};

    % Training calibration limits
    xmin = min(Xtrain(:));
    xmax = max(Xtrain(:));

    for b = 1:num_bits

        bits = bit_depths(b);
        levels = 2^bits;

        delta = (xmax-xmin)/levels;

        total_mse = 0;
        total_snr = 0;
        total_prd = 0;
        clipped_samples = 0;

        total_windows = size(Xtest,1);

        for w = 1:total_windows

            x = Xtest(w,:)';

            % Quantization index
            index = floor((x-xmin)/delta);

            % Count clipping
            clipped_samples = clipped_samples + ...
                sum(index < 0 | index >= levels);

            index = min(max(index,0),levels-1);

            % Mid-rise quantization
            x_quantized = xmin + (index+0.5)*delta;

            % Reconstruction error
            error_signal = x-x_quantized;

            mse = mean(error_signal.^2);

            snr = 10*log10( ...
                sum(x.^2)/max(sum(error_signal.^2),eps));

            prd = 100*sqrt( ...
                sum(error_signal.^2)/max(sum(x.^2),eps));

            total_mse = total_mse + mse;
            total_snr = total_snr + snr;
            total_prd = total_prd + prd;

            % Save example reconstruction
            if w == 1
                example_original = x;
                example_reconstructed = x_quantized;
            end

        end

        % Average window metrics
        avg_mse = total_mse/total_windows;
        avg_snr = total_snr/total_windows;
        avg_prd = total_prd/total_windows;

        % Nominal bit rate
        bitrate = fs*bits;

        % Compression ratio against 16-bit PCM
        compression_ratio = 16/bits;

        results = [results; ...
            {signals{s},bits,avg_mse,avg_snr,avg_prd, ...
            clipped_samples,delta,bitrate,compression_ratio}];

        fprintf('\n%s - %d bits\n',signals{s},bits);
        fprintf('MSE: %.8f\n',avg_mse);
        fprintf('SNR: %.4f dB\n',avg_snr);
        fprintf('PRD: %.4f %%\n',avg_prd);
        fprintf('Clipped samples: %d\n',clipped_samples);

        % Plot example
        figure('Color','w');

        t = (0:N-1)'/fs;

        plot(t,example_original,'b','LineWidth',1.2);
        hold on;

        plot(t,example_reconstructed,'r--','LineWidth',1.1);

        title(sprintf('%s PCM - %d bits',signals{s},bits));
        xlabel('Time (seconds)');
        ylabel('Amplitude');

        legend('Original','PCM reconstructed');
        grid on;

    end
end

%% 5. CREATE RESULTS TABLE

PCM_Results = cell2table(results, ...
    'VariableNames', ...
    {'Signal','Bit_Depth','MSE','SNR_dB','PRD_Percent', ...
    'Clipped_Samples','Quantization_Step', ...
    'Bit_Rate_bps','Compression_Ratio'});

%% 6. SAVE RESULTS

if ~exist('results','dir')
    mkdir('results');
end

writetable(PCM_Results,'results/PCM_Results.csv');

save('results/PCM_Results.mat','PCM_Results');

fprintf('\n====================================\n');
disp(PCM_Results);
disp('MATLAB PCM COMPLETED SUCCESSFULLY!');
fprintf('====================================\n');
