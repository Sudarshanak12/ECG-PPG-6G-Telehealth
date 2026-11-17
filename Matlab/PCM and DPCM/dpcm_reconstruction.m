
clc;
clear;
close all;

%% 1. LOAD DATA

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');

%% 2. PARAMETERS

bit_depths = [8 10 12];

fs = double(fs);
N = double(window_length);

signals = {'ECG','PPG'};

train_data = {double(train_ecg), double(train_ppg)};
test_data  = {double(test_ecg), double(test_ppg)};

results = [];

%% 3. DPCM PROCESSING

for s = 1:2

    fprintf('\nProcessing %s DPCM...\n',signals{s});

    Xtrain = train_data{s};
    Xtest = test_data{s};

    % Calculate training first differences
    differences = diff(Xtrain,1,2);

    % Expanded symmetric quantization range
    max_difference = max(abs(differences(:)));

    error_range = 2 * max_difference;

    fprintf('Quantization range: [%.6f, %.6f]\n', ...
        -error_range,error_range);

    for b = 1:length(bit_depths)

        bits = bit_depths(b);
        levels = 2^bits;

        delta = (2*error_range)/levels;

        total_mse = 0;
        total_snr = 0;
        total_prd = 0;
        clipped_samples = 0;

        total_windows = size(Xtest,1);

        for w = 1:total_windows

            x = Xtest(w,:)';

            % Reset predictor for every window
            previous_reconstructed = 0;

            x_reconstructed = zeros(N,1);

            for n = 1:N

                % First-order prediction
                prediction = previous_reconstructed;

                % Prediction error
                error_value = x(n)-prediction;

                % Quantization index
                index = floor( ...
                    (error_value+error_range)/delta);

                % Clipping count
                if index < 0 || index >= levels
                    clipped_samples = clipped_samples+1;
                end

                index = min(max(index,0),levels-1);

                % Mid-rise quantization
                quantized_error = ...
                    -error_range+(index+0.5)*delta;

                % Reconstruct
                reconstructed = ...
                    prediction+quantized_error;

                x_reconstructed(n) = reconstructed;

                % Feedback predictor
                previous_reconstructed = reconstructed;

            end

            % Performance metrics
            error_signal = x-x_reconstructed;

            mse = mean(error_signal.^2);

            snr = 10*log10( ...
                sum(x.^2)/max(sum(error_signal.^2),eps));

            prd = 100*sqrt( ...
                sum(error_signal.^2)/max(sum(x.^2),eps));

            total_mse = total_mse+mse;
            total_snr = total_snr+snr;
            total_prd = total_prd+prd;

            % Store first test example
            if w == 1
                example_original = x;
                example_reconstructed = x_reconstructed;
            end

        end

        % Average metrics
        avg_mse = total_mse/total_windows;
        avg_snr = total_snr/total_windows;
        avg_prd = total_prd/total_windows;

        bitrate = fs*bits;
        compression_ratio = 16/bits;

        results = [results; ...
            {signals{s},bits,avg_mse,avg_snr,avg_prd, ...
            clipped_samples,delta,bitrate,compression_ratio}];

        fprintf('\n%s - %d bits\n',signals{s},bits);
        fprintf('MSE: %.8f\n',avg_mse);
        fprintf('SNR: %.4f dB\n',avg_snr);
        fprintf('PRD: %.4f %%\n',avg_prd);
        fprintf('Clipped samples: %d\n',clipped_samples);

        % Plot first test window
        t = (0:N-1)'/fs;

        figure('Color','w');

        plot(t,example_original,'b','LineWidth',1.2);
        hold on;

        plot(t,example_reconstructed,'r--','LineWidth',1.1);

        title(sprintf('%s DPCM - %d bits', ...
            signals{s},bits));

        xlabel('Time (seconds)');
        ylabel('Amplitude');

        legend('Original','DPCM reconstructed');
        grid on;

    end
end

%% 4. CREATE RESULTS TABLE

DPCM_Results = cell2table(results, ...
    'VariableNames', ...
    {'Signal','Bit_Depth','MSE','SNR_dB','PRD_Percent', ...
    'Clipped_Samples','Quantization_Step', ...
    'Bit_Rate_bps','Compression_Ratio'});

%% 5. SAVE RESULTS

if ~exist('results','dir')
    mkdir('results');
end

writetable(DPCM_Results,'results/DPCM_Results.csv');

save('results/DPCM_Results.mat','DPCM_Results');

fprintf('\n====================================\n');
disp(DPCM_Results);
disp('MATLAB DPCM COMPLETED SUCCESSFULLY!');
fprintf('====================================\n');
