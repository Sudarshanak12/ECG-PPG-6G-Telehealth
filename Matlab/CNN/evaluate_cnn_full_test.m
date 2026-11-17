
clc;
clear;
close all;

%% 1. PROJECT SETUP

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

if ~exist('results','dir')
    mkdir('results');
end

%% 2. PARAMETERS

sampling_ratios = [10 20 30 40 50];
signals = {'ECG','PPG'};

test_data = {double(test_ecg), double(test_ppg)};

num_ratios = length(sampling_ratios);
num_windows = size(test_ecg,1);

fprintf('\n============================================\n');
fprintf(' FULL-TEST MATLAB CNN EVALUATION\n');
fprintf('============================================\n');

fprintf('ECG test windows: %d\n', size(test_ecg,1));
fprintf('PPG test windows: %d\n', size(test_ppg,1));

%% 3. RESULT STORAGE

total_rows = 2 * num_ratios * num_windows;

Signal = strings(total_rows,1);
Sampling_Ratio = zeros(total_rows,1);
Window = zeros(total_rows,1);
MSE = zeros(total_rows,1);
SNR_dB = zeros(total_rows,1);
PRD_percent = zeros(total_rows,1);

row_index = 0;

summary_signal = strings(10,1);
summary_ratio = zeros(10,1);
summary_mse = zeros(10,1);
summary_snr = zeros(10,1);
summary_prd = zeros(10,1);

summary_index = 0;

%% 4. EVALUATION LOOP

for s = 1:2

    signal_name = signals{s};
    XTest = test_data{s};

    fprintf('\n============================================\n');
    fprintf(' SIGNAL: %s\n', signal_name);
    fprintf('============================================\n');

    for r = 1:num_ratios

        ratio = sampling_ratios(r);

        fprintf('\nTesting %s CNN at %d%% sampling...\n', ...
            signal_name, ratio);

        %% Load measurement matrix

        A = double(Phi{r});

        %% Load trained CNN

        model_path = fullfile('models', ...
            sprintf('%s_CNN_%02dpct.mat',signal_name,ratio));

        if ~isfile(model_path)
            error('Missing model: %s', model_path);
        end

        model_data = load(model_path);

        if ~isfield(model_data,'net')
            error('Trained network variable "net" missing.');
        end

        net = model_data.net;

        %% Evaluate every test window

        mse_values = zeros(num_windows,1);
        snr_values = zeros(num_windows,1);
        prd_values = zeros(num_windows,1);

        for i = 1:num_windows

            % Original test signal
            x = XTest(i,:)';

            % Compressive measurements
            y = A*x;

            % Back Projection reconstruction
            x_bp = A'*y;

            % CNN input: same preprocessing as training
            cnn_input = {single(x_bp')};

            % CNN prediction
            prediction = predict(net,cnn_input);

            if iscell(prediction)
                x_reconstructed = double(prediction{1}(:));
            else
                x_reconstructed = double(prediction(:));
            end

            % Check output dimensions
            if length(x_reconstructed) ~= length(x)
                error('Output length mismatch at %s, %d%%, window %d', ...
                    signal_name, ratio, i);
            end

            % Reconstruction error
            error_signal = x - x_reconstructed;

            signal_energy = sum(x.^2);
            error_energy = sum(error_signal.^2);

            % Performance metrics
            mse_values(i) = mean(error_signal.^2);

            if error_energy == 0
                snr_values(i) = Inf;
                prd_values(i) = 0;
            else
                snr_values(i) = 10*log10( ...
                    signal_energy/error_energy);

                prd_values(i) = 100*sqrt( ...
                    error_energy/signal_energy);
            end

            % Store individual window results
            row_index = row_index + 1;

            Signal(row_index) = string(signal_name);
            Sampling_Ratio(row_index) = ratio;
            Window(row_index) = i;
            MSE(row_index) = mse_values(i);
            SNR_dB(row_index) = snr_values(i);
            PRD_percent(row_index) = prd_values(i);

        end

        %% Aggregate results for this model

        summary_index = summary_index + 1;

        summary_signal(summary_index) = string(signal_name);
        summary_ratio(summary_index) = ratio;
        summary_mse(summary_index) = mean(mse_values);
        summary_snr(summary_index) = mean(snr_values);
        summary_prd(summary_index) = mean(prd_values);

        fprintf('Completed: %s %d%%\n', signal_name, ratio);
        fprintf('Mean MSE: %.6f\n', mean(mse_values));
        fprintf('Mean SNR: %.4f dB\n', mean(snr_values));
        fprintf('Mean PRD: %.4f %%\n', mean(prd_values));

        clear net model_data;

    end
end

%% 5. CREATE RESULT TABLES

Window_Results = table( ...
    Signal, Sampling_Ratio, Window, MSE, SNR_dB, PRD_percent);

Summary = table( ...
    summary_signal, summary_ratio, summary_mse, ...
    summary_snr, summary_prd, ...
    'VariableNames', ...
    {'Signal','Sampling_Ratio','Mean_MSE','Mean_SNR_dB','Mean_PRD_percent'});

%% 6. SAVE RESULTS

writetable(Window_Results, ...
    'results/MATLAB_CNN_FullTest_WindowResults.csv');

writetable(Summary, ...
    'results/MATLAB_CNN_FullTest_Summary.csv');

save('results/MATLAB_CNN_FullTest_Results.mat', ...
    'Window_Results','Summary','sampling_ratios');

%% 7. GENERATE PERFORMANCE GRAPHS

metrics = {'Mean_MSE','Mean_SNR_dB','Mean_PRD_percent'};

metric_titles = {'MSE','SNR (dB)','PRD (%)'};

for s = 1:2

    signal_name = signals{s};

    signal_rows = Summary.Signal == string(signal_name);

    for m = 1:3

        figure('Color','w');

        plot(Summary.Sampling_Ratio(signal_rows), ...
            Summary.(metrics{m})(signal_rows), ...
            '-o','LineWidth',2,'MarkerSize',7);

        xlabel('Sampling Ratio (%)');
        ylabel(metric_titles{m});

        title(sprintf('%s CNN - %s', ...
            signal_name,metric_titles{m}));

        grid on;
        xticks(sampling_ratios);

        filename = sprintf('results/%s_CNN_%s.png', ...
            signal_name,metrics{m});

        saveas(gcf,filename);
        close(gcf);

    end
end

%% 8. FINAL VALIDATION

fprintf('\n============================================\n');
fprintf(' FULL-TEST EVALUATION COMPLETED\n');
fprintf('============================================\n');

fprintf('Total reconstructions: %d\n',row_index);
fprintf('Expected reconstructions: %d\n',total_rows);

fprintf('NaN values: %d\n', ...
    sum(isnan(MSE)) + sum(isnan(SNR_dB)) + sum(isnan(PRD_percent)));

fprintf('Inf values: %d\n', ...
    sum(isinf(MSE)) + sum(isinf(SNR_dB)) + sum(isinf(PRD_percent)));

disp(Summary);

fprintf('\nResults saved in the results folder.\n');
