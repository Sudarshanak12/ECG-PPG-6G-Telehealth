
clc;
clear;
close all;

%% =====================================================
% COMMUNICATION BANDWIDTH AND DATA RATE ANALYSIS
% ECG + PPG Wearable Telehealth System
% ======================================================

%% 1. PROJECT PARAMETERS

fs = 125;                  % Sampling frequency (Hz)
N = 512;                   % Samples per window
original_bits = 16;        % Reference PCM bit depth

bit_depths = [8 10 12];
sampling_ratios = [10 20 30 40 50];

signals = {'ECG','PPG'};
methods = {'PCM','DPCM','CS'};

%% 2. INITIALIZE RESULTS

results = {};

%% 3. PCM AND DPCM DATA RATE

for s = 1:length(signals)

    for m = 1:2

        for b = 1:length(bit_depths)

            bits = bit_depths(b);

            bitrate = fs * bits;

            compression_ratio = original_bits / bits;

            bits_per_window = N * bits;

            results = [results; ...
                {signals{s}, methods{m}, bits, NaN, ...
                bits_per_window, bitrate, ...
                compression_ratio}];

        end
    end
end

%% 4. COMPRESSIVE SENSING DATA RATE

for s = 1:length(signals)

    for r = 1:length(sampling_ratios)

        ratio = sampling_ratios(r);

        M = round(N * ratio / 100);

        % Number of transmitted bits per measurement
        measurement_bits = 16;

        bits_per_window = M * measurement_bits;

        % Effective rate based on measurements per window
        bitrate = (M * measurement_bits * fs) / N;

        compression_ratio = ...
            (N * original_bits) / bits_per_window;

        results = [results; ...
            {signals{s}, 'CS', measurement_bits, ratio, ...
            bits_per_window, bitrate, ...
            compression_ratio}];

    end
end

%% 5. CREATE RESULTS TABLE

Bandwidth_Results = cell2table(results, ...
    'VariableNames', ...
    {'Signal','Method','Bit_Depth','Sampling_Ratio', ...
    'Bits_Per_Window','Payload_Rate_bps', ...
    'Compression_Ratio'});

%% 6. COMBINED ECG + PPG DATA RATE

combined_results = {};

for m = 1:length(methods)

    method = methods{m};

    subset = Bandwidth_Results( ...
        strcmp(Bandwidth_Results.Method,method),:);

    if strcmp(method,'CS')

        for r = 1:length(sampling_ratios)

            ratio = sampling_ratios(r);

            idx = subset.Sampling_Ratio == ratio;

            total_rate = sum(subset.Payload_Rate_bps(idx));

            total_bits = sum(subset.Bits_Per_Window(idx));

            compression = ...
                (2 * N * original_bits) / total_bits;

            combined_results = [combined_results; ...
                {'ECG + PPG',method,NaN,ratio, ...
                total_bits,total_rate,compression}];

        end

    else

        for b = 1:length(bit_depths)

            bits = bit_depths(b);

            idx = subset.Bit_Depth == bits;

            total_rate = sum(subset.Payload_Rate_bps(idx));

            total_bits = sum(subset.Bits_Per_Window(idx));

            compression = ...
                (2 * N * original_bits) / total_bits;

            combined_results = [combined_results; ...
                {'ECG + PPG',method,bits,NaN, ...
                total_bits,total_rate,compression}];

        end
    end
end

Combined_Bandwidth = cell2table(combined_results, ...
    'VariableNames', ...
    {'Signal','Method','Bit_Depth','Sampling_Ratio', ...
    'Bits_Per_Window','Payload_Rate_bps', ...
    'Compression_Ratio'});

%% 7. DISPLAY RESULTS

fprintf('\n========================================\n');
fprintf('INDIVIDUAL SIGNAL BANDWIDTH RESULTS\n');
fprintf('========================================\n');

disp(Bandwidth_Results);

fprintf('\n========================================\n');
fprintf('COMBINED ECG + PPG BANDWIDTH RESULTS\n');
fprintf('========================================\n');

disp(Combined_Bandwidth);

%% 8. SAVE RESULTS

outputPath = fullfile(pwd,'results', ...
    'Communication_Analysis');

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

writetable(Bandwidth_Results, ...
    fullfile(outputPath,'Individual_Bandwidth.csv'));

writetable(Combined_Bandwidth, ...
    fullfile(outputPath,'Combined_Bandwidth.csv'));

save(fullfile(outputPath,'Bandwidth_Analysis.mat'), ...
    'Bandwidth_Results','Combined_Bandwidth');

%% 9. GRAPH 1: COMBINED PAYLOAD RATE

figure('Color','w');

hold on;

colors = lines(3);

for m = 1:3

    idx = strcmp(Combined_Bandwidth.Method,methods{m});

    T = Combined_Bandwidth(idx,:);

    if strcmp(methods{m},'CS')

        x = T.Sampling_Ratio;

    else

        x = T.Bit_Depth;

    end

    plot(x,T.Payload_Rate_bps,'-o', ...
        'LineWidth',1.8, ...
        'MarkerSize',7, ...
        'Color',colors(m,:));

end

xlabel('Sampling Ratio (%) / Bit Depth');
ylabel('Combined Payload Rate (bps)');
title('ECG + PPG Transmission Data Rate');

legend('PCM','DPCM','Compressive Sensing', ...
    'Location','best');

grid on;

saveas(gcf,fullfile(outputPath, ...
    'Combined_Data_Rate.png'));

%% 10. GRAPH 2: COMPRESSION RATIO

figure('Color','w');

hold on;

for m = 1:3

    idx = strcmp(Combined_Bandwidth.Method,methods{m});

    T = Combined_Bandwidth(idx,:);

    if strcmp(methods{m},'CS')
        x = T.Sampling_Ratio;
    else
        x = T.Bit_Depth;
    end

    plot(x,T.Compression_Ratio,'-s', ...
        'LineWidth',1.8, ...
        'MarkerSize',7, ...
        'Color',colors(m,:));

end

xlabel('Sampling Ratio (%) / Bit Depth');
ylabel('Compression Ratio');
title('ECG + PPG Compression Ratio');

legend('PCM','DPCM','Compressive Sensing', ...
    'Location','best');

grid on;

saveas(gcf,fullfile(outputPath, ...
    'Combined_Compression_Ratio.png'));

%% 11. COMPLETION

fprintf('\n========================================\n');
fprintf('BANDWIDTH ANALYSIS COMPLETED\n');
fprintf('========================================\n');

fprintf('Individual results: %d rows\n', ...
    height(Bandwidth_Results));

fprintf('Combined results: %d rows\n', ...
    height(Combined_Bandwidth));

fprintf('Output folder:\n%s\n',outputPath);
