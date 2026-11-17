
clc;
clear;
close all;

%% CORRECTED COMMUNICATION BANDWIDTH ANALYSIS

fs = 125;
N = 512;
stride = 256;
bits_CS = 16;

bit_depths = [8 10 12];
ratios = [10 20 30 40 50];

outputPath = fullfile(pwd,'results', ...
    'Communication_Analysis_Corrected');

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

%% PCM AND DPCM: CONTINUOUS STREAM TRANSMISSION

methodNames = {'PCM','DPCM'};
results = {};

for m = 1:2
    for b = 1:length(bit_depths)

        bits = bit_depths(b);

        ratePerSignal = fs * bits;

        combinedRate = 2 * ratePerSignal;

        compression = 16 / bits;

        results = [results; ...
            {methodNames{m},bits,NaN, ...
            ratePerSignal,combinedRate,compression}];

    end
end

%% CS: NON-OVERLAPPING AND OVERLAPPING WINDOWS

for r = 1:length(ratios)

    ratio = ratios(r);
    M = round(N * ratio / 100);

    bitsPerWindow = M * bits_CS;

    % Non-overlapping block transmission
    rateNonOverlap = ...
        bitsPerWindow * fs / N;

    % Every overlapping window transmitted independently
    rateOverlap = ...
        bitsPerWindow * fs / stride;

    combinedNonOverlap = 2 * rateNonOverlap;
    combinedOverlap = 2 * rateOverlap;

    compression = ...
        (N * 16) / bitsPerWindow;

    results = [results; ...
        {'CS_NonOverlap',16,ratio, ...
        rateNonOverlap,combinedNonOverlap,compression}];

    results = [results; ...
        {'CS_Overlap',16,ratio, ...
        rateOverlap,combinedOverlap,compression}];

end

%% CREATE TABLE

T = cell2table(results, ...
    'VariableNames', ...
    {'Method','Bit_Depth','Sampling_Ratio', ...
    'Individual_Rate_bps','Combined_Rate_bps', ...
    'Compression_Ratio'});

disp(T);

%% SAVE RESULTS

writetable(T,fullfile(outputPath, ...
    'Corrected_Bandwidth_Comparison.csv'));

save(fullfile(outputPath, ...
    'Corrected_Bandwidth_Comparison.mat'),'T');

%% GRAPH 1: COMBINED DATA RATE FOR CS

csRows = startsWith(string(T.Method),'CS');
CS = T(csRows,:);

figure('Color','w');
hold on;

for i = 1:2

    if i == 1
        method = 'CS_NonOverlap';
        label = 'Non-overlapping';
    else
        method = 'CS_Overlap';
        label = 'Overlapping';
    end

    idx = strcmp(T.Method,method);

    plot(T.Sampling_Ratio(idx), ...
        T.Combined_Rate_bps(idx),'-o', ...
        'LineWidth',2,'MarkerSize',7, ...
        'DisplayName',label);
end

xlabel('Sampling Ratio (%)');
ylabel('Combined Payload Rate (bps)');
title('CS Transmission Rate: Windowing Comparison');
legend('Location','best');
grid on;

saveas(gcf,fullfile(outputPath, ...
    'CS_Windowing_Rate_Comparison.png'));

%% COMPLETION

fprintf('\n====================================\n');
fprintf('CORRECTED ANALYSIS COMPLETED\n');
fprintf('====================================\n');

fprintf('Total configurations: %d\n',height(T));
fprintf('Output folder:\n%s\n',outputPath);
