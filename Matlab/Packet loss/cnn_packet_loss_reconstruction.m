
clc;
clear;
close all;

fprintf('\n========================================\n');
fprintf('DCT-OMP PACKET RELIABILITY CORRECTION\n');
fprintf('========================================\n\n');

%% 1. Define paths

inputFile = fullfile('results', ...
    'OMP_DCT_Packet_Loss', ...
    'OMP_DCT_Packet_Loss.mat');

outputFolder = fullfile('results', ...
    'OMP_DCT_Packet_Loss');

if ~isfile(inputFile)
    error('Input MAT file not found: %s', inputFile);
end

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% 2. Load original results

S = load(inputFile);

if ~isfield(S, 'SummaryTable') || ...
   ~isfield(S, 'WindowTable')

    error('SummaryTable or WindowTable is missing.');
end

CorrectedTable = S.SummaryTable;
WindowTable = S.WindowTable;

fprintf('Original results loaded successfully.\n');

%% 3. Verify required columns

requiredColumns = {
    'Signal'
    'SamplingRatio'
    'PacketLossPercent'
    'PacketCount'
    'MSE'
    'SNR_dB'
    'PRD_percent'
};

existingColumns = CorrectedTable.Properties.VariableNames;

for i = 1:numel(requiredColumns)
    if ~ismember(requiredColumns{i}, existingColumns)
        error('Required column missing: %s', requiredColumns{i});
    end
end

%% 4. Calculate theoretical all-packet delivery probability

% Convert packet loss percentage into probability
p = CorrectedTable.PacketLossPercent ./ 100;

% Number of packets per block
K = CorrectedTable.PacketCount;

% Probability that every packet is received
allPacketsProbability = (1-p).^K;

% Convert to percentage
CorrectedTable.TheoreticalAllPacketsReceivedPercent = ...
    100 .* allPacketsProbability;

%% 5. Remove misleading empirical success column

if ismember('BlockSuccessPercent', ...
        CorrectedTable.Properties.VariableNames)

    CorrectedTable.BlockSuccessPercent = [];
end

%% 6. Preserve original theoretical column if present

% Replace the old theoretical column with the recalculated value
if ismember('TheoreticalBlockSuccessPercent', ...
        CorrectedTable.Properties.VariableNames)

    CorrectedTable.TheoreticalBlockSuccessPercent = ...
        CorrectedTable.TheoreticalAllPacketsReceivedPercent;
end

%% 7. Display corrected summary

fprintf('\nCORRECTED SUMMARY\n');
fprintf('----------------------------------------\n');

disp(CorrectedTable(1:min(10,height(CorrectedTable)), :));

%% 8. Validate number of configurations

expectedConfigurations = 50;

fprintf('\nTotal configurations: %d\n', ...
    height(CorrectedTable));

if height(CorrectedTable) ~= expectedConfigurations
    warning('Expected %d configurations, found %d.', ...
        expectedConfigurations, height(CorrectedTable));
else
    fprintf('Configuration count verified: 50/50.\n');
end

%% 9. Save corrected CSV

csvFile = fullfile(outputFolder, ...
    'OMP_DCT_Packet_Loss_Corrected_Summary.csv');

writetable(CorrectedTable, csvFile);

%% 10. Save corrected MAT file

matFile = fullfile(outputFolder, ...
    'OMP_DCT_Packet_Loss_Corrected.mat');

save(matFile, ...
    'CorrectedTable', ...
    'WindowTable');

%% 11. Generate corrected reliability graph

signals = {'ECG', 'PPG'};
samplingRatios = unique(CorrectedTable.SamplingRatio);
lossRates = unique(CorrectedTable.PacketLossPercent);

for s = 1:numel(signals)

    figure('Name', ...
        [signals{s} ' Packet Reliability'], ...
        'Color', 'w');

    hold on;
    grid on;

    for r = 1:numel(samplingRatios)

        idx = strcmp(CorrectedTable.Signal, signals{s}) & ...
              CorrectedTable.SamplingRatio == samplingRatios(r);

        subset = CorrectedTable(idx, :);

        [~, order] = sort(subset.PacketLossPercent);

        plot( ...
            subset.PacketLossPercent(order), ...
            subset.TheoreticalAllPacketsReceivedPercent(order), ...
            '-o', ...
            'LineWidth', 1.5, ...
            'DisplayName', ...
            sprintf('%d%% CS', samplingRatios(r)));
    end

    xlabel('Packet Loss Probability (%)');
    ylabel('All-Packet Delivery Probability (%)');

    title([signals{s} ...
        ' - Theoretical Packet Delivery Reliability']);

    legend('Location', 'best');
    ylim([0 105]);

    saveas(gcf, fullfile(outputFolder, ...
        [signals{s} '_Corrected_Packet_Reliability.png']));

    close(gcf);
end

%% 12. Final completion message

fprintf('\n========================================\n');
fprintf('CORRECTION COMPLETED SUCCESSFULLY\n');
fprintf('========================================\n');

fprintf('\nFiles saved:\n');
fprintf('%s\n', csvFile);
fprintf('%s\n', matFile);

fprintf('\nCorrected reliability graphs saved.\n');
fprintf('Original reconstruction metrics preserved.\n');
fprintf('Original files were not overwritten.\n');

fprintf('\nDCT-OMP reliability correction completed!\n');
