
clc;
clear;
close all;

fprintf('\n========================================\n');
fprintf('DCT-LASSO PACKET RELIABILITY CORRECTION\n');
fprintf('========================================\n\n');

%% 1. Define paths

inputFile = fullfile('results', ...
    'LASSO_DCT_Packet_Loss', ...
    'LASSO_DCT_Packet_Loss.mat');

outputFolder = fullfile('results', ...
    'LASSO_DCT_Packet_Loss');

if ~isfile(inputFile)
    error('Input MAT file not found: %s', inputFile);
end

%% 2. Load original results

S = load(inputFile);

if ~isfield(S, 'SummaryTable') || ...
   ~isfield(S, 'WindowTable')
    error('SummaryTable or WindowTable is missing.');
end

CorrectedTable = S.SummaryTable;
WindowTable = S.WindowTable;

fprintf('Original LASSO results loaded successfully.\n');

%% 3. Verify required columns

requiredColumns = {
    'Signal'
    'SamplingRatio'
    'Lambda'
    'PacketLossPercent'
    'MSE'
    'SNR_dB'
    'PRD_percent'
    'PacketCount'
};

existingColumns = CorrectedTable.Properties.VariableNames;

for i = 1:numel(requiredColumns)
    if ~ismember(requiredColumns{i}, existingColumns)
        error('Required column missing: %s', requiredColumns{i});
    end
end

%% 4. Calculate theoretical all-packet delivery probability

p = CorrectedTable.PacketLossPercent ./ 100;
K = CorrectedTable.PacketCount;

allPacketsProbability = (1-p).^K;

allPacketsPercent = 100 .* allPacketsProbability;

%% 5. Remove misleading empirical success column

if ismember('BlockSuccessPercent', ...
        CorrectedTable.Properties.VariableNames)

    CorrectedTable.BlockSuccessPercent = [];
end

%% 6. Remove duplicate theoretical reliability columns

columnsToRemove = {
    'TheoreticalBlockSuccessPercent'
    'TheoreticalAllPacketsReceivedPercent'
};

for i = 1:numel(columnsToRemove)

    if ismember(columnsToRemove{i}, ...
            CorrectedTable.Properties.VariableNames)

        CorrectedTable.(columnsToRemove{i}) = [];
    end

end

%% 7. Add one corrected reliability column

CorrectedTable.AllPacketsReceivedProbabilityPercent = ...
    allPacketsPercent;

%% 8. Display corrected summary

fprintf('\nCORRECTED LASSO SUMMARY\n');
fprintf('----------------------------------------\n');

disp(CorrectedTable(1:min(10,height(CorrectedTable)),:));

%% 9. Validate configurations

expectedConfigurations = 50;

fprintf('\nTotal configurations: %d\n', ...
    height(CorrectedTable));

if height(CorrectedTable) ~= expectedConfigurations
    warning('Expected 50 configurations, found %d.', ...
        height(CorrectedTable));
else
    fprintf('Configuration count verified: 50/50.\n');
end

%% 10. Validate probability values

if any(allPacketsPercent < 0 | allPacketsPercent > 100)
    error('Invalid reliability probability detected.');
end

fprintf('Reliability probability validation passed.\n');

%% 11. Save corrected CSV

csvFile = fullfile(outputFolder, ...
    'LASSO_DCT_Packet_Loss_Corrected_Summary.csv');

writetable(CorrectedTable, csvFile);

%% 12. Save corrected MAT file

matFile = fullfile(outputFolder, ...
    'LASSO_DCT_Packet_Loss_Corrected.mat');

save(matFile, ...
    'CorrectedTable', ...
    'WindowTable', ...
    'S');

%% 13. Generate corrected reliability graphs

signals = {'ECG', 'PPG'};

samplingRatios = unique(CorrectedTable.SamplingRatio);

for s = 1:numel(signals)

    fig = figure( ...
        'Name', [signals{s} ' LASSO Packet Reliability'], ...
        'Color', 'w');

    hold on;
    grid on;

    for r = 1:numel(samplingRatios)

        idx = strcmp(CorrectedTable.Signal, signals{s}) & ...
              CorrectedTable.SamplingRatio == samplingRatios(r);

        subset = CorrectedTable(idx,:);

        [~, order] = sort(subset.PacketLossPercent);

        plot( ...
            subset.PacketLossPercent(order), ...
            subset.AllPacketsReceivedProbabilityPercent(order), ...
            '-o', ...
            'LineWidth', 1.5, ...
            'DisplayName', ...
            sprintf('%d%% CS', samplingRatios(r)));
    end

    xlabel('Packet Loss Probability (%)');
    ylabel('All-Packet Delivery Probability (%)');

    title([signals{s} ...
        ' - DCT-LASSO Theoretical Packet Reliability']);

    legend('Location', 'best');
    ylim([0 105]);

    grid minor;

    graphFile = fullfile(outputFolder, ...
        [signals{s} '_LASSO_Corrected_Packet_Reliability.png']);

    exportgraphics(fig, graphFile, 'Resolution', 300);

    close(fig);
end

%% 14. Completion message

fprintf('\n========================================\n');
fprintf('LASSO RELIABILITY CORRECTION COMPLETED\n');
fprintf('========================================\n');

fprintf('\nSaved files:\n');
fprintf('%s\n', csvFile);
fprintf('%s\n', matFile);

fprintf('\nCorrected ECG and PPG graphs saved.\n');
fprintf('Original reconstruction metrics preserved.\n');
fprintf('Original source files not overwritten.\n');

fprintf('\nDCT-LASSO correction completed successfully!\n');
