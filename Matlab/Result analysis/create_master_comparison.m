
clc;
clear;
close all;

%% =====================================================
%  MASTER PYTHON-MATLAB CNN COMPARISON
%  ECG and PPG Reconstruction
% ======================================================

%% 1. DEFINE PROJECT PATHS

basePath = 'D:\sudarshan\dsu\sem\sem 5\adc project';

pythonFile = fullfile(basePath, ...
    'results', '08_Final_Results', 'csv', ...
    'Final_Test_Summary.csv');

matlabFile = fullfile(basePath, ...
    'results', 'final final', ...
    'MATLAB_CNN_FullTest_Summary.csv');

outputPath = fullfile(basePath, ...
    'results', '08_Final_Results', ...
    'Master_Comparison');

%% 2. CHECK FILES

fprintf('\n====================================\n');
fprintf('CHECKING INPUT FILES\n');
fprintf('====================================\n');

if ~isfile(pythonFile)
    error('Python summary file not found: %s', pythonFile);
end

if ~isfile(matlabFile)
    error('MATLAB summary file not found: %s', matlabFile);
end

fprintf('Python file found.\n');
fprintf('MATLAB file found.\n');

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

%% 3. LOAD PYTHON RESULTS

P = readtable(pythonFile);

requiredPython = { ...
    'Signal','Ratio','Method','MSE_mean','MSE_std', ...
    'SNR_dB_mean','SNR_dB_std', ...
    'PRD_percent_mean','PRD_percent_std'};

if ~all(ismember(requiredPython, ...
        P.Properties.VariableNames))
    error('Python summary has unexpected column names.');
end

fprintf('\nPython summary rows: %d\n',height(P));

%% 4. LOAD MATLAB CNN RESULTS

M = readtable(matlabFile);

requiredMatlab = { ...
    'Signal','Sampling_Ratio','Mean_MSE', ...
    'Mean_SNR_dB','Mean_PRD_percent'};

if ~all(ismember(requiredMatlab, ...
        M.Properties.VariableNames))
    error('MATLAB summary has unexpected column names.');
end

% Standardize MATLAB column names

M.Properties.VariableNames = { ...
    'Signal','Ratio','MSE_mean', ...
    'SNR_dB_mean','PRD_percent_mean'};

% MATLAB standard deviations are not available
% in the current summary file.

M.MSE_std = NaN(height(M),1);
M.SNR_dB_std = NaN(height(M),1);
M.PRD_percent_std = NaN(height(M),1);

%% 5. ADD PLATFORM LABELS

P.Platform = repmat({'Python'},height(P),1);
M.Platform = repmat({'MATLAB'},height(M),1);

%% 6. STANDARDIZE COLUMN ORDER

columnOrder = { ...
    'Signal','Ratio','Method', ...
    'MSE_mean','MSE_std', ...
    'SNR_dB_mean','SNR_dB_std', ...
    'PRD_percent_mean','PRD_percent_std', ...
    'Platform'};

% Add method labels for MATLAB CNN

M.Method = repmat({'CNN'},height(M),1);

P = P(:,columnOrder);
M = M(:,columnOrder);

%% 7. COMBINE RESULTS

Combined = [P; M];

fprintf('\n====================================\n');
fprintf('COMBINED DATA VALIDATION\n');
fprintf('====================================\n');

fprintf('Python rows: %d\n',height(P));
fprintf('MATLAB rows: %d\n',height(M));
fprintf('Total rows: %d\n',height(Combined));

if height(P) ~= 40 || height(M) ~= 10 || ...
        height(Combined) ~= 50
    error('Unexpected number of rows. Check input files.');
end

%% 8. SAVE MASTER DATASET

writetable(Combined, ...
    fullfile(outputPath,'Master_Comparison.csv'));

save(fullfile(outputPath,'Master_Comparison.mat'), ...
    'Combined');

fprintf('\nMaster CSV and MAT files saved.\n');

%% 9. PYTHON VS MATLAB CNN COMPARISON

signals = {'ECG','PPG'};

metrics = { ...
    'MSE_mean', ...
    'SNR_dB_mean', ...
    'PRD_percent_mean'};

metricTitles = { ...
    'Mean MSE', ...
    'Mean SNR (dB)', ...
    'Mean PRD (%)'};

for s = 1:length(signals)

    figure('Name',[signals{s} ' CNN Comparison'], ...
        'Position',[100 100 1400 450]);

    for m = 1:length(metrics)

        subplot(1,3,m);

        % Select Python CNN results

        idxPython = strcmp(Combined.Signal,signals{s}) & ...
            strcmp(Combined.Method,'CNN') & ...
            strcmp(Combined.Platform,'Python');

        % Select MATLAB CNN results

        idxMatlab = strcmp(Combined.Signal,signals{s}) & ...
            strcmp(Combined.Method,'CNN') & ...
            strcmp(Combined.Platform,'MATLAB');

        PY = Combined(idxPython,:);
        MA = Combined(idxMatlab,:);

        PY = sortrows(PY,'Ratio');
        MA = sortrows(MA,'Ratio');

        plot(PY.Ratio,PY.(metrics{m}), ...
            '-o','LineWidth',1.8,'MarkerSize',7);

        hold on;

        plot(MA.Ratio,MA.(metrics{m}), ...
            '-s','LineWidth',1.8,'MarkerSize',7);

        xlabel('Sampling Ratio (%)');
        ylabel(metricTitles{m});
        title(metricTitles{m});

        legend('Python CNN','MATLAB CNN', ...
            'Location','best');

        grid on;
        xticks([10 20 30 40 50]);

    end

    sgtitle([signals{s} ...
        ' CNN: Python vs MATLAB']);

    saveas(gcf,fullfile(outputPath, ...
        [signals{s} '_Python_vs_MATLAB_CNN.png']));

end

%% 10. SAVE CNN PLATFORM COMPARISON TABLE

pythonCNN = Combined( ...
    strcmp(Combined.Method,'CNN') & ...
    strcmp(Combined.Platform,'Python'),:);

matlabCNN = Combined( ...
    strcmp(Combined.Method,'CNN') & ...
    strcmp(Combined.Platform,'MATLAB'),:);

CNNComparison = [pythonCNN; matlabCNN];

writetable(CNNComparison, ...
    fullfile(outputPath,'CNN_Platform_Comparison.csv'));

%% 11. FINAL REPORT

fprintf('\n====================================\n');
fprintf('MASTER COMPARISON COMPLETED\n');
fprintf('====================================\n');

fprintf('Total combined rows: %d\n',height(Combined));
fprintf('Python full-test rows: %d\n',height(P));
fprintf('MATLAB CNN full-test rows: %d\n',height(M));

fprintf('\nFiles saved in:\n%s\n',outputPath);

fprintf('\nGenerated files:\n');
fprintf('1. Master_Comparison.csv\n');
fprintf('2. Master_Comparison.mat\n');
fprintf('3. CNN_Platform_Comparison.csv\n');
fprintf('4. ECG_Python_vs_MATLAB_CNN.png\n');
fprintf('5. PPG_Python_vs_MATLAB_CNN.png\n');

fprintf('\nALL OPERATIONS COMPLETED SUCCESSFULLY!\n');
