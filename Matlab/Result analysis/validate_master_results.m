
clc;
clear;

folder = '..\results\08_Final_Results\Master_Comparison';

M = readtable(fullfile(folder,'Master_Comparison.csv'));
C = readtable(fullfile(folder,'CNN_Platform_Comparison.csv'));

fprintf('\n====================================\n');
fprintf('MASTER RESULTS VALIDATION\n');
fprintf('====================================\n');

%% 1. Check master row count

fprintf('\nMaster rows: %d\n', height(M));

assert(height(M) == 50, ...
    'Unexpected master comparison row count.');

%% 2. Check platform distribution

fprintf('\nPlatform distribution:\n');
disp(groupsummary(M,'Platform'));

%% 3. Check method distribution

fprintf('\nMethod distribution:\n');
disp(groupsummary(M,'Method'));

%% 4. Check missing numerical values

numericColumns = {'Ratio','MSE_mean','MSE_std', ...
    'SNR_dB_mean','SNR_dB_std', ...
    'PRD_percent_mean','PRD_percent_std'};

fprintf('\nMissing values in master comparison:\n');

for i = 1:numel(numericColumns)

    col = M.(numericColumns{i});

    fprintf('%s: %d\n', ...
        numericColumns{i}, sum(ismissing(col)));

end

%% 5. Check duplicate configurations

[~,~,groupID] = unique( ...
    M(:,{'Signal','Ratio','Method','Platform'}), ...
    'rows');

counts = accumarray(groupID,1);

fprintf('\nDuplicate configuration groups: %d\n', ...
    sum(counts > 1));

%% 6. Check CNN platform comparison

fprintf('\nCNN platform comparison rows: %d\n',height(C));

disp(groupsummary(C,'Platform'));

%% 7. Check MATLAB CNN standard deviations

matlabCNN = C(strcmp(C.Platform,'MATLAB'),:);

fprintf('\nMATLAB CNN missing standard deviations:\n');

disp(sum(ismissing(matlabCNN(:, ...
    {'MSE_std','SNR_dB_std','PRD_percent_std'}))));

%% 8. Final completion

fprintf('\n====================================\n');
fprintf('MASTER VALIDATION FINISHED\n');
fprintf('====================================\n');
