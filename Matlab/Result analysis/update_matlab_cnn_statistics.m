
clc;
clear;

fprintf('\n========================================\n');
fprintf('MATLAB CNN STATISTICS CORRECTION\n');
fprintf('========================================\n\n');

%% 1. File paths

detailFile = fullfile('..','results', ...
    '08_Final_Results','final final', ...
    'MATLAB_CNN_FullTest_WindowResults.csv');

masterFolder = fullfile('..','results', ...
    '08_Final_Results','Master_Comparison');

masterFile = fullfile(masterFolder, ...
    'Master_Comparison.csv');

cnnFile = fullfile(masterFolder, ...
    'CNN_Platform_Comparison.csv');

%% 2. Load files

if ~isfile(detailFile) || ...
   ~isfile(masterFile) || ...
   ~isfile(cnnFile)

    error('One or more input files could not be found.');
end

T = readtable(detailFile);
M = readtable(masterFile);
C = readtable(cnnFile);

fprintf('Detailed rows: %d\n',height(T));

%% 3. Verify detailed dataset

expectedRows = 20970;
expectedWindows = 2097;

assert(height(T) == expectedRows, ...
    'Unexpected detailed result row count.');

signals = {'ECG','PPG'};
ratios = [10 20 30 40 50];

%% 4. Backup original comparison files

masterBackup = [masterFile '.backup'];
cnnBackup = [cnnFile '.backup'];

if ~isfile(masterBackup)
    copyfile(masterFile,masterBackup);
end

if ~isfile(cnnBackup)
    copyfile(cnnFile,cnnBackup);
end

fprintf('\nOriginal comparison files backed up.\n');

%% 5. Calculate MATLAB CNN statistics

updatedCount = 0;

for s = 1:numel(signals)

    for r = 1:numel(ratios)

        signalName = signals{s};
        ratioValue = ratios(r);

        idx = strcmp(T.Signal,signalName) & ...
              T.Sampling_Ratio == ratioValue;

        D = T(idx,:);

        if height(D) ~= expectedWindows
            error('%s %d%% has %d windows, expected %d.', ...
                signalName,ratioValue,height(D),expectedWindows);
        end

        % Check window identifiers
        if numel(unique(D.Window)) ~= expectedWindows
            error('Duplicate or missing windows: %s %d%%.', ...
                signalName,ratioValue);
        end

        % Calculate means and sample standard deviations
        mseMean = mean(D.MSE);
        mseStd = std(D.MSE,0);

        snrMean = mean(D.SNR_dB);
        snrStd = std(D.SNR_dB,0);

        prdMean = mean(D.PRD_percent);
        prdStd = std(D.PRD_percent,0);

        % Locate MATLAB CNN row in master comparison
        masterIdx = ...
            strcmp(M.Signal,signalName) & ...
            M.Ratio == ratioValue & ...
            strcmp(M.Method,'CNN') & ...
            strcmp(M.Platform,'MATLAB');

        % Locate both platform rows in CNN comparison
        cnnIdx = ...
            strcmp(C.Signal,signalName) & ...
            C.Ratio == ratioValue & ...
            strcmp(C.Method,'CNN') & ...
            strcmp(C.Platform,'MATLAB');

        if sum(masterIdx) ~= 1 || sum(cnnIdx) ~= 1
            error('Could not uniquely locate %s %d%%.', ...
                signalName,ratioValue);
        end

        % Update master comparison
        M.MSE_mean(masterIdx) = mseMean;
        M.MSE_std(masterIdx) = mseStd;

        M.SNR_dB_mean(masterIdx) = snrMean;
        M.SNR_dB_std(masterIdx) = snrStd;

        M.PRD_percent_mean(masterIdx) = prdMean;
        M.PRD_percent_std(masterIdx) = prdStd;

        % Update CNN platform comparison
        C.MSE_mean(cnnIdx) = mseMean;
        C.MSE_std(cnnIdx) = mseStd;

        C.SNR_dB_mean(cnnIdx) = snrMean;
        C.SNR_dB_std(cnnIdx) = snrStd;

        C.PRD_percent_mean(cnnIdx) = prdMean;
        C.PRD_percent_std(cnnIdx) = prdStd;

        updatedCount = updatedCount + 1;

        fprintf(['%s | %d%% CS | ' ...
            'MSE %.6f +/- %.6f | ' ...
            'SNR %.4f +/- %.4f | ' ...
            'PRD %.4f +/- %.4f\n'], ...
            signalName,ratioValue, ...
            mseMean,mseStd, ...
            snrMean,snrStd, ...
            prdMean,prdStd);

    end
end

%% 6. Verify updated configurations

if updatedCount ~= 10
    error('Expected 10 MATLAB CNN configurations.');
end

matlabRows = strcmp(M.Platform,'MATLAB') & ...
             strcmp(M.Method,'CNN');

if any(ismissing(M(matlabRows, ...
        {'MSE_std','SNR_dB_std','PRD_percent_std'})))
    error('Missing MATLAB CNN standard deviations remain.');
end

%% 7. Save updated comparison files

writetable(M,masterFile);
writetable(C,cnnFile);

%% 8. Reload and verify saved files

Mcheck = readtable(masterFile);
Ccheck = readtable(cnnFile);

matlabRows = strcmp(Mcheck.Platform,'MATLAB') & ...
             strcmp(Mcheck.Method,'CNN');

fprintf('\nUpdated MATLAB master rows: %d\n', ...
    sum(matlabRows));

fprintf('Missing MATLAB standard deviations: %d\n', ...
    sum(ismissing(Mcheck.MSE_std(matlabRows))) + ...
    sum(ismissing(Mcheck.SNR_dB_std(matlabRows))) + ...
    sum(ismissing(Mcheck.PRD_percent_std(matlabRows))));

fprintf('Master comparison rows: %d\n',height(Mcheck));
fprintf('CNN comparison rows: %d\n',height(Ccheck));

fprintf('\n========================================\n');
fprintf('MATLAB CNN STATISTICS UPDATED\n');
fprintf('========================================\n');

fprintf('Updated configurations: %d/10\n',updatedCount);
fprintf('Original CSV backups preserved.\n');
fprintf('Both comparison CSVs saved successfully.\n');
