
%% OMP DIAGNOSTIC USING DCT SPARSE REPRESENTATION
% Version 2: validation-based sparsity selection
% Original OMP results are not overwritten.

clear; clc; close all;

%% 1. Load datasets
load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

signals = {'ECG','PPG'};
ratios = sampling_ratios;
N = double(N);

% Use a maximum of 64 validation windows for initial tuning
numVal = 256;

% Candidate sparsity levels
K_candidates = 5:5:120;

% Separate output folder
outDir = fullfile(pwd,'results','OMP_DCT_Diagnostic');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 2. Construct orthonormal DCT-II basis
% x = Psi * alpha
% alpha = DCT(x)

C = zeros(N,N);

for k = 0:N-1
    if k == 0
        scale = sqrt(1/N);
    else
        scale = sqrt(2/N);
    end

    for n = 0:N-1
        C(k+1,n+1) = scale*cos( ...
            pi*(2*n+1)*k/(2*N));
    end
end

Psi = C';

fprintf('DCT orthonormality error: %.3e\n', ...
    norm(C*C' - eye(N),'fro'));

%% 3. Initialize result storage
tuningResults = [];
testResults = [];

%% 4. Process ECG and PPG
for s = 1:2

    signalName = signals{s};

    if strcmp(signalName,'ECG')
        Xval = val_ecg;
        Xtest = test_ecg;
    else
        Xval = val_ppg;
        Xtest = test_ppg;
    end

    fprintf('\n====================================\n');
    fprintf('PROCESSING %s\n',signalName);
    fprintf('====================================\n');

    %% 5. Process sampling ratios
    for r = 1:length(ratios)

        Phi_current = double(Phi{r});
        M = size(Phi_current,1);

        % Compressive sensing dictionary
        D = Phi_current * Psi;

        % Normalize dictionary columns
        columnNorms = sqrt(sum(D.^2,1));
        columnNorms(columnNorms < eps) = 1;

        Dnorm = D ./ columnNorms;

        % Limit K to a safe range
        validK = K_candidates(K_candidates < M);

        fprintf('\n%s | %d%% CS | M = %d\n', ...
            signalName,ratios(r),M);

        %% 6. Tune K using validation data
        nVal = min(numVal,size(Xval,1));

        valMSE = zeros(length(validK),1);

        for ki = 1:length(validK)

            K = validK(ki);
            totalMSE = 0;

            for w = 1:nVal

                x = Xval(w,:)';
                y = Phi_current*x;

                xhat = ompDCT(y,Dnorm,Psi,columnNorms,K);

                totalMSE = totalMSE + mean((x-xhat).^2);
            end

            valMSE(ki) = totalMSE/nVal;

            fprintf('K = %2d | Validation MSE = %.6f\n', ...
                K,valMSE(ki));
        end

        % Select K with minimum validation MSE
        [bestValMSE,bestIndex] = min(valMSE);
        bestK = validK(bestIndex);

        fprintf('SELECTED K = %d | Validation MSE = %.6f\n', ...
            bestK,bestValMSE);

        tuningResults = [tuningResults; ...
            {signalName,ratios(r),bestK,bestValMSE}];

        %% 7. Evaluate selected K on complete test set
        nTest = size(Xtest,1);

        mseValues = zeros(nTest,1);
        snrValues = zeros(nTest,1);
        prdValues = zeros(nTest,1);

        for w = 1:nTest

            x = Xtest(w,:)';
            y = Phi_current*x;

            xhat = ompDCT(y,Dnorm,Psi,columnNorms,bestK);

            err = x-xhat;

            mseValues(w) = mean(err.^2);

            signalPower = sum(x.^2);
            errorPower = sum(err.^2);

            snrValues(w) = 10*log10( ...
                signalPower/(errorPower+eps));

            prdValues(w) = 100*sqrt( ...
                errorPower/(signalPower+eps));
        end

        avgMSE = mean(mseValues);
        avgSNR = mean(snrValues);
        avgPRD = mean(prdValues);

        fprintf(['TEST RESULTS | MSE: %.6f | ' ...
            'SNR: %.4f dB | PRD: %.4f%%\n'], ...
            avgMSE,avgSNR,avgPRD);

        testResults = [testResults; ...
            {signalName,ratios(r),bestK, ...
            avgMSE,avgSNR,avgPRD,nTest}];

    end
end

%% 8. Save results
TuningTable = cell2table(tuningResults, ...
    'VariableNames',{'Signal','SamplingRatio', ...
    'BestK','ValidationMSE'});

TestTable = cell2table(testResults, ...
    'VariableNames',{'Signal','SamplingRatio', ...
    'BestK','MSE','SNR_dB','PRD_percent','TestWindows'});

writetable(TuningTable, ...
    fullfile(outDir,'OMP_DCT_Tuning.csv'));

writetable(TestTable, ...
    fullfile(outDir,'OMP_DCT_Test_Results.csv'));

save(fullfile(outDir,'OMP_DCT_Diagnostic.mat'), ...
    'TuningTable','TestTable','K_candidates');

%% 9. Plot results
figure('Color','w','Position',[100 100 1200 700]);

for s = 1:2

    signalName = signals{s};
    rows = strcmp(TestTable.Signal,signalName);

    subplot(2,2,2*s-1);
    plot(TestTable.SamplingRatio(rows), ...
        TestTable.MSE(rows),'-o','LineWidth',2);
    grid on;
    xlabel('Sampling Ratio (%)');
    ylabel('MSE');
    title([signalName ' OMP-DCT MSE']);

    subplot(2,2,2*s);
    plot(TestTable.SamplingRatio(rows), ...
        TestTable.SNR_dB(rows),'-o','LineWidth',2);
    grid on;
    xlabel('Sampling Ratio (%)');
    ylabel('SNR (dB)');
    title([signalName ' OMP-DCT SNR']);

end

saveas(gcf,fullfile(outDir,'OMP_DCT_Performance.png'));

fprintf('\n====================================\n');
fprintf('OMP DCT DIAGNOSTIC COMPLETED\n');
fprintf('Results saved in:\n%s\n',outDir);
fprintf('====================================\n');

%% 10. OMP reconstruction function
function xhat = ompDCT(y,Dnorm,Psi,columnNorms,K)

    residual = y;
    support = [];
    alpha = zeros(size(Dnorm,2),1);

    for iteration = 1:K

        correlations = Dnorm' * residual;

        % Prevent selecting an existing atom again
        if ~isempty(support)
            correlations(support) = 0;
        end

        [~,selected] = max(abs(correlations));

        support = [support selected];

        % Least-squares coefficient estimation
        A = Dnorm(:,support);
        beta = A \ y;

        residual = y-A*beta;

        if norm(residual) < 1e-8
            break;
        end
    end

    % Convert normalized dictionary coefficients
    alpha(support) = beta ./ columnNorms(support)';

    % Reconstruct original signal
    xhat = Psi*alpha;
end
