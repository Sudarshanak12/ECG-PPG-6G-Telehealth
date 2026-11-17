
%% DCT-LASSO RECONSTRUCTION UNDER PACKET LOSS
% Uses DCT-domain sparse representation.
% Uses the existing MATLAB LASSO lambda = 0.01.
% Original LASSO and other results are preserved.

clear; clc; close all;
rng(42,'twister');

%% 1. Load dataset and measurement matrices
load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

signals = {'ECG','PPG'};
ratios = [10 20 30 40 50];
lossRates = [0 1 3 5 10];

N = 512;
measurementsPerPacket = 4;

lambda = 0.01;
maxIterations = 100;
tolerance = 1e-5;

%% 2. Construct orthonormal DCT-II basis
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
    norm(C*C'-eye(N),'fro'));

%% 3. Output directory
outDir = fullfile(pwd,'results', ...
    'LASSO_DCT_Packet_Loss');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 4. Initialize storage
summaryResults = {};
windowResults = {};

%% 5. Process ECG and PPG
for s = 1:2

    signalName = signals{s};

    if strcmp(signalName,'ECG')
        Xtest = double(test_ecg);
    else
        Xtest = double(test_ppg);
    end

    fprintf('\n====================================\n');
    fprintf('PROCESSING %s\n',signalName);
    fprintf('====================================\n');

    for r = 1:length(ratios)

        PhiCurrent = double(Phi{r});
        M = size(PhiCurrent,1);

        % DCT-domain dictionary
        D = PhiCurrent*Psi;

        % Normalize dictionary columns
        columnNorms = sqrt(sum(D.^2,1));
        columnNorms(columnNorms < eps) = 1;

        Dnorm = D ./ columnNorms;

        % Lipschitz constant of gradient
        % Power iteration estimates largest eigenvalue of D'*D
        v = ones(size(Dnorm,2),1);
        v = v/norm(v);

        for q = 1:30
            v = Dnorm'*(Dnorm*v);
            v = v/(norm(v)+eps);
        end

        L = norm(Dnorm*v)^2;

        packetCount = ceil(M/measurementsPerPacket);

        fprintf('\n%s | %d%% CS | M=%d\n', ...
            signalName,ratios(r),M);

        for lp = 1:length(lossRates)

            lossPercent = lossRates(lp);
            lossProbability = lossPercent/100;

            nTest = size(Xtest,1);

            mseValues = zeros(nTest,1);
            snrValues = zeros(nTest,1);
            prdValues = zeros(nTest,1);

            successfulBlocks = 0;
            lostPacketsTotal = 0;

            for w = 1:nTest

                x = Xtest(w,:)';

                % Compressed measurements
                y = PhiCurrent*x;

                % Simulate packet losses
                packetLost = rand(packetCount,1) ...
                    < lossProbability;

                lostPacketsTotal = lostPacketsTotal ...
                    + sum(packetLost);

                receivedMask = true(M,1);

                for p = 1:packetCount

                    firstIndex = ...
                        (p-1)*measurementsPerPacket+1;

                    lastIndex = min( ...
                        p*measurementsPerPacket,M);

                    if packetLost(p)
                        receivedMask(firstIndex:lastIndex) ...
                            = false;
                    end
                end

                receivedIndices = find(receivedMask);

                if isempty(receivedIndices)

                    xhat = zeros(N,1);

                else

                    successfulBlocks = successfulBlocks+1;

                    A = Dnorm(receivedIndices,:);
                    yReceived = y(receivedIndices);

                    % FISTA LASSO reconstruction
                    alpha = fistaLasso( ...
                        A,yReceived,lambda,L, ...
                        maxIterations,tolerance);

                    % Undo dictionary column normalization
                    alphaOriginal = alpha ./ columnNorms';

                    % Reconstruct time-domain signal
                    xhat = Psi*alphaOriginal;
                end

                % Metrics
                err = x-xhat;

                signalPower = sum(x.^2);
                errorPower = sum(err.^2);

                mseValues(w) = mean(err.^2);

                snrValues(w) = 10*log10( ...
                    signalPower/(errorPower+eps));

                prdValues(w) = 100*sqrt( ...
                    errorPower/(signalPower+eps));

                windowResults(end+1,:) = { ...
                    signalName,ratios(r),lossPercent, ...
                    w,mseValues(w),snrValues(w), ...
                    prdValues(w)};
            end

            avgMSE = mean(mseValues);
            avgSNR = mean(snrValues);
            avgPRD = mean(prdValues);

            blockSuccess = ...
                successfulBlocks/nTest*100;

            theoreticalSuccess = ...
                (1-lossProbability)^packetCount*100;

            avgLostPackets = lostPacketsTotal/nTest;

            fprintf(['Loss=%2d%% | MSE=%.6f | ' ...
                'SNR=%.4f dB | PRD=%.4f%% | ' ...
                'Block success=%.2f%%\n'], ...
                lossPercent,avgMSE,avgSNR,avgPRD, ...
                blockSuccess);

            summaryResults(end+1,:) = { ...
                signalName,ratios(r),lambda,lossPercent, ...
                avgMSE,avgSNR,avgPRD,packetCount, ...
                avgLostPackets,blockSuccess, ...
                theoreticalSuccess};
        end
    end
end

%% 6. Create result tables
SummaryTable = cell2table(summaryResults, ...
    'VariableNames',{ ...
    'Signal','SamplingRatio','Lambda','PacketLossPercent', ...
    'MSE','SNR_dB','PRD_percent','PacketCount', ...
    'AverageLostPackets','BlockSuccessPercent', ...
    'TheoreticalBlockSuccessPercent'});

WindowTable = cell2table(windowResults, ...
    'VariableNames',{ ...
    'Signal','SamplingRatio','PacketLossPercent', ...
    'Window','MSE','SNR_dB','PRD_percent'});

%% 7. Save results
writetable(SummaryTable, ...
    fullfile(outDir,'LASSO_DCT_Packet_Loss_Summary.csv'));

writetable(WindowTable, ...
    fullfile(outDir,'LASSO_DCT_Packet_Loss_WindowResults.csv'));

save(fullfile(outDir,'LASSO_DCT_Packet_Loss.mat'), ...
    'SummaryTable','WindowTable','lambda', ...
    'maxIterations','tolerance');

%% 8. Generate plots
for s = 1:2

    signalName = signals{s};

    figure('Color','w','Position',[100 100 1100 700]);

    subplot(2,1,1);
    hold on;

    for r = 1:length(ratios)

        mask = strcmp(SummaryTable.Signal,signalName) & ...
            SummaryTable.SamplingRatio == ratios(r);

        plot(SummaryTable.PacketLossPercent(mask), ...
            SummaryTable.MSE(mask),'-o', ...
            'LineWidth',1.8, ...
            'DisplayName',sprintf('%d%% CS',ratios(r)));
    end

    grid on;
    xlabel('Packet Loss (%)');
    ylabel('MSE');
    title([signalName ' DCT-LASSO MSE under Packet Loss']);
    legend('Location','best');
    hold off;

    subplot(2,1,2);
    hold on;

    for r = 1:length(ratios)

        mask = strcmp(SummaryTable.Signal,signalName) & ...
            SummaryTable.SamplingRatio == ratios(r);

        plot(SummaryTable.PacketLossPercent(mask), ...
            SummaryTable.SNR_dB(mask),'-o', ...
            'LineWidth',1.8, ...
            'DisplayName',sprintf('%d%% CS',ratios(r)));
    end

    grid on;
    xlabel('Packet Loss (%)');
    ylabel('SNR (dB)');
    title([signalName ' DCT-LASSO SNR under Packet Loss']);
    legend('Location','best');
    hold off;

    saveas(gcf,fullfile(outDir, ...
        [signalName '_LASSO_DCT_Packet_Loss.png']));
end

%% 9. Final verification
fprintf('\n====================================\n');
fprintf('DCT-LASSO PACKET LOSS COMPLETED\n');
fprintf('Summary configurations: %d\n',height(SummaryTable));
fprintf('Window evaluations: %d\n',height(WindowTable));

fprintf('NaN MSE: %d\n',sum(isnan(WindowTable.MSE)));
fprintf('Inf MSE: %d\n',sum(isinf(WindowTable.MSE)));

fprintf('Results saved in:\n%s\n',outDir);
fprintf('====================================\n');

%% 10. FISTA LASSO solver
function alpha = fistaLasso(A,y,lambda,L,maxIter,tol)

    n = size(A,2);

    alpha = zeros(n,1);
    z = alpha;
    t = 1;

    for iter = 1:maxIter

        gradient = A'*(A*z-y);

        alphaNew = softThreshold( ...
            z-gradient/L,lambda/L);

        tNew = (1+sqrt(1+4*t^2))/2;

        zNew = alphaNew + ...
            ((t-1)/tNew)*(alphaNew-alpha);

        relativeChange = norm(alphaNew-alpha) / ...
            (norm(alpha)+1e-10);

        alpha = alphaNew;
        z = zNew;
        t = tNew;

        if relativeChange < tol
            break;
        end
    end
end

%% 11. Soft thresholding
function output = softThreshold(x,threshold)

    output = sign(x).*max(abs(x)-threshold,0);

end
