
%% DCT-OMP RECONSTRUCTION UNDER PACKET LOSS
% Uses validation-selected DCT-OMP sparsity values.
% Original OMP and BP results are not overwritten.

clear; clc; close all;

rng(42,'twister');

%% 1. Load data
load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

signals = {'ECG','PPG'};
ratios = [10 20 30 40 50];
lossRates = [0 1 3 5 10];

N = 512;
bitsPerMeasurement = 16;
measurementsPerPacket = 4;
numTrials = 1;

% Selected using 256 validation windows
bestK = [
     5  10  20  40  65;   % ECG
     5  25  60  70 105    % PPG
];

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

%% 3. Output folder
outDir = fullfile(pwd,'results', ...
    'OMP_DCT_Packet_Loss');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 4. Result storage
summaryResults = {};
windowResults = [];

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

        K = bestK(s,r);

        % DCT-domain measurement dictionary
        D = PhiCurrent*Psi;

        columnNorms = sqrt(sum(D.^2,1));
        columnNorms(columnNorms < eps) = 1;

        Dnorm = D ./ columnNorms;

        % Packet count for this measurement block
        packetCount = ceil(M/measurementsPerPacket);

        fprintf('\n%s | %d%% CS | M=%d | K=%d\n', ...
            signalName,ratios(r),M,K);

        for lp = 1:length(lossRates)

            lossPercent = lossRates(lp);
            lossProbability = lossPercent/100;

            mseValues = zeros(size(Xtest,1),1);
            snrValues = zeros(size(Xtest,1),1);
            prdValues = zeros(size(Xtest,1),1);

            successfulBlocks = 0;
            failedBlocks = 0;
            lostPacketsTotal = 0;

            for w = 1:size(Xtest,1)

                x = Xtest(w,:)';

                % Original measurements
                y = PhiCurrent*x;

                % Generate packet loss
                packetLost = rand(packetCount,1) ...
                    < lossProbability;

                lostPacketsTotal = lostPacketsTotal ...
                    + sum(packetLost);

                % Map packet losses to measurement indices
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
                    failedBlocks = failedBlocks+1;

                else

                    % Received measurement rows
                    yReceived = y(receivedIndices);

                    Dreceived = Dnorm(receivedIndices,:);

                    % DCT-OMP using received measurements
                    xhat = ompDCTLoss( ...
                        yReceived,Dreceived,Psi, ...
                        columnNorms,K);

                    successfulBlocks = successfulBlocks+1;
                end

                % Reconstruction metrics
                err = x-xhat;

                signalPower = sum(x.^2);
                errorPower = sum(err.^2);

                mseValues(w) = mean(err.^2);

                snrValues(w) = 10*log10( ...
                    signalPower/(errorPower+eps));

                prdValues(w) = 100*sqrt( ...
                    errorPower/(signalPower+eps));

                % Store window-level metrics
                windowResults = [windowResults; ...
                    {signalName,ratios(r),lossPercent, ...
                    w,mseValues(w),snrValues(w), ...
                    prdValues(w)}];

            end

            avgMSE = mean(mseValues);
            avgSNR = mean(snrValues);
            avgPRD = mean(prdValues);

            blockSuccessRate = ...
                successfulBlocks/size(Xtest,1)*100;

            theoreticalBlockSuccess = ...
                (1-lossProbability)^packetCount*100;

            avgLostPackets = ...
                lostPacketsTotal/size(Xtest,1);

            fprintf(['Loss=%2d%% | MSE=%.6f | ' ...
                'SNR=%.4f dB | PRD=%.4f%% | ' ...
                'Block success=%.2f%%\n'], ...
                lossPercent,avgMSE,avgSNR,avgPRD, ...
                blockSuccessRate);

            summaryResults = [summaryResults; ...
                {signalName,ratios(r),K,lossPercent, ...
                avgMSE,avgSNR,avgPRD,packetCount, ...
                avgLostPackets,blockSuccessRate, ...
                theoreticalBlockSuccess}];

        end
    end
end

%% 6. Convert and save tables
SummaryTable = cell2table(summaryResults, ...
    'VariableNames',{ ...
    'Signal','SamplingRatio','BestK','PacketLossPercent', ...
    'MSE','SNR_dB','PRD_percent','PacketCount', ...
    'AverageLostPackets','BlockSuccessPercent', ...
    'TheoreticalBlockSuccessPercent'});

WindowTable = cell2table(windowResults, ...
    'VariableNames',{ ...
    'Signal','SamplingRatio','PacketLossPercent', ...
    'Window','MSE','SNR_dB','PRD_percent'});

writetable(SummaryTable, ...
    fullfile(outDir,'OMP_DCT_Packet_Loss_Summary.csv'));

writetable(WindowTable, ...
    fullfile(outDir,'OMP_DCT_Packet_Loss_WindowResults.csv'));

save(fullfile(outDir,'OMP_DCT_Packet_Loss.mat'), ...
    'SummaryTable','WindowTable','bestK','lossRates');

%% 7. Plot MSE versus packet loss
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
    title([signalName ' DCT-OMP MSE under Packet Loss']);
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
    title([signalName ' DCT-OMP SNR under Packet Loss']);
    legend('Location','best');
    hold off;

    saveas(gcf,fullfile(outDir, ...
        [signalName '_OMP_DCT_Packet_Loss.png']));
end

%% 8. Completion checks
fprintf('\n====================================\n');
fprintf('DCT-OMP PACKET LOSS COMPLETED\n');
fprintf('Summary configurations: %d\n',height(SummaryTable));
fprintf('Window evaluations: %d\n',height(WindowTable));

fprintf('NaN MSE: %d\n',sum(isnan(WindowTable.MSE)));
fprintf('Inf MSE: %d\n',sum(isinf(WindowTable.MSE)));

fprintf('Results saved in:\n%s\n',outDir);
fprintf('====================================\n');

%% 9. DCT-OMP reconstruction function
function xhat = ompDCTLoss(y,Dnorm,Psi,columnNorms,K)

    residual = y;
    support = [];
    beta = [];

    maxIterations = min(K,size(Dnorm,1)-1);

    for iteration = 1:maxIterations

        correlations = Dnorm'*residual;

        if ~isempty(support)
            correlations(support) = 0;
        end

        [~,selected] = max(abs(correlations));

        support = [support selected];

        A = Dnorm(:,support);

        beta = A\y;

        residual = y-A*beta;

        if norm(residual) < 1e-8
            break;
        end
    end

    alpha = zeros(size(Dnorm,2),1);

    if ~isempty(support)
        alpha(support) = beta./columnNorms(support)';
    end

    xhat = Psi*alpha;
end
