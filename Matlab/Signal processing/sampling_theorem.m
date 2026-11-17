
clc;
clear;
close all;

fprintf('RUNNING UPDATED SAMPLING EXPERIMENT\n');

%% LOAD ECG DATA

projectRoot = fileparts(fileparts(mfilename('fullpath')));

filePath = fullfile(projectRoot, ...
    'Results', 'ECG_PPG_Data.mat');

data = load(filePath);

ecg = double(data.ecg(:));
time = double(data.time(:));
fs = double(data.fs);

%% SELECT FIRST 2 SECONDS

duration = 2;

N = round(duration * fs);

x = ecg(1:N);
t = time(1:N);

%% SAMPLING FREQUENCIES

sampling_rates = [125 40 20 10];

%% CREATE NEW FIGURE

figure('Name','UPDATED SAMPLING EXPERIMENT', ...
    'Color','w', ...
    'Position',[100 100 1300 750]);

for k = 1:length(sampling_rates)

    fs_new = sampling_rates(k);

    %% GENERATE SAMPLES

    ts = (t(1):1/fs_new:t(end))';

    xs = interp1(t,x,ts,'linear');

    %% RECONSTRUCTION

    xr = interp1(ts,xs,t,'linear','extrap');

    %% ERROR

    mse = mean((x-xr).^2);

    %% PLOT

    subplot(2,2,k);

    plot(t,x,'b-','LineWidth',1.5);
    hold on;

    plot(ts,xs,'ro', ...
        'MarkerSize',7, ...
        'MarkerFaceColor','r');

    plot(t,xr,'k--','LineWidth',1.5);

    title(sprintf('Sampling Frequency = %.1f Hz',fs_new));

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    legend('Original ECG','Sampling Points', ...
        'Reconstructed ECG','Location','best');

    xlim([0 duration]);

    grid on;

    fprintf('Fs = %.1f Hz | MSE = %.6f\n',fs_new,mse);

end

sgtitle('UPDATED ECG SAMPLING ANALYSIS');

drawnow;

%% SAVE RESULT

resultsFolder = fullfile(projectRoot,'Results');

saveas(gcf,fullfile(resultsFolder, ...
    'Updated_Sampling_Analysis.png'));

fprintf('\nUPDATED EXPERIMENT COMPLETED!\n');
