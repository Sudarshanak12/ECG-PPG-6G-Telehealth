
clc;
clear;
close all;

fprintf('\n====================================\n');
fprintf(' COMMUNICATION LATENCY SIMULATION\n');
fprintf('====================================\n');

%% 1. Load Packetization Results

input_file = fullfile('results', ...
    'Packetization_Analysis', ...
    'Packetization_Results.csv');

T = readtable(input_file);

%% 2. Simulation Parameters

bandwidth_kbps = [10 50 100];
propagation_delay_ms = 20;

% Illustrative processing-delay assumptions
pcm_processing_ms = 1;
dpcm_processing_ms = 2;
cs_processing_ms = 5;

%% 3. Initialize Results

Results = table();

%% 4. Calculate Latency

for i = 1:height(T)

    method = string(T.Method(i));

    total_bytes = T.Total_Transmitted_Bytes(i);

    % Select assumed processing delay
    if method == "PCM"
        processing_delay = pcm_processing_ms;
    elseif method == "DPCM"
        processing_delay = dpcm_processing_ms;
    else
        processing_delay = cs_processing_ms;
    end

    for j = 1:length(bandwidth_kbps)

        bandwidth_bps = bandwidth_kbps(j) * 1000;

        % Transmission delay
        transmission_delay_ms = ...
            (total_bytes * 8 / bandwidth_bps) * 1000;

        % Total end-to-end latency
        total_latency_ms = ...
            transmission_delay_ms + ...
            propagation_delay_ms + ...
            processing_delay;

        new_row = table( ...
            method, ...
            T.Bit_Depth(i), ...
            T.Sampling_Ratio(i), ...
            bandwidth_kbps(j), ...
            total_bytes, ...
            transmission_delay_ms, ...
            propagation_delay_ms, ...
            processing_delay, ...
            total_latency_ms, ...
            'VariableNames', { ...
            'Method', ...
            'Bit_Depth', ...
            'Sampling_Ratio', ...
            'Bandwidth_kbps', ...
            'Total_Bytes', ...
            'Transmission_Delay_ms', ...
            'Propagation_Delay_ms', ...
            'Processing_Delay_ms', ...
            'Total_Latency_ms'});

        Results = [Results; new_row];

    end
end

%% 5. Save Results

output_folder = fullfile('results', ...
    'Communication_Latency');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

writetable(Results, ...
    fullfile(output_folder, ...
    'Communication_Latency_Results.csv'));

save(fullfile(output_folder, ...
    'Communication_Latency_Results.mat'), ...
    'Results');

%% 6. Display Results

disp('LATENCY RESULTS');
disp(Results);


%% 7. Plot Total Latency - Corrected

figure('Name', 'Total Communication Latency');

methods = ["PCM", "DPCM", "CS"];

for i = 1:length(methods)

    subplot(1, 3, i);

    idx = Results.Method == methods(i);
    groups = Results(idx, :);

    if methods(i) == "CS"
        configs = unique(groups.Sampling_Ratio, 'stable');
        labels = "CS " + string(configs) + "%";
        config_column = groups.Sampling_Ratio;
    else
        configs = unique(groups.Bit_Depth, 'stable');
        labels = methods(i) + " " + string(configs) + "-bit";
        config_column = groups.Bit_Depth;
    end

    hold on;

    for b = 1:length(bandwidth_kbps)

        latency_values = zeros(length(configs), 1);

        for k = 1:length(configs)

            if methods(i) == "CS"
                selected = config_column == configs(k) & ...
                    groups.Bandwidth_kbps == bandwidth_kbps(b);
            else
                selected = config_column == configs(k) & ...
                    groups.Bandwidth_kbps == bandwidth_kbps(b);
            end

            latency_values(k) = ...
                groups.Total_Latency_ms(selected);
        end

        plot(1:length(configs), latency_values, ...
            '-o', 'LineWidth', 1.5, ...
            'MarkerSize', 7);

    end

    title(methods(i) + " Latency");
    ylabel('Latency (ms)');
    xlabel('Configuration');

    xticks(1:length(configs));
    xticklabels(labels);
    xtickangle(30);

    grid on;
    legend("10 kbps", "50 kbps", "100 kbps", ...
        'Location', 'best');

    hold off;

end

sgtitle('Communication Latency Comparison');

exportgraphics(gcf, ...
    fullfile(output_folder, ...
    'Total_Communication_Latency.png'), ...
    'Resolution', 300);
