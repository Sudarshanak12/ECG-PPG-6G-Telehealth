
clc;
clear;
close all;

%% =====================================================
% PACKETIZATION AND COMMUNICATION OVERHEAD SIMULATION
% ECG + PPG Wearable Telehealth
% ======================================================

%% 1. PARAMETERS

fs = 125;
N = 512;

payload_capacity = 64;  % Bytes per packet
header_bytes = 8;
crc_bytes = 2;

overhead_bytes = header_bytes + crc_bytes;

bit_depths = [8 10 12];
ratios = [10 20 30 40 50];

methods = {'PCM','DPCM','CS'};

%% 2. OUTPUT FOLDER

outputPath = fullfile(pwd,'results', ...
    'Packetization_Analysis');

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

%% 3. INITIALIZE RESULTS

results = {};

%% 4. PCM AND DPCM PACKETIZATION

for m = 1:2

    method = methods{m};

    for b = 1:length(bit_depths)

        bits = bit_depths(b);

        % One signal, one 512-sample block
        payload_bits = N * bits;

        payload_bytes = ceil(payload_bits / 8);

        packet_count = ceil( ...
            payload_bytes / payload_capacity);

        total_overhead = ...
            packet_count * overhead_bytes;

        total_transmitted = ...
            payload_bytes + total_overhead;

        overhead_percentage = ...
            100 * total_overhead / total_transmitted;

        combined_payload = 2 * payload_bytes;

        combined_packets = ...
            2 * packet_count;

        combined_transmitted = ...
            2 * total_transmitted;

        results = [results; ...
            {'ECG + PPG',method,bits,NaN, ...
            combined_payload,combined_packets, ...
            2*total_overhead,combined_transmitted, ...
            overhead_percentage}];

    end
end

%% 5. COMPRESSIVE SENSING PACKETIZATION

for r = 1:length(ratios)

    ratio = ratios(r);

    M = round(N * ratio / 100);

    measurement_bits = 16;

    payload_bits = M * measurement_bits;

    payload_bytes = ceil(payload_bits / 8);

    packet_count = ceil( ...
        payload_bytes / payload_capacity);

    total_overhead = ...
        packet_count * overhead_bytes;

    total_transmitted = ...
        payload_bytes + total_overhead;

    overhead_percentage = ...
        100 * total_overhead / total_transmitted;

    combined_payload = 2 * payload_bytes;

    combined_packets = 2 * packet_count;

    combined_transmitted = 2 * total_transmitted;

    results = [results; ...
        {'ECG + PPG','CS',16,ratio, ...
        combined_payload,combined_packets, ...
        2*total_overhead,combined_transmitted, ...
        overhead_percentage}];

end

%% 6. CREATE RESULTS TABLE

T = cell2table(results, ...
    'VariableNames', ...
    {'Signal','Method','Bit_Depth','Sampling_Ratio', ...
    'Payload_Bytes','Packet_Count','Overhead_Bytes', ...
    'Total_Transmitted_Bytes','Overhead_Percent'});

fprintf('\n====================================\n');
fprintf('PACKETIZATION RESULTS\n');
fprintf('====================================\n');

disp(T);

%% 7. CALCULATE EFFECTIVE TRANSMISSION RATE

block_duration = N / fs;

T.Effective_Rate_bps = ...
    (T.Total_Transmitted_Bytes * 8) / block_duration;

fprintf('\n====================================\n');
fprintf('EFFECTIVE TRANSMISSION RATES\n');
fprintf('====================================\n');

disp(T(:,{'Method','Bit_Depth','Sampling_Ratio', ...
    'Packet_Count','Payload_Bytes', ...
    'Total_Transmitted_Bytes','Effective_Rate_bps'}));

%% 8. SAVE RESULTS

writetable(T,fullfile(outputPath, ...
    'Packetization_Results.csv'));

save(fullfile(outputPath, ...
    'Packetization_Results.mat'),'T');

%% 9. GRAPH: PACKET OVERHEAD

figure('Color','w');

bar(categorical( ...
    strcat(string(T.Method),"_", ...
    string(T.Bit_Depth),"_", ...
    string(T.Sampling_Ratio))), ...
    T.Overhead_Percent);

ylabel('Packet Overhead (%)');
xlabel('Transmission Configuration');
title('Packetization Overhead Comparison');

grid on;
xtickangle(45);

exportgraphics(gcf,fullfile(outputPath, ...
    'Packet_Overhead.png'),'Resolution',300);

%% 10. GRAPH: EFFECTIVE TRANSMISSION RATE

figure('Color','w');

bar(categorical( ...
    strcat(string(T.Method),"_", ...
    string(T.Bit_Depth),"_", ...
    string(T.Sampling_Ratio))), ...
    T.Effective_Rate_bps);

ylabel('Effective Transmission Rate (bps)');
xlabel('Transmission Configuration');
title('Payload Rate vs Packetized Transmission Rate');

grid on;
xtickangle(45);

exportgraphics(gcf,fullfile(outputPath, ...
    'Effective_Transmission_Rate.png'), ...
    'Resolution',300);

%% 11. COMPLETION

fprintf('\n====================================\n');
fprintf('PACKETIZATION SIMULATION COMPLETED\n');
fprintf('====================================\n');

fprintf('Configurations: %d\n',height(T));
fprintf('Output folder:\n%s\n',outputPath);
