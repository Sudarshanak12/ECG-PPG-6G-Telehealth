
clc;
clear;
close all;

fprintf('\n========================================\n');
fprintf(' PACKET LOSS AND RELIABILITY SIMULATION\n');
fprintf('========================================\n');

%% 1. LOAD PACKETIZATION RESULTS

input_file = fullfile('results', ...
    'Packetization_Analysis', ...
    'Packetization_Results.csv');

if ~isfile(input_file)
    error('Packetization_Results.csv not found. Check the input folder.');
end

T = readtable(input_file);

%% 2. SIMULATION PARAMETERS

rng(42);

packet_loss_rates = [0 0.01 0.03 0.05 0.10];

num_simulations = 10000;

methods = ["PCM", "DPCM", "CS"];

%% 3. INITIALIZE RESULTS TABLE

Results = table();

%% 4. MONTE CARLO PACKET LOSS SIMULATION

for i = 1:height(T)

    method = string(T.Method(i));

    bit_depth = T.Bit_Depth(i);

    sampling_ratio = T.Sampling_Ratio(i);

    packet_count = T.Packet_Count(i);

    payload_bytes = T.Payload_Bytes(i);

    bytes_per_packet = payload_bytes / packet_count;

    for j = 1:length(packet_loss_rates)

        p = packet_loss_rates(j);

        % Simulate packet delivery
        packet_received = ...
            rand(num_simulations, packet_count) >= p;

        % Block succeeds only if all packets arrive
        block_success = all(packet_received, 2);

        % Calculate reliability
        success_percent = mean(block_success) * 100;

        failure_percent = 100 - success_percent;

        % Calculate lost packets
        lost_packets = sum(~packet_received, 2);

        average_lost_packets = mean(lost_packets);

        % Calculate received payload
        received_bytes = ...
            sum(packet_received, 2) * bytes_per_packet;

        average_received_bytes = mean(received_bytes);

        % Theoretical probability of complete block delivery
        theoretical_success = ...
            (1-p)^packet_count * 100;

        % Store results
        new_row = table( ...
            method, ...
            bit_depth, ...
            sampling_ratio, ...
            packet_count, ...
            p*100, ...
            success_percent, ...
            failure_percent, ...
            average_lost_packets, ...
            average_received_bytes, ...
            theoretical_success, ...
            'VariableNames', { ...
            'Method', ...
            'Bit_Depth', ...
            'Sampling_Ratio', ...
            'Packet_Count', ...
            'Packet_Loss_Percent', ...
            'Block_Success_Percent', ...
            'Block_Failure_Percent', ...
            'Average_Lost_Packets', ...
            'Average_Received_Bytes', ...
            'Theoretical_Success_Percent'});

        Results = [Results; new_row];

    end
end

%% 5. CREATE OUTPUT FOLDER

output_folder = fullfile('results', ...
    'Packet_Loss_Analysis');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% 6. SAVE RESULTS

writetable(Results, ...
    fullfile(output_folder, ...
    'Packet_Loss_Results.csv'));

save(fullfile(output_folder, ...
    'Packet_Loss_Results.mat'), ...
    'Results');

%% 7. DISPLAY RESULTS

fprintf('\nPACKET LOSS RESULTS\n\n');

disp(Results);

%% 8. PLOT RELIABILITY RESULTS

figure('Name', 'Packet Loss Analysis', ...
    'Color', 'w', ...
    'Position', [100 100 1500 550]);

for i = 1:length(methods)

    subplot(1, 3, i);

    method = methods(i);

    idx = Results.Method == method;

    groups = Results(idx, :);

    hold on;

    for j = 1:length(packet_loss_rates)

        loss_percent = packet_loss_rates(j) * 100;

        selected = groups.Packet_Loss_Percent == loss_percent;

        data = groups(selected, :);

        % Sort configurations
        if method == "CS"

            [x_values, order] = sort(data.Sampling_Ratio);

        else

            [x_values, order] = sort(data.Bit_Depth);

        end

        data = data(order, :);

        % Plot block failure percentage
        plot(x_values, ...
            data.Block_Failure_Percent, ...
            '-o', ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6);

    end

    % Correct x-axis labels
    if method == "CS"

        xlabel('Sampling Ratio (%)');

    else

        xlabel('Bit Depth');

    end

    ylabel('Block Failure (%)');

    title(method + " Reliability");

    grid on;

    ylim([0 100]);

    legend("0% Loss", ...
        "1% Loss", ...
        "3% Loss", ...
        "5% Loss", ...
        "10% Loss", ...
        'Location', 'best');

    hold off;

end

sgtitle('Packet Loss Impact on Block Delivery');

%% 9. SAVE GRAPH

exportgraphics(gcf, ...
    fullfile(output_folder, ...
    'Packet_Loss_Analysis.png'), ...
    'Resolution', 300);

%% 10. VERIFY SIMULATION AGAINST THEORY

fprintf('\n========================================\n');
fprintf(' THEORETICAL VALIDATION\n');
fprintf('========================================\n');

difference = abs( ...
    Results.Block_Success_Percent - ...
    Results.Theoretical_Success_Percent);

fprintf('Maximum absolute error: %.4f percentage points\n', ...
    max(difference));

fprintf('Total simulation cases: %d\n', height(Results));

fprintf('Monte Carlo trials per case: %d\n', ...
    num_simulations);

%% 11. COMPLETION

fprintf('\n========================================\n');
fprintf(' PACKET LOSS SIMULATION COMPLETED\n');
fprintf('========================================\n');

fprintf('Output folder:\n%s\n', output_folder);

fprintf('\nGenerated files:\n');
fprintf('1. Packet_Loss_Results.csv\n');
fprintf('2. Packet_Loss_Results.mat\n');
fprintf('3. Packet_Loss_Analysis.png\n');

fprintf('\n========================================\n');
