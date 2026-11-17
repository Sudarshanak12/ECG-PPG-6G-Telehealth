
clc;
clear;
close all;

fprintf('\n============================================\n');
fprintf(' OMP RECONSTRUCTION UNDER PACKET LOSS\n');
fprintf('============================================\n');

rng(42);

%% 1. LOAD DATASET

data = load('ECG_PPG_Data.mat');

test_ecg = data.test_ecg;
test_ppg = data.test_ppg;

N = data.window_length;

fprintf('\nDataset loaded successfully.\n');
fprintf('ECG test windows: %d\n', size(test_ecg,1));
fprintf('PPG test windows: %d\n', size(test_ppg,1));

%% 2. LOAD MEASUREMENT MATRICES

matrix_data = load('Measurement_Matrices.mat');

Phi_all = matrix_data.Phi;
sampling_ratios = matrix_data.sampling_ratios;

fprintf('Measurement matrices loaded.\n');

%% 3. EXPERIMENT PARAMETERS

packet_loss_rates = [0 0.01 0.03 0.05 0.10];

packet_payload_bytes = 64;
measurement_bits = 16;

measurements_per_packet = ...
    packet_payload_bytes * 8 / measurement_bits;

max_sparsity = 40;

signal_names = ["ECG"; "PPG"];

num_windows = size(test_ecg,1);

Results = table();

%% 4. MAIN EXPERIMENT

for s = 1:2

    if s == 1
        Xtest = test_ecg;
    else
        Xtest = test_ppg;
    end

    fprintf('\n============================================\n');
    fprintf('PROCESSING %s\n', signal_names(s));
    fprintf('============================================\n');

    for r = 1:length(sampling_ratios)

        Phi = Phi_all{r};

        M = size(Phi,1);

        packet_count = ceil( ...
            M / measurements_per_packet);

        fprintf('\nSampling ratio: %.0f%%\n', ...
            sampling_ratios(r)*100);

        fprintf('Measurements: %d\n', M);

        for p = 1:length(packet_loss_rates)

            loss_rate = packet_loss_rates(p);

            mse_values = zeros(num_windows,1);
            snr_values = zeros(num_windows,1);
            prd_values = zeros(num_windows,1);

            successful_windows = 0;
            actual_sparsities = zeros(num_windows,1);

            for w = 1:num_windows

                %% ORIGINAL SIGNAL

                x = Xtest(w,:)';

                %% CS MEASUREMENTS

                y = Phi*x;

                %% SIMULATE PACKET LOSS

                packet_received = ...
                    rand(packet_count,1) >= loss_rate;

                measurement_received = false(M,1);

                for packet = 1:packet_count

                    first_index = ...
                        (packet-1)*measurements_per_packet+1;

                    last_index = min( ...
                        packet*measurements_per_packet,M);

                    if packet_received(packet)

                        measurement_received( ...
                            first_index:last_index) = true;

                    end

                end

                %% SELECT RECEIVED MEASUREMENTS

                A = Phi(measurement_received,:);
                y_received = y(measurement_received);

                %% OMP RECONSTRUCTION

                x_reconstructed = zeros(N,1);

                num_received = size(A,1);

                if num_received >= 3

                    % Normalize dictionary columns
                    column_norms = sqrt(sum(A.^2,1));

                    valid_columns = column_norms > 1e-12;

                    A = A(:,valid_columns);
                    column_norms = column_norms(valid_columns);

                    dictionary = A ./ column_norms;

                    % Adaptive sparsity:
                    % Keep the number of selected atoms
                    % below half the available measurements.
                    K = min(max_sparsity, ...
                        floor((num_received-1)/2));

                    residual = y_received;
                    support = [];

                    coefficients = [];

                    for iteration = 1:K

                        correlations = ...
                            abs(dictionary' * residual);

                        correlations(support) = 0;

                        [~, selected_atom] = max(correlations);

                        support = [support selected_atom];

                        selected_dictionary = ...
                            dictionary(:,support);

                        coefficients = ...
                            selected_dictionary \ y_received;

                        residual = ...
                            y_received - ...
                            selected_dictionary*coefficients;

                        if norm(residual) < ...
                                1e-6*max(norm(y_received),eps)
                            break;
                        end

                    end

                    % Recover original signal-domain coefficients
                    valid_indices = find(valid_columns);

                    selected_original_indices = ...
                        valid_indices(support);

                    x_reconstructed( ...
                        selected_original_indices) = ...
                        coefficients ./ ...
                        column_norms(support)';

                    actual_sparsities(w) = length(support);

                end

                %% PERFORMANCE METRICS

                error_signal = x-x_reconstructed;

                mse_values(w) = mean(error_signal.^2);

                signal_power = mean(x.^2);
                noise_power = mean(error_signal.^2);

                if noise_power < 1e-12
                    snr_values(w) = 120;
                else
                    snr_values(w) = ...
                        10*log10(signal_power/noise_power);
                end

                signal_energy = sum(x.^2);

                if signal_energy < 1e-12
                    prd_values(w) = 0;
                else
                    prd_values(w) = ...
                        sqrt(sum(error_signal.^2)/ ...
                        signal_energy)*100;
                end

                if all(packet_received)
                    successful_windows = ...
                        successful_windows+1;
                end

            end

            %% AGGREGATE METRICS

            avg_mse = mean(mse_values);
            avg_snr = mean(snr_values);
            avg_prd = mean(prd_values);

            delivery_percent = ...
                successful_windows/num_windows*100;

            theoretical_delivery = ...
                (1-loss_rate)^packet_count*100;

            average_sparsity = mean(actual_sparsities);

            %% SAVE RESULT ROW

            new_row = table( ...
                signal_names(s), ...
                sampling_ratios(r)*100, ...
                M, ...
                loss_rate*100, ...
                packet_count, ...
                average_sparsity, ...
                avg_mse, ...
                avg_snr, ...
                avg_prd, ...
                delivery_percent, ...
                theoretical_delivery, ...
                'VariableNames', { ...
                'Signal', ...
                'Sampling_Ratio', ...
                'Measurements', ...
                'Packet_Loss_Percent', ...
                'Packet_Count', ...
                'Average_Sparsity', ...
                'Average_MSE', ...
                'Average_SNR_dB', ...
                'Average_PRD_Percent', ...
                'Block_Delivery_Percent', ...
                'Theoretical_Delivery_Percent'});

            Results = [Results; new_row];

            fprintf(['Loss: %2.0f%% | MSE: %.5f | ' ...
                'SNR: %.3f dB | PRD: %.3f%%\n'], ...
                loss_rate*100, avg_mse, avg_snr, avg_prd);

        end
    end
end

%% 5. SAVE RESULTS

output_folder = fullfile('results', ...
    'OMP_Packet_Loss');

if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

writetable(Results, ...
    fullfile(output_folder, ...
    'OMP_Packet_Loss_Results.csv'));

save(fullfile(output_folder, ...
    'OMP_Packet_Loss_Results.mat'), ...
    'Results');

%% 6. DISPLAY RESULTS

fprintf('\n============================================\n');
fprintf(' OMP RESULTS\n');
fprintf('============================================\n');

disp(Results);

%% 7. PLOT RESULTS

figure('Color','w', ...
    'Position',[100 100 1450 850]);

metrics = { ...
    'Average_MSE','MSE'; ...
    'Average_SNR_dB','SNR (dB)'; ...
    'Average_PRD_Percent','PRD (%)'};

for m = 1:3

    subplot(2,3,m);
    hold on;

    for r = 1:length(sampling_ratios)

        idx = Results.Signal == "ECG" & ...
            Results.Sampling_Ratio == ...
            sampling_ratios(r)*100;

        group = Results(idx,:);

        plot(group.Packet_Loss_Percent, ...
            group.(metrics{m,1}), ...
            '-o','LineWidth',1.5);

    end

    xlabel('Packet Loss (%)');
    ylabel(metrics{m,2});
    title("ECG OMP " + metrics{m,2});
    grid on;

    legend("10% CS","20% CS","30% CS", ...
        "40% CS","50% CS", ...
        'Location','best');

    hold off;

end

for m = 1:3

    subplot(2,3,m+3);
    hold on;

    for r = 1:length(sampling_ratios)

        idx = Results.Signal == "PPG" & ...
            Results.Sampling_Ratio == ...
            sampling_ratios(r)*100;

        group = Results(idx,:);

        plot(group.Packet_Loss_Percent, ...
            group.(metrics{m,1}), ...
            '-o','LineWidth',1.5);

    end

    xlabel('Packet Loss (%)');
    ylabel(metrics{m,2});
    title("PPG OMP " + metrics{m,2});
    grid on;

    legend("10% CS","20% CS","30% CS", ...
        "40% CS","50% CS", ...
        'Location','best');

    hold off;

end

sgtitle('OMP Reconstruction Under Packet Loss');

%% 8. EXPORT GRAPH

exportgraphics(gcf, ...
    fullfile(output_folder, ...
    'OMP_Packet_Loss_Analysis.png'), ...
    'Resolution',300);

%% 9. COMPLETION

fprintf('\n============================================\n');
fprintf(' OMP PACKET LOSS SIMULATION COMPLETED\n');
fprintf('============================================\n');

fprintf('Total configurations: %d\n',height(Results));

fprintf('Test windows per signal: %d\n',num_windows);

fprintf('Output folder:\n%s\n',output_folder);

fprintf('\nGenerated files:\n');
fprintf('1. OMP_Packet_Loss_Results.csv\n');
fprintf('2. OMP_Packet_Loss_Results.mat\n');
fprintf('3. OMP_Packet_Loss_Analysis.png\n');

fprintf('\n============================================\n');
