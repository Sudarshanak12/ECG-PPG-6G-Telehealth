
clc;
clear;
close all;

%% COMPRESSIVE SENSING MEASUREMENT MATRIX GENERATION

N = 512;

sampling_ratios = [0.10 0.20 0.30 0.40 0.50];

Phi = cell(length(sampling_ratios), 1);
M_values = zeros(length(sampling_ratios), 1);

rng(42);  % Reproducibility

fprintf('\nMEASUREMENT MATRIX GENERATION\n');

for i = 1:length(sampling_ratios)

    % Number of measurements
    M = round(sampling_ratios(i) * N);

    % Generate random Gaussian matrix
    A = randn(N, M);

    % QR decomposition gives orthonormal columns
    [Q, ~] = qr(A, 0);

    % Transpose to obtain orthonormal rows
    Phi{i} = Q';

    M_values(i) = M;

    fprintf('\nSampling Ratio: %.0f%%\n', ...
        sampling_ratios(i) * 100);

    fprintf('Matrix Dimensions: %d x %d\n', ...
        size(Phi{i}, 1), size(Phi{i}, 2));

    % Verify orthonormal rows
    error_value = norm( ...
        Phi{i} * Phi{i}' - eye(M), 'fro');

    fprintf('Orthonormality Error: %.3e\n', ...
        error_value);

end

%% SAVE MEASUREMENT MATRICES

save('Measurement_Matrices.mat', ...
    'Phi', 'sampling_ratios', 'M_values', 'N');

fprintf('\n====================================\n');
disp('ALL MEASUREMENT MATRICES GENERATED!');
disp('Saved as Measurement_Matrices.mat');
fprintf('====================================\n');
