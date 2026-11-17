
clc;
clear;
close all;

%% 1. LOAD DATA

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

%% 2. PARAMETERS

fs = double(fs);
N = double(window_length);

sampling_ratio_index = 1;  % 10% sampling
Phi_current = double(Phi{sampling_ratio_index});

num_filters = 32;
K_blocks = 4;

max_epochs = 20;
mini_batch_size = 64;
learning_rate = 1e-3;

%% 3. PREPARE ECG TRAINING DATA

fprintf('\nPreparing ECG training sequences...\n');

XTrain = cell(size(train_ecg,1),1);
YTrain = cell(size(train_ecg,1),1);

for i = 1:size(train_ecg,1)

    x = double(train_ecg(i,:)');

    % Compressed measurements
    y = Phi_current * x;

    % Back Projection
    x_bp = Phi_current' * y;

    % CNN input: Back Projection reconstruction
    XTrain{i} = single(x_bp');

    % Target: Original ECG signal
    YTrain{i} = single(x');

end

%% 4. PREPARE VALIDATION DATA

fprintf('Preparing ECG validation sequences...\n');

XVal = cell(size(val_ecg,1),1);
YVal = cell(size(val_ecg,1),1);

for i = 1:size(val_ecg,1)

    x = double(val_ecg(i,:)');

    y = Phi_current * x;

    x_bp = Phi_current' * y;

    XVal{i} = single(x_bp');
    YVal{i} = single(x');

end

fprintf('Training windows: %d\n',numel(XTrain));
fprintf('Validation windows: %d\n',numel(XVal));

%% 5. BUILD 1D RESIDUAL CNN

% Input layer
lgraph = layerGraph();

input_layer = sequenceInputLayer(1, ...
    'Normalization','none', ...
    'Name','input');

lgraph = addLayers(lgraph,input_layer);

% Initial convolution
stem_conv = convolution1dLayer(7,num_filters, ...
    'Padding','same', ...
    'Name','stem_conv');

stem_relu = reluLayer('Name','stem_relu');

lgraph = addLayers(lgraph,stem_conv);
lgraph = addLayers(lgraph,stem_relu);

lgraph = connectLayers(lgraph,'input','stem_conv');
lgraph = connectLayers(lgraph,'stem_conv','stem_relu');

previous_layer = 'stem_relu';

% Four residual blocks
for b = 1:K_blocks

    block_name = sprintf('block%d',b);

    [lgraph,previous_layer] = addResidualBlock( ...
        lgraph,previous_layer,block_name,num_filters);

end

%% 6. OUTPUT CORRECTION LAYER

output_conv = convolution1dLayer(7,1, ...
    'Padding','same', ...
    'Name','output_conv');

output_add = additionLayer(2, ...
    'Name','output_add');

output_layer = regressionLayer( ...
    'Name','regression_output');

lgraph = addLayers(lgraph,output_conv);
lgraph = addLayers(lgraph,output_add);
lgraph = addLayers(lgraph,output_layer);

lgraph = connectLayers(lgraph, ...
    previous_layer,'output_conv');

lgraph = connectLayers(lgraph, ...
    'output_conv','output_add/in1');

% Add the original Back Projection input
lgraph = connectLayers(lgraph, ...
    'input','output_add/in2');

lgraph = connectLayers(lgraph, ...
    'output_add','regression_output');

%% 7. DISPLAY NETWORK

figure('Name','Residual CNN Architecture');

plot(lgraph);

title('1D Residual CNN for ECG Reconstruction');

%% 8. TRAINING OPTIONS

options = trainingOptions('adam', ...
    'MaxEpochs',max_epochs, ...
    'MiniBatchSize',mini_batch_size, ...
    'InitialLearnRate',learning_rate, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',200, ...
    'Verbose',true, ...
    'Plots','training-progress');

%% 9. TRAIN NETWORK

fprintf('\nStarting ECG Residual CNN training...\n');

net = trainNetwork(XTrain,YTrain,lgraph,options);

fprintf('\nECG CNN TRAINING COMPLETED!\n');

%% 10. VALIDATION PREDICTION

prediction = predict(net,XVal(1));

x_original = double(val_ecg(1,:)');
x_reconstructed = double(prediction{1}(:));

mse_value = mean((x_original-x_reconstructed).^2);

fprintf('First validation window MSE: %.6f\n',mse_value);

%% 11. PLOT RECONSTRUCTION

t = (0:N-1)'/fs;

figure('Name','ECG CNN Reconstruction','Color','w');

plot(t,x_original,'b','LineWidth',1.2);
hold on;

plot(t,x_reconstructed,'r--','LineWidth',1.2);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('ECG Residual CNN Reconstruction - 10% Sampling');

legend('Original ECG','CNN Reconstructed ECG');

grid on;

%% 12. SAVE MODEL

if ~exist('models','dir')
    mkdir('models');
end

save('models/ECG_CNN_10pct.mat', ...
    'net','sampling_ratio_index','learning_rate', ...
    'max_epochs','mini_batch_size','-v7.3');

fprintf('\nModel saved successfully!\n');
fprintf('models/ECG_CNN_10pct.mat\n');

%% 13. RESIDUAL BLOCK FUNCTION

function [lgraph,output_name] = addResidualBlock( ...
    lgraph,input_name,block_name,num_filters)

    conv1_name = [block_name '_conv1'];
    bn1_name   = [block_name '_bn1'];
    relu1_name = [block_name '_relu1'];

    conv2_name = [block_name '_conv2'];
    bn2_name   = [block_name '_bn2'];

    add_name   = [block_name '_add'];
    relu_name  = [block_name '_out'];

    % Create residual block layers
    lgraph = addLayers(lgraph, ...
        convolution1dLayer(5,num_filters, ...
        'Padding','same','Name',conv1_name));

    lgraph = addLayers(lgraph, ...
        batchNormalizationLayer('Name',bn1_name));

    lgraph = addLayers(lgraph, ...
        reluLayer('Name',relu1_name));

    lgraph = addLayers(lgraph, ...
        convolution1dLayer(5,num_filters, ...
        'Padding','same','Name',conv2_name));

    lgraph = addLayers(lgraph, ...
        batchNormalizationLayer('Name',bn2_name));

    lgraph = addLayers(lgraph, ...
        additionLayer(2,'Name',add_name));

    lgraph = addLayers(lgraph, ...
        reluLayer('Name',relu_name));

    % Main convolution path
    lgraph = connectLayers(lgraph,input_name,conv1_name);

    lgraph = connectLayers(lgraph,conv1_name,bn1_name);
    lgraph = connectLayers(lgraph,bn1_name,relu1_name);

    lgraph = connectLayers(lgraph,relu1_name,conv2_name);
    lgraph = connectLayers(lgraph,conv2_name,bn2_name);

    lgraph = connectLayers(lgraph,bn2_name, ...
        [add_name '/in1']);

    % Skip connection
    lgraph = connectLayers(lgraph,input_name, ...
        [add_name '/in2']);

    lgraph = connectLayers(lgraph,add_name,relu_name);

    output_name = relu_name;

end
