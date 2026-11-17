
clc;
clear;
close all;

%% 1. PROJECT SETUP

project_path = 'D:\sudarshan\dsu\sem\sem 5\adc project\matlab';
cd(project_path);

load('ECG_PPG_Data.mat');
load('Measurement_Matrices.mat');

if ~exist('models','dir')
    mkdir('models');
end

%% 2. PARAMETERS

sampling_ratios = [0.10 0.20 0.30 0.40 0.50];

signals = {'ECG','PPG'};

train_data = {double(train_ecg), double(train_ppg)};
val_data = {double(val_ecg), double(val_ppg)};

fs = double(fs);
N = double(window_length);

num_filters = 32;
num_blocks = 4;

max_epochs = 20;
mini_batch_size = 64;
learning_rate = 1e-3;

%% 3. TRAINING LOG

training_log = {};

fprintf('\n============================================\n');
fprintf(' AUTOMATED RESIDUAL CNN TRAINING\n');
fprintf('============================================\n');

%% 4. AUTOMATED TRAINING LOOP

for s = 1:2

    signal_name = signals{s};

    XTrainOriginal = train_data{s};
    XValOriginal = val_data{s};

    for r = 1:length(sampling_ratios)

        ratio = sampling_ratios(r);

        model_name = sprintf('%s_CNN_%02dpct', ...
            signal_name,round(ratio*100));

        model_path = fullfile('models',[model_name '.mat']);

        fprintf('\n--------------------------------------------\n');
        fprintf('Signal: %s | Sampling: %.0f%%\n', ...
            signal_name,ratio*100);
        fprintf('--------------------------------------------\n');

        %% Skip existing trained model

        if exist(model_path,'file')

            fprintf('Existing model found. Skipping training.\n');

            training_log(end+1,:) = ...
                {signal_name,ratio*100,'Already saved'};

            continue;

        end

        %% 5. SELECT MEASUREMENT MATRIX

        A = double(Phi{r});

        %% 6. PREPARE TRAINING SEQUENCES

        fprintf('Preparing training sequences...\n');

        XTrain = cell(size(XTrainOriginal,1),1);
        YTrain = cell(size(XTrainOriginal,1),1);

        for i = 1:size(XTrainOriginal,1)

            x = XTrainOriginal(i,:)';

            y = A*x;

            x_bp = A'*y;

            XTrain{i} = single(x_bp');
            YTrain{i} = single(x');

        end

        %% 7. PREPARE VALIDATION SEQUENCES

        fprintf('Preparing validation sequences...\n');

        XVal = cell(size(XValOriginal,1),1);
        YVal = cell(size(XValOriginal,1),1);

        for i = 1:size(XValOriginal,1)

            x = XValOriginal(i,:)';

            y = A*x;

            x_bp = A'*y;

            XVal{i} = single(x_bp');
            YVal{i} = single(x');

        end

        %% 8. BUILD RESIDUAL CNN

        lgraph = layerGraph();

        input_layer = sequenceInputLayer(1, ...
            'Normalization','none','Name','input');

        lgraph = addLayers(lgraph,input_layer);

        stem_conv = convolution1dLayer(7,num_filters, ...
            'Padding','same','Name','stem_conv');

        stem_relu = reluLayer('Name','stem_relu');

        lgraph = addLayers(lgraph,stem_conv);
        lgraph = addLayers(lgraph,stem_relu);

        lgraph = connectLayers(lgraph,'input','stem_conv');
        lgraph = connectLayers(lgraph,'stem_conv','stem_relu');

        previous_layer = 'stem_relu';

        for b = 1:num_blocks

            block_name = sprintf('block%d',b);

            [lgraph,previous_layer] = addResidualBlock( ...
                lgraph,previous_layer,block_name,num_filters);

        end

        %% 9. OUTPUT CORRECTION AND SKIP CONNECTION

        output_conv = convolution1dLayer(7,1, ...
            'Padding','same','Name','output_conv');

        output_add = additionLayer(2,'Name','output_add');

        regression_output = regressionLayer( ...
            'Name','regression_output');

        lgraph = addLayers(lgraph,output_conv);
        lgraph = addLayers(lgraph,output_add);
        lgraph = addLayers(lgraph,regression_output);

        lgraph = connectLayers(lgraph, ...
            previous_layer,'output_conv');

        lgraph = connectLayers(lgraph, ...
            'output_conv','output_add/in1');

        lgraph = connectLayers(lgraph, ...
            'input','output_add/in2');

        lgraph = connectLayers(lgraph, ...
            'output_add','regression_output');

        %% 10. TRAINING OPTIONS

        options = trainingOptions('adam', ...
            'MaxEpochs',max_epochs, ...
            'MiniBatchSize',mini_batch_size, ...
            'InitialLearnRate',learning_rate, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{XVal,YVal}, ...
            'ValidationFrequency',200, ...
            'Verbose',true, ...
            'Plots','none');

        %% 11. TRAIN NETWORK

        fprintf('\nStarting training: %s\n',model_name);

        [net,info] = trainNetwork( ...
            XTrain,YTrain,lgraph,options);

        %% 12. VALIDATION EVALUATION

        prediction = predict(net,XVal(1));

        x_original = double(XValOriginal(1,:)');
        x_reconstructed = double(prediction{1}(:));

        validation_mse = mean( ...
            (x_original-x_reconstructed).^2);

        fprintf('\nModel: %s\n',model_name);
        fprintf('Validation MSE: %.6f\n',validation_mse);

        %% 13. SAVE MODEL IMMEDIATELY

        save(model_path, ...
            'net','info','ratio','signal_name', ...
            'validation_mse','num_filters','num_blocks', ...
            'max_epochs','mini_batch_size','learning_rate', ...
            '-v7.3');

        fprintf('MODEL SAVED: %s\n',model_path);

        training_log(end+1,:) = ...
            {signal_name,ratio*100,'Completed'};

        %% 14. SAVE TRAINING LOG AFTER EACH MODEL

        log_table = cell2table(training_log, ...
            'VariableNames', ...
            {'Signal','Sampling_Ratio','Status'});

        writetable(log_table, ...
            'models/CNN_Training_Log.csv');

        clear net info XTrain YTrain XVal YVal lgraph;

    end
end

%% 15. FINAL TRAINING SUMMARY

fprintf('\n============================================\n');
fprintf(' AUTOMATED CNN TRAINING FINISHED\n');
fprintf('============================================\n');

disp(log_table);

fprintf('\nModels stored in the models folder.\n');

%% 16. RESIDUAL BLOCK FUNCTION

function [lgraph,output_name] = addResidualBlock( ...
    lgraph,input_name,block_name,num_filters)

    conv1 = [block_name '_conv1'];
    bn1 = [block_name '_bn1'];
    relu1 = [block_name '_relu1'];

    conv2 = [block_name '_conv2'];
    bn2 = [block_name '_bn2'];

    add_name = [block_name '_add'];
    relu_out = [block_name '_out'];

    lgraph = addLayers(lgraph, ...
        convolution1dLayer(5,num_filters, ...
        'Padding','same','Name',conv1));

    lgraph = addLayers(lgraph, ...
        batchNormalizationLayer('Name',bn1));

    lgraph = addLayers(lgraph, ...
        reluLayer('Name',relu1));

    lgraph = addLayers(lgraph, ...
        convolution1dLayer(5,num_filters, ...
        'Padding','same','Name',conv2));

    lgraph = addLayers(lgraph, ...
        batchNormalizationLayer('Name',bn2));

    lgraph = addLayers(lgraph, ...
        additionLayer(2,'Name',add_name));

    lgraph = addLayers(lgraph, ...
        reluLayer('Name',relu_out));

    % Main path
    lgraph = connectLayers(lgraph,input_name,conv1);
    lgraph = connectLayers(lgraph,conv1,bn1);
    lgraph = connectLayers(lgraph,bn1,relu1);
    lgraph = connectLayers(lgraph,relu1,conv2);
    lgraph = connectLayers(lgraph,conv2,bn2);
    lgraph = connectLayers(lgraph,bn2,[add_name '/in1']);

    % Residual skip path
    lgraph = connectLayers(lgraph,input_name,[add_name '/in2']);

    lgraph = connectLayers(lgraph,add_name,relu_out);

    output_name = relu_out;

end
