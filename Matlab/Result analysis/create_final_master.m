
clc;
clear;

% Run this script from the MATLAB project folder.

projectRoot = '..\results';
sourceReconstruction = fullfile(projectRoot, '08_Final_Results');
sourceCommunication = fullfile('results');

destination = fullfile(sourceReconstruction, 'Final_Master');

% Check source folders
if ~isfolder(sourceReconstruction)
    error('Reconstruction source folder not found: %s', sourceReconstruction);
end

if ~isfolder(sourceCommunication)
    error('MATLAB results folder not found: %s', sourceCommunication);
end

% Create destination folders
if ~isfolder(destination)
    mkdir(destination);
end

reconstructionFolder = fullfile(destination, ...
    '01_Reconstruction_Results');

communicationFolder = fullfile(destination, ...
    '02_Communication_and_PacketLoss');

archiveFolder = fullfile(destination, ...
    '99_Archive_Not_For_Publication');

mkdir(reconstructionFolder);
mkdir(communicationFolder);
mkdir(archiveFolder);

%% 1. Copy reconstruction results safely
fprintf('\nCopying reconstruction results...\n');

items = dir(sourceReconstruction);

for i = 1:numel(items)

    name = items(i).name;

    if strcmp(name, '.') || strcmp(name, '..') || ...
            strcmp(name, 'Final_Master')
        continue;
    end

    sourceItem = fullfile(sourceReconstruction, name);

    copyfile(sourceItem, reconstructionFolder);
end

%% 2. Copy communication and valid packet-loss experiments
fprintf('\nCopying communication experiments...\n');

folders = {
    'Communication_Analysis'
    'Communication_Analysis_Corrected'
    'Communication_Latency'
    'Packetization_Analysis'
    'Packet_Loss_Analysis'
    'Packet_Loss_Reconstruction'
    'OMP_DCT_Diagnostic'
    'OMP_DCT_Packet_Loss'
    'LASSO_DCT_Packet_Loss'
    'CNN_Packet_Loss'
};

for i = 1:numel(folders)

    sourceFolder = fullfile(sourceCommunication, folders{i});

    if isfolder(sourceFolder)
        copyfile(sourceFolder, communicationFolder);
        fprintf('Copied: %s\n', folders{i});
    else
        fprintf('Not found, skipped: %s\n', folders{i});
    end
end

%% 3. Archive the invalid original OMP packet-loss experiment
fprintf('\nArchiving invalid original OMP experiment...\n');

invalidOMP = fullfile(sourceCommunication, 'OMP_Packet_Loss');

if isfolder(invalidOMP)
    copyfile(invalidOMP, ...
        fullfile(archiveFolder, 'OMP_Packet_Loss_Invalid_Original'));
end

%% 4. Generate a file manifest
fprintf('\nGenerating file manifest...\n');

allFiles = dir(fullfile(destination, '**', '*'));
allFiles = allFiles(~[allFiles.isdir]);

fileNames = strings(numel(allFiles), 1);
filePaths = strings(numel(allFiles), 1);
fileTypes = strings(numel(allFiles), 1);

for i = 1:numel(allFiles)

    fileNames(i) = string(allFiles(i).name);
    filePaths(i) = string(fullfile(allFiles(i).folder, ...
        allFiles(i).name));

    [~,~,ext] = fileparts(allFiles(i).name);
    fileTypes(i) = string(ext);
end

Manifest = table(fileNames, fileTypes, filePaths, ...
    'VariableNames', {'FileName','FileType','FullPath'});

writetable(Manifest, ...
    fullfile(destination, '03_Results_Manifest.csv'));

%% 5. Print summary
fprintf('\n========================================\n');
fprintf('FINAL MASTER FOLDER CREATED SUCCESSFULLY\n');
fprintf('========================================\n');

fprintf('Destination: %s\n', destination);
fprintf('Total copied files: %d\n', height(Manifest));
fprintf('CSV files: %d\n', sum(fileTypes == ".csv"));
fprintf('MAT files: %d\n', sum(fileTypes == ".mat"));
fprintf('PNG files: %d\n', sum(fileTypes == ".png"));

fprintf('\nOriginal results have not been modified.\n');
fprintf('Invalid original OMP packet-loss results are archived separately.\n');
fprintf('\nCONSOLIDATION COMPLETED!\n');
