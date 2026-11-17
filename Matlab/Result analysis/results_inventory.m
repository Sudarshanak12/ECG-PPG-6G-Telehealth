
clc;
clear;

fprintf('\n====================================\n');
fprintf('PROJECT RESULTS INVENTORY\n');
fprintf('====================================\n\n');

rootFolder = 'results';

files = dir(fullfile(rootFolder, '**', '*'));

files = files(~[files.isdir]);

FileName = strings(length(files),1);
Folder = strings(length(files),1);
Extension = strings(length(files),1);
FileSizeKB = zeros(length(files),1);

for i = 1:length(files)

    FileName(i) = string(files(i).name);

    Folder(i) = string(files(i).folder);

    [~,~,ext] = fileparts(files(i).name);

    Extension(i) = string(ext);

    FileSizeKB(i) = files(i).bytes / 1024;

end

Inventory = table( ...
    Folder, FileName, Extension, FileSizeKB);

disp(Inventory);

% Count file types
fprintf('\nFILE TYPE SUMMARY\n');

uniqueExtensions = unique(Extension);

for i = 1:length(uniqueExtensions)

    count = sum(Extension == uniqueExtensions(i));

    fprintf('%s : %d files\n', ...
        uniqueExtensions(i), count);

end

% Save inventory
writetable(Inventory, ...
    fullfile(rootFolder,'Project_Results_Inventory.csv'));

fprintf('\nInventory saved successfully!\n');
