%% Download the CIFAR-10 dataset
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end    

%% Prepare the CIFAR-10 dataset
if ~exist('cifar10','dir')
    disp('Saving the Images with class subfolders. This might take some time...');    
    saveCIFAR10WithClasses('cifar-10-batches-mat', pwd);
end

function saveCIFAR10WithClasses(inputPath, outputPath)

% Check input directory is valid
if(~isempty(inputPath))
    assert(exist(inputPath,'dir') == 7);
end
if(~isempty(outputPath))
    assert(exist(outputPath,'dir') == 7);
end

% Set name for the output directory
outputDirectoryName = 'cifar10';

% Create directory for the output
mkdir(fullfile(outputPath, outputDirectoryName));

% Set names for directories
labelNames = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
iMakeTheseDirectories(fullfile(outputPath, outputDirectoryName), labelNames);

% Load test batch
iLoadBatchAndWriteAsImagesToClassFolders(fullfile(inputPath,'test_batch.mat'), fullfile(outputPath, outputDirectoryName), labelNames, 0);

% Load training batches
for i = 1:5
    iLoadBatchAndWriteAsImagesToClassFolders(fullfile(inputPath,['data_batch_' num2str(i) '.mat']), fullfile(outputPath, outputDirectoryName), labelNames, (i-1)*10000);
end

end

function iLoadBatchAndWriteAsImagesToClassFolders(fullInputBatchPath, fullOutputDirectoryPath, labelNames, nameIndexOffset)
    load(fullInputBatchPath);
    data = data'; %#ok<NODEF>
    data = reshape(data, 32,32,3,[]);
    data = permute(data, [2 1 3 4]);
    for i = 1:size(data,4)
        classFolder = fullfile(fullOutputDirectoryPath, labelNames{labels(i)+1});
        imwrite(data(:,:,:,i), fullfile(classFolder, ['image' num2str(i + nameIndexOffset) '.png']));
    end
end

function iMakeTheseDirectories(outputPath, directoryNames)
    for i = 1:numel(directoryNames)
        mkdir(fullfile(outputPath, directoryNames{i}));
    end
end
