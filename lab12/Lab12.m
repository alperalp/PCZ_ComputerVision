% Lab12.m

digitDatasetPath = fullfile(matlabroot, "toolbox", "nnet", ...
    "nndemos", "nndatasets", "DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadSize = 500;
imds = shuffle(imds);
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.95, 0.025);

transformFcn = @(x) addNoise(x);
combineFcn = @(x, y) cat(3, x, y);

dsTrainNoisy = transform(imdsTrain, transformFcn);
dsValNoisy = transform(imdsVal, transformFcn);
dsTestNoisy = transform(imdsTest, transformFcn);

dsTrain = combine(dsTrainNoisy, imdsTrain);
dsVal = combine(dsValNoisy, imdsVal);
dsTest = combine(dsTestNoisy, imdsTest);

dsTrain = transform(dsTrain, @commonPreprocessing);
dsVal = transform(dsVal, @commonPreprocessing);
dsTest = transform(dsTest, @commonPreprocessing);

dsTrain = transform(dsTrain, @augmentImages);

imageLayer = imageInputLayer([32, 32, 1]); % Set to 1 channel

encodingLayers = [ ...
    convolution2dLayer(3,8,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2), ...
    convolution2dLayer(3,16,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2), ...
    convolution2dLayer(3,32,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2)];

decodingLayers = [ ...
    transposedConv2dLayer(2,32,Stride=2), ...
    reluLayer, ...
    transposedConv2dLayer(2,16,Stride=2), ...
    reluLayer, ...
    transposedConv2dLayer(2,8,Stride=2), ...
    reluLayer, ...
    convolution2dLayer(1,1,Padding="same"), ...
    clippedReluLayer(1.0), ...
    regressionLayer];    

layers = [imageLayer, encodingLayers, decodingLayers];

% 8. Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', imds.ReadSize, ...
    'ValidationData', dsVal, ...
    'ValidationPatience', 5, ...
    'Plots', 'training-progress', ...
    'OutputNetwork', 'best-validation-loss', ...
    'Verbose', false);

%net = trainNetwork(dsTrain, layers, options);


ypred = predict(net, dsTest);
testBatch = preview(dsTest);

numExamples = 8;


for i = 1 : numExamples
    idx = i;
    y = ypred(:,:,:,idx);
    x = testBatch{idx,1};
    ref = testBatch{idx,2};
    

    % Display the images
    subplot(4, 4, 2 * i - 1);
    imshow(x);
    title('Noisy Input');

    subplot(4, 4, 2 * i);
    imshow(y);
    title('Denoised Result');

end



function dataOut = addNoise(data)
    dataOut = data;
    for idx = 1:size(data, 1)
        dataOut{idx} = imnoise(data{idx}, "salt & pepper");
    end
end

function dataOut = commonPreprocessing(data)
    dataOut = cell(size(data));
    for col = 1:size(data, 2)
        for idx = 1:size(data, 1)
            temp = single(data{idx, col});
            temp = imresize(temp, [32, 32]);
            temp = rescale(temp);
            dataOut{idx, col} = temp;
        end
    end
end

function dataOut = augmentImages(data)
    dataOut = cell(size(data));
    for idx = 1:size(data, 1)
        rot90Val = randi(4, 1, 1) - 1;
        dataOut(idx, :) = {rot90(data{idx, 1}, rot90Val), ...
            rot90(data{idx, 2}, rot90Val)};
    end
end
