imds2 = imageDatastore('Baza2/', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain2, imdsTest2] = splitEachLabel(imds2, 0.7);

layers2 = [
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20)
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

MiniBatchSize = 200;
options2 = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', MiniBatchSize, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net2 = trainNetwork(imdsTrain2, layers2, options2);

[YPred2, probs2] = classify(net2, imdsTest2);
YTest2 = imdsTest2.Labels;

accuracy2 = sum(YPred2 == YTest2) / numel(YTest2);

numImagesToShow2 = 16;
randomIndices2 = randperm(length(imdsTest2.Files), numImagesToShow2);

figure;

for i = 1:numImagesToShow2
    subplot(4, 4, i);
    img2 = readimage(imdsTest2, randomIndices2(i));
    imshow(img2);
    
    trueLabel2 = YTest2(randomIndices2(i));
    predictedLabel2 = YPred2(randomIndices2(i));
    
    title(['True: ' char(trueLabel2) ', Predicted: ' char(predictedLabel2)]);
end

sgtitle('Images from Test Set Digits 0 and 1)');
