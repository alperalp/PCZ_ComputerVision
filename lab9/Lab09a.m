imds = imageDatastore('Baza/', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
[imdsTrain, imdsTest]=splitEachLabel(imds,0.7);
lenTrain=length(imdsTrain.Labels);
lenTest=length(imdsTest.Labels);
layers = [ ...
imageInputLayer([28 28 1])
convolution2dLayer(5,20)
reluLayer
maxPooling2dLayer(2,'Stride',2)
fullyConnectedLayer(10)
softmaxLayer
classificationLayer];
MiniBatchSize=200;
options = trainingOptions('sgdm', ...
'MaxEpochs',20,...
'MiniBatchSize', MiniBatchSize, ...
'InitialLearnRate',1e-4, ...
'Shuffle','every-epoch',...
'Verbose',false, ...
'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
[YPred,probs] = classify(net,imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);


numImagesToShow = 25;
randomIndices = randperm(length(imdsTest.Files), numImagesToShow);

figure;

for i = 1:numImagesToShow
    subplot(5, 5, i);
    img = readimage(imdsTest, randomIndices(i));
    imshow(img);
    
    trueLabel = YTest(randomIndices(i));
    predictedLabel = YPred(randomIndices(i));
    
    title(['True: ' char(trueLabel) ', Predicted: ' char(predictedLabel)]);
end

sgtitle('Images from Test Set with Predictions');
