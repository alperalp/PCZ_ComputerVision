imds = imageDatastore('Baza/', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');


[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);


layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
];


options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'ValidationData', imdsTest);


% net = trainNetwork(imdsTrain, layers, options);


Ypred = classify(net, imdsTest);


conf_matrix = confusionmat(imdsTest.Labels, Ypred);


figure;
confusionchart(imdsTest.Labels, Ypred,"ColumnSummary","column-normalized",'RowSummary','row-normalized');
figure;

numClasses = size(conf_matrix,1);
TP = zeros(1,numClasses);
TN = zeros(1,numClasses);
FP = zeros(1,numClasses);
FN = zeros(1,numClasses);

for i = 1:numClasses
    TP(i) = conf_matrix(i,i);
    FP(i) = sum(conf_matrix(:,i)) - TP(i);
    FN(i) = sum(conf_matrix(i,:)) - TP(i);
    TN(i) = sum(conf_matrix(:)) - TP(i) - FN(i) - FP(i);
end

disp(['TP for classes :', num2str(TP)]);
disp(['FP for classes :', num2str(FP)]);
disp(['FN for classes :', num2str(FN)]);
disp(['TN for classes :', num2str(TN)]);

recall = zeros(1,numClasses);
specifity = zeros(1,numClasses);
F1 = zeros(1,numClasses);
MCC= zeros(1,numClasses);
precision = zeros(1,numClasses);



for i = 1:numClasses
    accuracy = (TP(i) + TN(i)) / (TP(i)+FP(i)+TN(i)+FN(i));
    disp([num2str(i),'. class accuracy : ',num2str(accuracy)]);
    recall(i) = TP(i) / (TP(i) + FN(i));
    disp([num2str(i),'. class recall : ',num2str(recall(i))]);
    specifity(i) = TN(i) / (TN(i) + FP(i));
    disp([num2str(i),'. class specifity: ',num2str(specifity(i))]);
    precision(i) = TP(i) / (TP(i) + FP(i));
    disp([num2str(i),'. class precision: ',num2str(precision(i))]);
    F1(i) = (2*TP(i)) / ((2 * TP(i)) + FP(i) + FN(i));
    disp([num2str(i),'. class f1-score : ',num2str(F1(i))]);
    MCC(i) = ((TP(i) * TN(i))- (FP(i) * FN(i))) / (sqrt( (TP(i) + FP(i)) * (TP(i) + FN(i)) * (TN(i) + FP(i)) * (TN(i) + FN(i)) ));
    disp([num2str(i),'. mcc : ',num2str(MCC(i))]);
    
end

random_idx = randperm(length(imdsTest.Labels), 12);
for i = 1:12
    subplot(3, 4, i);
    img = readimage(imdsTest, random_idx(i));
    imshow(img);
    title(sprintf('True: %s\nPred: %s', imdsTest.Labels(random_idx(i)), Ypred(random_idx(i))));
end



