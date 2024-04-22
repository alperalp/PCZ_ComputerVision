rootFolder = 'cifar10'; 

% Create an imageDatastore for all images
allImages = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Set the random seed for reproducibility
rng(42);

% Define the percentage split for training, validation, and test sets
trainPercentage = 0.7;
valPercentage = 0.2;
testPercentage = 0.1;

% Split the imageDatastore into training, validation, and test sets
[trainImages, tempImages] = splitEachLabel(allImages, trainPercentage, 'randomized');
[valImages, testImages] = splitEachLabel(tempImages, valPercentage/(valPercentage + testPercentage), 'randomized');

% Display the number of images in each set
disp(['Number of training images: ' num2str(length(trainImages.Files))]);
disp(['Number of validation images: ' num2str(length(valImages.Files))]);
disp(['Number of test images: ' num2str(length(testImages.Files))]);

% Example: Display a few images from each set
figure;

% Display training set examples
subplot(1, 3, 1);
randomTrainIdx = randperm(length(trainImages.Files), 9);
montage(trainImages.Files(randomTrainIdx));
title('Training Set Examples');

% Display validation set examples
subplot(1, 3, 2);
randomValIdx = randperm(length(valImages.Files), 9);
montage(valImages.Files(randomValIdx));
title('Validation Set Examples');

% Display test set examples
subplot(1, 3, 3);
randomTestIdx = randperm(length(testImages.Files), 9);
montage(testImages.Files(randomTestIdx));
title('Test Set Examples');


imageSize = [32, 32, 3];

% Define the CNN architecture
layers = [
    imageInputLayer(imageSize, 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')

    fullyConnectedLayer(10, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valImages, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the CNN
cnnModel = trainNetwork(trainImages, layers, options);

% Evaluate the trained model on the validation set
valPred = classify(cnnModel, valImages);
valActual = valImages.Labels;

accuracy = sum(valPred == valActual) / numel(valActual);
disp(['Validation Accuracy: ' num2str(accuracy * 100) '%']);

% Save the trained model
save('trained_cnn_model.mat', 'cnnModel');
disp('Trained model saved.');

% Load the trained model
load('trained_cnn_model.mat'); 

% Display test images from different classes
numClassesToShow = 10;  % Number of different classes to display
numExamplesPerClass = 5;  % Number of examples per class to display

figure;

% Iterate over classes
for i = 1:numClassesToShow
    currentClass = unique(testImages.Labels);
    currentClass = currentClass(i);

    % Find indices of images belonging to a specific class
    classIdx = find(testImages.Labels == currentClass, numExamplesPerClass, 'first');

    % Iterate over examples in the current class
    for j = 1:numExamplesPerClass
        subplot(numClassesToShow, numExamplesPerClass, (i-1)*numExamplesPerClass + j);
        imgIdx = classIdx(j);
        img = readimage(testImages, imgIdx);
        trueLabel = testImages.Labels(imgIdx);
        predictedLabel = classify(cnnModel, img);

        imshow(img);
        title(['True: ' char(trueLabel) ', Predicted: ' char(predictedLabel)], 'FontSize', 8);
    end
end
% Classify the test set
testPred = classify(cnnModel, testImages);

% Extract true labels from the test set
trueLabels = testImages.Labels;

% Create a confusion matrix
confMat = confusionmat(trueLabels, testPred);

% Get the number of classes
numClasses = size(confMat, 1);

% Initialize arrays to store metrics for each class
precisionPerClass = zeros(1, numClasses);
recallPerClass = zeros(1, numClasses);
f1ScorePerClass = zeros(1, numClasses);
mccPerClass = zeros(1, numClasses);

% Loop over each class
for i = 1:numClasses
    % Extract TP, TN, FP, FN for the current class
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;

    % Calculate precision, recall, F1 score, and MCC for the current class
    precisionPerClass(i) = TP / (TP + FP);
    recallPerClass(i) = TP / (TP + FN);
    f1ScorePerClass(i) = 2 * (precisionPerClass(i) * recallPerClass(i)) / (precisionPerClass(i) + recallPerClass(i));
    mccPerClass(i) = (TP * confMat(1, 1) - FP * confMat(1, 2)) / sqrt((TP + FP) * (TP + confMat(2, 1)) * (confMat(1, 1) + FP) * (confMat(2, 2) + confMat(2, 1)));

    % Print metrics for the current class
    disp(['Class ' num2str(i) ' - Precision: ' num2str(precisionPerClass(i)) ', Recall: ' num2str(recallPerClass(i)) ', F1 Score: ' num2str(f1ScorePerClass(i)) ', MCC: ' num2str(mccPerClass(i))]);
end

% Calculate average metrics
avgPrecision = mean(precisionPerClass);
avgRecall = mean(recallPerClass);
avgF1Score = mean(f1ScorePerClass);
avgMCC = mean(mccPerClass);

% Print average metrics
disp('Average Metrics:');
disp(['Average Precision: ' num2str(avgPrecision)]);
disp(['Average Recall: ' num2str(avgRecall)]);
disp(['Average F1 Score: ' num2str(avgF1Score)]);
disp(['Average MCC: ' num2str(avgMCC)]);

% Plot bar graphs for each metric
figure;

% Bar plot for Precision
subplot(4, 1, 1);
bar(precisionPerClass);
title('Precision Per Class');
xlabel('Class');
ylabel('Precision');

% Bar plot for Recall
subplot(4, 1, 2);
bar(recallPerClass);
title('Recall Per Class');
xlabel('Class');
ylabel('Recall');

% Bar plot for F1 Score
subplot(4, 1, 3);
bar(f1ScorePerClass);
title('F1 Score Per Class');
xlabel('Class');
ylabel('F1 Score');

% Bar plot for MCC
subplot(4, 1, 4);
bar(mccPerClass);
title('MCC Per Class');
xlabel('Class');
ylabel('MCC');
