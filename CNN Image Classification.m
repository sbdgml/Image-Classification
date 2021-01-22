rootFolder = fullfile( 'pami09_preRelease');
categories = {'croquet','tennis','volleyball_smash','cricket'};


imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

croquet = find(imds.Labels == 'croquet',1);
tennis = find(imds.Labels == 'tennis',1);
volleyball_smash = find(imds.Labels == 'volleyball_smash',1);
cricket = find (imds.Labels == 'cricket',1);

figure
subplot(2,2,1);
imshow(readimage(imds,croquet));
subplot(2,2,2);
imshow(readimage(imds,tennis));
subplot(2,2,3);
imshow(readimage(imds,volleyball_smash));
subplot(2,2,4);
imshow(readimage(imds,cricket));

net = resnet50();
net.Layers(end);

numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds,0.3,'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ...
    'ColorPreprocessing', 'gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, augmentedTestSet, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
testLabels = testSet.Labels;

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat));

folder = 'pami09_preRelease\test101'
fileList = dir(fullfile(folder, '/*.png'))
randomIndex = randi(length(fileList), 1, 1) % Get random number.
fullFileName = fullfile(folder, fileList(randomIndex).name)
newImage = imread(fullFileName);

ds = augmentedImageDatastore(imageSize, ...
    newImage, 'ColorPreprocessing', 'gray2rgb');

ImageFeatures = activations(net, ds, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, ImageFeatures, 'ObservationsIn', 'columns');

sprintf('The loaded image belongs to %s class',label);

figure
imshow(newImage);
title(label);