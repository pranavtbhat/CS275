close all
clear
clc

%% Locate images
image_dir = "";
if ismac
    image_dir = '/../resources/ORL/Train/s';
else
    image_dir = '\..\resources\ORL\Train\s';
end


load baboon6400_256.mat

%Customize here

nFolder = 40; % No. of folders
nTrain = 5; % No. of images per subject for train
nTest = 5; % No. of images per subject for test
nTrainTotal = nTrain*nFolder;


% Training Phase
% Loading the images to compute TrainSet, 200 x 10304 matrix - 
fpath = mfilename('fullpath');
[path fname ext] = fileparts(fpath);
i = 0;
label = 1;
TrainTarget = zeros(nTrainTotal, nFolder);
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path, image_dir, temp);
    cd(folder);
    myfiles = dir('*.pgm');
    n = length(myfiles);
    for j = 1:n
        filename = myfiles(j).name;
        I = im2double(imread(filename));
        [X Y Z] = size(I);
        if(Z == 3)
            J = rgb2gray(I);
        else
            J = I;
        end
        K = reshape(J, 1, X*Y);  %trad PCA
        %use imgimport
        K = importimg(imgGraph,J);
        TrainSet(:, i + j) = K;
        
        TrainTarget(label, i + j) = 1;
    end
    i = i + nTrain;
    label = label + 1;
end


Mean = sum(TrainSet, 2)/(nTrainTotal);
for i = 1:nTrainTotal
    TrainSet(:,i) = TrainSet(:,i) - Mean;
end


% Testing Phase Preparation
% Loading the images to compute TestSet matrix, typically sized: 200 x 10304
nTestTotal = nTest*nFolder;
i = 0;
label = 1;
TestTarget = zeros(nTestTotal, nFolder);
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path, image_dir, temp);
    cd(folder);
    myfiles = dir('*.pgm');
    n=length(myfiles);
    for j = 1:n
        filename = myfiles(j).name;
        I = im2double(imread(filename));        
        [X Y Z] = size(I);
        if(Z == 3)
            J = rgb2gray(I);
        else
            J = I;
        end
        K = reshape(J, 1, X*Y);
        
        K = importimg(imgGraph,J);
        TestSet(:, i + j) = K;
        TestTarget(label, i + j) = 1;
    end
    i = i + nTest;
    label = label + 1;
end

% Mean centering for Images in test set, stored in TestSet
Mean = sum(TestSet,2)/nTestTotal;
for i = 1:nTestTotal
    TestSet(:,i) = TestSet(:,i) - Mean;
end

inputs = horzcat(TrainSet, TestSet);
targets = horzcat(TrainTarget, TestTarget);

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 50/100;


% Train the Network
[net,tr] = train(net, inputs, targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
%view(net)

%figure, plotperform(tr)

tInd = tr.testInd;
tstOutputs = net(inputs(:,tInd));
tstPerform = perform(net, targets(:,tInd), tstOutputs)
