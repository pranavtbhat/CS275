% Script for face recognition using PCA and 1-NN classifier on ORL Image 
% Database
% - Pranav Sodhani (04/23/2017)
% =======================================================================
% About Training set:
% Number of subjects: 40
% Number of Images per subject: 5
% Image size: 92 x 112 (jpg format)
% =======================================================================
% About Testing set:
% Number of subjects: 40
% Number of Images per subject: 5
% Image size: 92 x 112 (jpg format)
% =======================================================================

close all
clear
clc

load baboon6400_256.mat

%Customize here

nFolder = 40; % No. of folders
nTrain = 5; % No. of images per subject for train
nTest = 5; % No. of images per subject for test

% Training Phase
% Loading the images to compute TrainSet, 200 x 10304 matrix - 
fpath = mfilename('fullpath');
[path fname ext] = fileparts(fpath);
i = 0;
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path,'\..\resources\Dataset for ICA\Train\s', temp);
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
        %K = reshape(J, 1, X*Y);  %trad PCA
        %use imgimport
        K = importimg(imgGraph,J);
        TrainSet(i + j, :) = K;
    end
    i = i + nTrain;
end

nTrainTotal = nTrain*nFolder;
% Mean = sum(TrainSet,1)/(nTrainTotal);
% for i = 1:nTrainTotal
%     TrainSet(i,:) = TrainSet(i,:) - Mean;
% end


% Testing Phase Preparation
% Loading the images to compute TestSet matrix, typically sized: 200 x 10304
nTestTotal = nTest*nFolder;
i = 0;
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path,'\..\resources\Dataset for ICA\Test\s', temp);
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
        %K = reshape(J, 1, X*Y);
        
        K = importimg(imgGraph,J);
        TestSet(i + j, :) = K;
    end
    i = i + nTest;
end

% Mean centering for Images in test set, stored in TestSet
Mean = sum(TestSet,1)/nTestTotal;
for i = 1:nTestTotal
    TestSet(i,:) = TestSet(i,:) - Mean;
end


%addpath('\..\resources\FastICA_25');

%ICA

[icasig,A,W]= fastica(TrainSet);
winv = inv(W);

normica = normr(icasig);

normicatrans = normica';

result = TestSet*normicatrans;

%TrainVect = TrainSet*W;
% Test image read from Test and comparing with nTrain images in reduced
% dimensional space of size ev


for i = 1:nTestTotal
    Z1 = result(i,:);
    for j = 1:nTrainTotal
        Z2 = winv(j,:);
        ABC = (Z1-Z2)';
        Dist(j)= (ABC)'*(ABC); % Distance between Z1 and Z2
    end
    % Finding index of the least distant sample
    [value, index] = min(Dist);
    Tracker(i) = index;
end

% Checking Classification accuracy
accuracy = 0;
cnt = 1;
tmpcnt = 1;
for i = 1:nFolder
    for j = cnt: cnt + 4
        if ((Tracker(tmpcnt) >= cnt) & (Tracker(tmpcnt) <= cnt + 4))
            accuracy = accuracy + 1;
        end
        tmpcnt = tmpcnt + 1;
    end
    cnt = cnt + nTest;
end
        
%disp('Accuracy')
accuracy = (accuracy)/2
