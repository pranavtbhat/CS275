% Script for face recognition using PCA and 1-NN classifier on ORL Image 
% Database [SPACE VARIANT VERSION]
% - Pranav Sodhani (04/23/2017)
% =======================================================================
% About Training set:
% Number of subjects: 40
% Number of Images per subject: 5
% Image size: 92 x 112 (pgm format)
% =======================================================================
% About Testing set:
% Number of subjects: 40
% Number of Images per subject: 5
% Image size: 92 x 112 (pgm format)
% =======================================================================

close all
clear
clc

%Customize here
ev = 25; % No. of eigenvectors to consider
nFolder = 40; % No. of folders
nTrain = 5; % No. of images per subject for train
nTest = 5; % No. of images per subject for test

% Training Phase
% Loading eye structure
load baboon6400_256.mat
% Loading the images to compute TrainSet, 200 x 10304 matrix - 
fpath = mfilename('fullpath');
[path fname ext] = fileparts(fpath);
i = 0;
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path,'\Train\s', temp);
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
        vals = importimg(imgGraph,J);
        TrainSet(i + j, :) = vals;
    end
    i = i + nTrain;
end

% Mean centering for Images in Gallery, stored in TrainSet
nTrainTotal = nTrain*nFolder;
Mean = sum(TrainSet,1)/(nTrainTotal);
for i = 1:nTrainTotal
    TrainSet(i,:) = TrainSet(i,:) - Mean;
end

% Finding Covariance matrix and eigenvectors -
Covar = (TrainSet*TrainSet')/nTrainTotal;
[V, D] = eig(Covar);
TransformedEV = TrainSet'*V; % TransformedEV contains the final eigenvectors, transformed into the original space.

% Evaluating Eigenvector matrix for top ev eigenfaces. 
% The eigenvectors corresponding to top ev eigenvalues are moved to N
for i = 0:ev-1
    N(:,i+1) = TransformedEV(:,nTrainTotal-i);
    U = N(:,i+1);
    norm = (U'*U)^0.5;
    N(:,i+1) = N(:,i+1)/norm; % eigenvectors need to be normalized to form an orthogonal set.
end

for i = 1:nTrainTotal
    TrainVect(:,i) = (N)'*(TrainSet(i,:))'; % Training set
end

% Testing Phase Preparation
% Loading the images to compute TestSet matrix, typically sized: 200 x 10304
nTestTotal = nTest*nFolder;
i = 0;
for k = 1:nFolder
    temp = sprintf('%d', k);
    folder = strcat(path,'\Test\s', temp);
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
        vals = importimg(imgGraph,J);
        TestSet(i + j, :) = vals;
    end
    i = i + nTest;
end

% Mean centering for Images in test set, stored in TestSet
Mean = sum(TestSet,1)/nTestTotal;
for i = 1:nTestTotal
    TestSet(i,:) = TestSet(i,:) - Mean;
end

% Test image read from Test and comparing with nTrain images in reduced
% dimensional space of size ev

for i = 1:nTestTotal
    TestVect(:,i) = (N)'*(TestSet(i,:))'; % Testing set
end

for i = 1:nTestTotal
    Z1 = TestVect(:,i);
    for j = 1:nTrainTotal
        Z2 = TrainVect(:,j);
        Dist(j)= (Z1-Z2)'*(Z1-Z2); % Distance between Z1 and Z2
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
        
accuracy = (accuracy)/2
