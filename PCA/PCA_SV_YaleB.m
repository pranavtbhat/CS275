% Script for face recognition using PCA and 1-NN classifier on YALE-B Image 
% Database [SPACE VARIANT VERSION]
% - Pranav Sodhani (04/23/2017)
% =======================================================================
% About Training set:
% Number of subjects: 39
% Number of Images per subject: nTrain
% Image size: 192 x 168 (pgm format)
% =======================================================================
% About Testing set:
% Number of subjects: 39
% Number of Images per subject: nTest
% Image size: 192 x 168 (pgm format)
% =======================================================================

close all
clear
clc

%Customize here
ev = 60 % No. of eigenvectors to consider
nFolder = 39; % No. of folders (folder 14 is missing)
nTrain = 10; % No. of images per subject for train
nTest = 60-nTrain; % No. of images per subject for test

% Training Phase
% Loading eye structure
load baboon6400_256.mat
% Loading the images to compute TrainSet
fpath = mfilename('fullpath');
[path fname ext] = fileparts(fpath);
i = 0;
var = 1;
sigma = 3;
W = spv_gaussfilter(edges,points, sigma);
for k = 1:nFolder
    k
    temp = sprintf('%d', k);
    if k==14
        continue;
    end
    if k<10
        folder = strcat(path,'\CroppedYale\CroppedYale\yaleB0', temp);
    else
        folder = strcat(path,'\CroppedYale\CroppedYale\yaleB', temp);
    end
    cd(folder);
    myfiles = dir('*.pgm');
    %n = length(myfiles);
    n = nTrain;
    for j = 1:n
        filename = myfiles(j).name;
        I = im2double(imread(filename));
        [X Y Z] = size(I);
        if(Z == 3)
            J = rgb2gray(I);
        else
            J = I;
        end
        if X~=192
            K = imresize(J, [192 168]);
            clear J
            J = K;
            X = 192;
            Y = 168;
        end
        vals = importimg(imgGraph,J);
        vals = spv_sqi(vals, edges, 5, 3, 0, W);
        TrainSet(i + j, :) = vals;
    end
    i = i + nTrain;
end

% Mean centering for Images in Gallery, stored in TrainSet
nTrainTotal = nTrain*(nFolder-1);
Mean = sum(TrainSet,1)/(nTrainTotal);
for i = 1:nTrainTotal
    TrainSet(i,:) = TrainSet(i,:) - Mean;
end
clear Mean
% Finding Covariance matrix and eigenvectors -
Covar = (TrainSet*TrainSet')/nTrainTotal;
[V, D] = eigs(Covar, ev);
TransformedEV = TrainSet'*V; % TransformedEV contains the final eigenvectors, transformed into the original space.
clear Covar

% Evaluating Eigenvector matrix for top ev eigenfaces. 
% The eigenvectors corresponding to top ev eigenvalues are moved to N
for i = 0:ev-1
    N(:,i+1) = TransformedEV(:,i+1);
    U = N(:,i+1);
    norm = (U'*U)^0.5;
    N(:,i+1) = N(:,i+1)/norm; % eigenvectors need to be normalized to form an orthogonal set.
end
for i = 1:nTrainTotal
    TrainVect(:,i) = (N)'*(TrainSet(i,:))'; % Training set
end
clear U
clear TrainSet
clear TransformedEV
% Testing Phase Preparation
% Loading the images to compute TestSet matrix
nTestTotal = nTest*(nFolder-1);
i = 0;
for k = 1:nFolder
    
    temp = sprintf('%d', k);
    if k==14 % adjustments for dataset discrepancy
        continue;
    end
    if k<10
        folder = strcat(path,'\CroppedYale\CroppedYale\yaleB0', temp);
    else
        folder = strcat(path,'\CroppedYale\CroppedYale\yaleB', temp);
    end
    cd(folder);
    myfiles = dir('*.pgm');
    n = 60; % adjustments for dataset discrepancy
    for j = 1:n
        if j<nTrain+1
            continue;
        end
        filename = myfiles(j).name;
        I = im2double(imread(filename));        
        [X Y Z] = size(I);
        if(Z == 3)
            J = rgb2gray(I);
        else
            J = I;
        end
        if X~=192 % adjustments for dataset discrepancy
            K = imresize(J, [192 168]);
            clear J
            J = K;
            X = 192;
            Y = 168;
        end
        vals = importimg(imgGraph,J);
        vals = spv_sqi(vals, edges, 5, 3, 0, W);
        TestSet(i + j-nTrain, :) = vals;
        
    end
    i = i + nTest;
end

% Mean centering for Images in test set, stored in TestSet
Mean = sum(TestSet,1)/nTestTotal;
for i = 1:nTestTotal
    TestSet(i,:) = TestSet(i,:) - Mean;
end
clear Mean
% Test image read from Test and comparing with nTrain images in reduced
% dimensional space of size ev

for i = 1:nTestTotal
    TestVect(:,i) = (N)'*(TestSet(i,:))'; % Testing set
end

for i = 1:nTestTotal
    Z1 = TestVect(:,i);
    for j = 1:nTrainTotal
        Z2 = TrainVect(:,j);
        Dist(j)= ((Z1-Z2)'*(Z1-Z2)); % Distance between Z1 and Z2
    end
    % Finding index of the least distant sample
    [value, index] = min(Dist);
    Tracker(var, i) = index;
    error(var,i) = value;
end
figure, plot(Tracker);
% Checking Classification accuracy
accuracy = 0;
cnt = 1;
tmpcnt = 1;
sum = 0;

for i = 1:nFolder-1
    for j = cnt: cnt + nTest-1
        if ((Tracker(tmpcnt) >= cnt) & (Tracker(tmpcnt) <= cnt + nTrain-1))
            accuracy = accuracy + 1;
            %correct_error(tmpcnt) = error(cnt);
        else
            %incorrect_error(tmpcnt) = error(cnt);
        end
        tmpcnt = tmpcnt + 1;
    end
    cnt = cnt + nTrain;
end
accuracy*100/(nTest*38) % 38 folders effectively
