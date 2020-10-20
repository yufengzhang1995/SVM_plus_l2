% First download libsvm file grom github:https://github.com/cjlin1/libsvm
% unzip the file
% cd to libsvm-master and type make. It will appear 4 more files
% check use the command ./svm-train heart_scale then a .model will be generated
% we can continue check use ./svm-predict heart_scale heart_scale.model heart_scale.out to predict
% then go into matlab directory
% make
% generated mexa64 files
% add the whole matlab to the path then the functions inside can be applied to the data


% load data and utils
addpath('./utils');
addpath('./matlab')
addpath('./dataset')
load('mnist_plus.mat');

% preprocessing data with L1-normlaization
train_features = L1_normalization(train_features');
test_features =  L1_normalization(test_features');
train_PFfeatures = L1_normalization(train_PFfeatures');

train_labels(train_labels==5) = 1;
train_labels(train_labels~=1) = -1;
test_labels(test_labels==5) = 1;
test_labels(test_labels~=1) = -1;

train_features = full(train_features);

model = svmtrain(train_labels, train_features, ['libsvm_options']);

% [heart_scale_label,heart_scale_inst]=libsvmread('heart_scale');
% model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');
% [predict_label, accuracy, dec_values] =svmpredict(heart_scale_label, heart_scale_inst, model); 
