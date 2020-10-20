clear; clc;
addpath('./utils');
addpath('./matlab');

% load data
mat = load('./eight_mnist.mat');
data = mat.S;

train_features = data.train_features;
test_features = data.test_features;
train_PFfeatures = data.train_PFfeatures;
train_labels = data.train_labels;
test_labels = data.test_labels;
label_uncert = data.eight_label;

train_labels(train_labels==5) = 1;
train_labels(train_labels~=1) = -1;
test_labels(test_labels==5) = 1;
test_labels(test_labels~=1) = -1;

[~] = compare_privileged(train_labels,train_features,train_PFfeatures,test_labels,test_features);
[~] = compare_LULUPI(train_labels,train_features,train_PFfeatures,label_uncert,test_labels,test_features);




    