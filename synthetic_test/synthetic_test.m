clear; clc;
addpath('./utils');
addpath('./matlab');

% load data
load('synthetic_data.mat');

% split the data into train and test
[m,n] = size(X_exist) ;
P = 0.050 ;
idx = randperm(m)  ;
train_features = X_exist(idx(1:round(P*m)),:);
train_labels = new_Y(idx(1:round(P*m)));
test_features = X_exist(idx(round(P*m)+1:end),:);
test_labels = new_Y(idx(round(P*m)+1:end));
train_PFfeatures = X_add(idx(1:round(P*m)),:);
weight = weight(idx(1:round(P*m)),:);

% preprocessing data with L1-normalization
train_features      = L1_normalization(train_features');
test_features       = L1_normalization(test_features');
train_PFfeatures    = L1_normalization(train_PFfeatures');

% % ================ train oracle-SVM+ ====================
train_features = full(train_features);
test_features = full(test_features);

tic;
model_oracle = fitcsvm(train_features',train_labels);
t_oracle =  toc;

[labels_oracle,~] = predict(model_oracle,test_features.');
test_labels = (test_labels + 1)/2;
labels_oracle = (labels_oracle + 1) / 2;
[~,~,~,AUC_oracle,~,~, sens_oracle,spec_oracle, acc_oracle] = ROC_AUC(labels_oracle, test_labels, 2, 10000, 0, 0);


fprintf("\n================\n")
fprintf(2, 'oracle-SVM, time=%f, Accuracy = %.4f.\n', t_oracle, acc_oracle);
fprintf(2, 'oracle-SVM, Sensitivity=%.2f. Specificty = %.2f.\n', sens_oracle, spec_oracle);
fprintf(1, 'oracle-SVM , AUC score = %.2f', AUC_oracle);
fprintf("\n================\n")



% ================ train l2-SVM+ ====================

% calculate kernels
kparam = struct();
kparam.kernel_type = 'gaussian';
[K, train_kparam] = getKernel(train_features, kparam);
testK       = getKernel(test_features, train_features, train_kparam);

kparam = struct();
kparam.kernel_type = 'gaussian';
tK = getKernel(train_PFfeatures, kparam);


svmplus_param.svm_C = 1; 
svmplus_param.gamma = 1;

tic;
model = solve_l2svmplus_kernel(train_labels, K, tK, svmplus_param.svm_C, svmplus_param.gamma);
t_l2 = toc;
alpha       = zeros(length(train_labels), 1);
alpha(model.SVs) = full(model.sv_coef);
alpha       = abs(alpha);
decs = (testK + 1)*(alpha.*train_labels);
l2_label = 2*(decs>0)-1;

l2_label = (l2_label + 1) / 2;
[~,~,~,AUC_l2,~,~, sens_l2,spec_l2, acc_l2] = ROC_AUC(l2_label, test_labels, 2, 10000, 0, 0);

fprintf("\n================\n")
fprintf(2, 'L2-SVM+ , time=%f, Accuracy = %.4f.\n', t_l2, acc_l2);
fprintf(2, 'L2-SVM+ , Sensitivity=%.2f. Specificty = %.2f.\n', sens_l2, spec_l2);
fprintf(1, 'L2-SVM+ , AUC score = %.2f', AUC_l2);
fprintf("\n================\n")

