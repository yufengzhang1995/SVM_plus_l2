%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Author: Yufeng Zhang
%%%%%%%%% Date: 9/28/20
%%%%%%%%%
%%%%%%%%% General Comments: This function compares the specificity,
%%%%%%%%% sensitivity, accuracy of standard svm, l2-SVMplus and 
%%%%%%%%% our svm+ model
%%%%%%%%%
%%%%%%%%% Input: data
%%%%%%%%% Output: three list of specificity,sensitivity,accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [flag] = compare_privileged(train_labels,train_features,train_PFfeatures,test_labels,test_features)

    flag = -1;
    
    % preprocessing data with L1-normalization
    train_features = L1_normalization(train_features');
    test_features  = L1_normalization(test_features');
    train_PFfeatures = L1_normalization(train_PFfeatures');

    % calculate kernels
    kparam = struct();
    kparam.kernel_type = 'gaussian';
    [K, train_kparam] = getKernel(train_features, kparam);
    testK = getKernel(test_features, train_features, train_kparam);

    kparam = struct();
    kparam.kernel_type = 'gaussian';
    tK = getKernel(train_PFfeatures, kparam);

    % ================ train l2-SVM+ ====================


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
    % acc_l2  = sum(l2_label == test_labels)/length(test_labels);
    % [sens_l2,spec_l2] = AUC(l2_label,test_labels);

    test_labels = (test_labels + 1)/2;
    l2_label = (l2_label + 1) / 2;
    [~,~,~,AUC_l2,~,~, sens_l2,spec_l2, acc_l2] = ROC_AUC(l2_label, test_labels, 2, 10000, 0, 0);
    
    fprintf("\n================\n")
    fprintf(2, 'L2-SVM+ , time=%f, Accuracy = %.4f.\n', t_l2, acc_l2);
    fprintf(2, 'L2-SVM+ , Sensitivity=%.2f. Specificty = %.2f.\n', sens_l2, spec_l2);
    fprintf(1, 'L2-SVM+ , AUC score = %.2f', AUC_l2);
    fprintf("\n================\n")

    
end






