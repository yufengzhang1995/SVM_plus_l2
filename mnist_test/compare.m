%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Author: Yufeng Zhang
%%%%%%%%% Date: 9/28/20
%%%%%%%%%
%%%%%%%%% General Comments: This function compares the specificity,
%%%%%%%%% sensitivity, accuracy of standard LIBSVM, l2-SVMplus and 
%%%%%%%%% our svm+ model
%%%%%%%%%
%%%%%%%%% Input: data
%%%%%%%%% Output: three list of specificity,sensitivity,accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [flag] = compare_func(labels,features,PFfeatures)
    flag = -1;

    % hyperparameter
    svmplus_param.svm_C = 1;
    svmplus_param.gamma = 1;

    % calculate kernels
    kparam = struct();
    kparam.kernel_type = 'gaussian';
    [K,kparam] = getKernel(train_features,kparam);
    tK = getKernel(train_PFfeatures,kparam);

    % ============= train l2-SVM+ ==========
    tic;
    model_l2 = solve_l2svmplus_kernel(train_labels,K,tK,svmplus_param.svm_C,svmplus_param.gamma);
    t_l2 = toc;

    [labels_l2,~] = predict(model_l2,features);
    [~,~,~,AUC_l2,~,~, sens_l2,spec_l2, acc_12] = ROC_AUC(labels, labels_l2, 2, 10000, 0, 0);
    metric_l2 = [AUC_l2;sens_l2;spec_l2;acc_12]; 

    disp(metric_l2)
    fprintf(1, 'SVM-l2, time=%f', t_l2);

    % ============= train oracle-SVM+ ==========
    tic;
    model_oracle = fitcsvm(train_features,train_labels);
    t_oracle = toc;   

    [labels_oracle,~] = predict(model_oracle,features);
    [~,~,~,AUC_oracle,~,~, sens_oracle,spec_oracle, acc_oracle] = ROC_AUC(labels, labels_oracle, 2, 10000, 0, 0);
    metric_oracle = [AUC_oracle;sens_oracle;spec_oracle;acc_oracle]; 

    disp(metric_oracle)
    fprintf(1, 'oracle-SVM, time=%f', t_oracle);

    if t_l2 < t_oracle
        flag = 1;
    else
        flag = 0;
    end
end






