function [K] = return_LinearKernel(featuresA)
    K = featuresA'*featuresA;
end