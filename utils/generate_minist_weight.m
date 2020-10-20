clear; clc;
% generate label uncertainty mnist data

load('mnist_plus.mat');
m = length(train_labels);

a = 0.5;
b = 1;
weight = (b-a).*rand(m,1) + a;

save mnist_plus.mat