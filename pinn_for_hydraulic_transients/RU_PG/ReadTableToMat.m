clc
clear all
Train=readmatrix('train.xlsx')
X_test = [10];
F_test=Train(:,1)
t_test=Train(:,2)
a_test=Train(:,3)
save test.mat a_test F_test t_test X_test 


