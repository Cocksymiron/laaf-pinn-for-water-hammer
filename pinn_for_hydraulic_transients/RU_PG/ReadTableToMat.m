clc
clear all
Train=readmatrix('test.xlsx')
X_test = [10;30;50];
F_test=Train(:,2)
F_test(:,2)=Train(:,3)
F_test(:,3)=Train(:,4)
t_test=Train(:,1)
a_test=Train(:,5)
save test.mat a_test F_test t_test X_test 


