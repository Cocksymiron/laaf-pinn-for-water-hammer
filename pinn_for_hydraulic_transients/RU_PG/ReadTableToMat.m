clc
clear all
Train=readmatrix('train.xlsx')
X = [10];
F=Train(:,1)
t=Train(:,2)
a=Train(:,3)
save train.mat a F t X 


