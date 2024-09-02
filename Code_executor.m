%%% This is a code executor file for PCA 
clear 
clc
close all

data=rand(50,3);

mypca=pca_nipals(data);

x_new=data(1:5,:);
[x_hat,t_point,SPE,tsquared,x_new_scaled]=pca_evaluation(mypca,x_new)

pca_visual_plot(mypca,[1,2],x_new)