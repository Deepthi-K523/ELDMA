clc;clear;clear all;

SM = load("C:\Users\.............\miRNA similarity.txt");
SDr = load("C:\Users\............\Drug similarity.txt");


 a= 0.6; b= 0.9;
drug_feature=PCA((SDr)/2,a);  
miRNA_feature=PCA((SM)/2,b);    

  
dlmwrite('C:\Users\deept\Desktop\Drug similarity-PCA.txt',drug_feature);
dlmwrite('C:\Users\deept\Desktop\miRNA similarity-PCA.txt',miRNA_feature);



