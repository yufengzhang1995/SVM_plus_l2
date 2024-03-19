
Repo for 'Increasing efficiency of SVMp+ for handling missing values in healthcare prediction.'


- LULUPI: Learning Using Privileged Information with Label Uncertainty  
- LUPAPI: Learning Using Partially Available Privileged Information  
- LULUPAI: Learning Using Partially Available Privileged Information with Label Uncertainty

- # SVM_plus_l2

The algorithm is for fast computation of LULUPI,LUPAPI and LULUPAPI

## Author: 

Yufeng Zhang: chloezh@umich.edu

## Note:
Based on the SVM+ algorithm from Wen Li (Paper; Fast Algorithms for Linear and Kernel SVM+)  
And the Model from Elyas Sabeti (Paper: Learning Using Partially Available Privileged Information and Label Uncertainty)  

## Table of Contents

- Introduction
- Installation of libsvm
- Usage


### Introduction

#### This Algorithm uses SVM-l2 based model for binary classification: for LULUPI, LUPAPI and LULUPAPI efficient computation.  

* LULUPI: Learning Using Privileged Information with Label Uncertainty  
* LUPAPI: Learning Using Partially Availabel Privileged Information    
* LULUPAPI: Learning Using Partially Availabel Privileged Information with Label Uncertainty   

### Installation of Libsvm

#### use test_libsvm.m file to check libsvm installed successfully. 


* download libsvm file grom github:https://github.com/cjlin1/libsvm and then unzip the file. 
* cd to libsvm-master and type make. It will appear 4 more files. 
* check use the command ./svm-train heart_scale then a .model will be generated. 
* continue check use ./svm-predict heart_scale heart_scale.model heart_scale.out to predict. 
* go into matlab directory and type make to generate mexa64 files. 
* add the whole matlab to the path then the functions inside can be applied to the data. 

### Usage

#### Details are described in specific folders and functions. 

##### check synthetic_LULUPAPI for more usage information

* In LULUPI: all training samples have privileged information. And every label goes with a weight to indicate the confidence level.  
* In LUPAPI: part of the training samples have privileged information.   
* In LULUPAPI: part of the trainning samples have privileged information. Every label goes with a weight to indicate the confidence level.
