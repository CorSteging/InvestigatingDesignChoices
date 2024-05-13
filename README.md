# Taking the Law More Seriously by Investigating Design Choices in Machine Learning Prediction Research

Approaches to court case prediction using machine learning differ widely with varying levels of success and legal reasonableness. In part this is due to some aspects of law, such as justification, being inherently difficult for machine learning approaches. Another aspect is the effect of design choices and the extent to which these are legally reasonable, which has not yet been extensively studied. We create four machine learning models tasked with predicting cases from the European Court of Human Rights and we perform experiments in order to measure the role of the following four design choices and effects: the choice of performance metric; the effect of including different parts of the legal case; the effect of a more or less specialized legal focus; and the temporal effects of the available past legal decisions. Through this research, we aim to study design decisions and their limitations and how they affect the performance of machine learning models. 

This repository contains all of the files needed to replicate the experiments. 

File overview:
* analyse_results.ipynb - Analysis of all of the results
* echr.py - The extended replication
* bert.py - The extended replication with the BERT classifier 
* parts_BERT.ipynb
* parts_NB_SVM_RF.ipynb
* generalist_vs_specialist_BERT.ipynb - The generalist vs specialist experiment for the BERT classifier 
* generalist_vs_specialist_NB_SVM_RF.ipynb - The generalist vs specialist experiment for the naive Bayes, SVM and Random Forest 
* temporal_effects_BERT.ipynb - The temporal effects experiments for the BERT classifier 
* temporal_effects_NB.ipynb - The temporal effects experiments using the Naive Bayes classifier
* temporal_effects_RF.ipynb - The temporal effects experiments using the Random Forest
* temporal_effects_SVM.ipynb - The temporal effects experiments using the SVM

Please note that additional Python libraries are required, as indicated at the top of each jupyter notebook.
