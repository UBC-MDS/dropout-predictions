# Student Dropout Predictor


* author: Ranjit Sundaramurthi
* contributors: Andy Wang, Caesar Wong, ZhiyiChen



## Introduction


### Objective


Academic performance/graduation in a population is an important factor in their overall employability which contributes towards economic development. This Data Science project predicts the Student Dropout given the factors on demography, socioeconomics, macroeconomics and relevant academic data provided by the Student on enrollment. This prediction is important to understand the student academic capacity. This important knowledge can be used to identify key areas of development such as, development of socially disadvantaged communities, improvement of academic programs, development of educational funding programs  etc.  


### Dataset


The dataset used in the project contains data collected at the time of student enrollment and a snapshot of their performance at the end of the 2nd sememster at University. This include discrete and continuous data that capture the various facets of the student. These include macroeconomic factors of inflation, GDP and unemployment rate. It covers the personal/family details of the student such as gender, previous grade, educational special needs, financial status, parents education, parents occupation. It captures aspects of the educational system such as courswork enrolled, day/evening classes, scholarships offered etc. The dataset is created by Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista from the Politechnic Institue of Portalegre. It was sourced from the UCI Machine Learning Repository and can be downloaded from [here](https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). Each row represent the detaining pertaining to an individual student and there are no duplicates. 


The orignal dataset exhibits three classes of students - Graduate, Enrolled and Dropout. For the binary classification question pursued in this project, the class Enrolled is omitted from the dataset. The prliminary EDA shows there are 2209 examples of Graduate students an 1421 examples of Dropouts. Thus the dataset imbalance is not a major concern and can be addressed through balancing techniques learnt in MDS program.  


### Analysis Roadmap


To discuss in group: (note: we need to do train, test split before EDA)


We plan to partition the dataset into training and test sets (80%: 20%). A detailed EDA will be perfomed to understand the distribution of the 36 features and their correlation. The insights from EDA will be used to either eliminate features to reduce redundancy or to create new features that may better explain the variance between the target classes. Box plots will be used on the continuous features such as Inflation rate, GDP and Unemployment rate. A t-test may be performed to draw further inference on whether these metrics come from separate distributions which could increase their feature importance. Pairwise scatter plots between the selected features are planned to develop an overall understanding of the features correlation.    


We aim to perform a perliminary modelling using the K-nearest neighbors, Naive Bayes, Logistic Regression, and Decision Tree classification algorithms to identify the best performing model. The chosen performance metrics for our problem statement are `Recall`, `f1 score` and `AUC` in the order of importance. Type 2 errors where actual dropouts are not identified reduce the usefulness of our machine learning. The Type 1 errors indicated by the precision of the model are of relatively lesser significance as actual graduates incorrectly classified as dropouts will provide a conservative model which is relatively acceptable.




### Results and Conclusions Roadmap


The hyperparameters of the aforementioned models will be optimized using cross-validation to determine the best estimator. The performance of these models will be tabulated in the report for comparison. The reasons for best estimator selection will be documented along will modelling assumptions and identified deficiencies. The train data will be refit on the best estimator and the final predictions will be made on the test data. The confusion matrix will be documented and included in the final report along with comments on any misclassifications and their affect on model performance.

The prelimary EDA performed as part of the Milestone 1 objectives can be found [here](https://github.com/UBC-MDS/dropout-predictions/blob/eda-caesar/src/dropout_pred_EDA.ipynb).
