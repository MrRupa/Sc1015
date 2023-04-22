# SC1015: Data Science Mini Project - Stroke Prediction

School of Computer Science and Engineering

Nanyang Technological University

Lab: A140

Group:

## Members:
	
1. Ruparaj - [@MrRupa](https://github.com/MrRupa)
2. Quang - [@Quangdo0](https://github.com/Quangdo0)
3. Vu - [@VuTheAmser](https://github.com/VuTheAmser)

## About our project
The project of Data Science and Artificial Intelligence (SC1015) using stroke dataset in Kaggle (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) to explore and predict the occurence of stroke using attributes from dataset and models

## Description:

### Table of Contents:
1. Problem Formulation
2. [Data Extraction and Resampling](https://github.com/MrRupa/Sc1015/blob/37de7f199fd46d94c51e8990c349ad3687513222/Data%20Exraction%20and%20resampling.ipynb)
3. [Data Visualisation](https://github.com/MrRupa/Sc1015/blob/9aacca324845dfc257022e55c3744d9d48aa04c0/Data%20Exraction%20and%20resampling.ipynb)
4. [Data Splitting]()
5. [Logistic Regression]()
6. [Neural Network]()


### 1.  Problem Formulation
Our DataSet: Stroke Prediction Dataset | Kaggle 
Our Question: 
- Can we predict stroke based on the attributes in dataset
- Which models would perform best in predicting the occurrence of having stroke


### 2.Data Extraction and Resampling


### 3. Data Visualisation

We visualize the distribution of the stroke outcome variable in a bar chart, which shows the number of people who experienced a stroke and analyze the categorical variables in the dataset. For each of these variables, we display the count, unique values, top values, and their frequency.

Additionally, we visualize the stroke count by each categorical variable using bar charts and calculate the probability of having a stroke within each unique variable and display the results. This helps us understand the association between each variable and the likelihood of experiencing a stroke.

We also calculate the frequency of different types of variables and visualize the proportion of strokes by each categorical variable using bar plots.
And Lastly, we use the chi-squared test to determine if there is a significant association between the categorical variables and stroke outcomes. The p-values help us understand the significance of these associations.This data visualization and analysis parts provides an in-depth understanding of the stroke dataset, the relationships between various factors, and the potential implications of these factors on stroke outcomes.

### 4. Data Splitting
In this section, we performed, defined the numerical variable to meet the requiremrnt of the neural network. 
we use the train_test_split to split the data of 80% train and 20% test and assigned the train dataset to do data sampling for the four models:
- Original
- Random Oversampling 
- SMOTETomek Resampling
- SMOTEENN Resampling

we save all the resampled data in test.csv file accordingly.

### 5. Logistic Regression

In this section we used Logistic Regression model which is a statistical model used to predict the probability of a binary outcome based predictor variables. In the case of stroke prediction, the binary outcome could be whether a person is likely to experience a stroke or not, and the predictor variables could include factors such as age, gender and smoking status. Logistic regression models the relationship between the predictor variables and the probability of the binary outcome using a logistic function. The logistic function maps any real-valued input to a value between 0 and 1, which can be interpreted as the probability of the binary outcome. 

We divided the dataset into a training set and a testing set. The logistic regression model is trained on the training set, and its performance is evaluated on the testing set. The performance of the logistic regression model are evaluated using metrics:
- Accuracy
- Sensitivity
- Specificity,
- Area under the receiver operating characteristic curve (AUC-ROC). 

In summary, we manage derive an equation that would calculate a person's probability of getting stroke using Logistic Regression Model


### 6. Neural Network


## Conclusion

## What did we learn from this project?


## References




