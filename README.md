# Credit Risk Report

----------------------------------------
----------------------------------------

## Overview of the Analysis

* ##### *Purpose of the analysis:*
This analysis aims to evaluate the effectiveness of a logistic regression machine learning model in predicting loan risk. The goal of the logistic regression model was to be able to accurately classifiy and differentiate healthy loans(0) and high-risk loans(1) based on seven different financial features of a loan.  

* ##### *Financial information needed to train the logistic regression model and predict classifications:*
The model utilized historical loan data containing financial information about borrowers. The target variable that was used to predict was the loan status, categorized as either "healthy loans" (0) or "high-risk loans" (1). The other seven columns from the dataset were used as features to train the model ("loan size", "interest rate", "borrower income", "debt-to-income", "num-of-accounts", "derogatory marks", and "total debt").

* ##### *Information about target variable for predicting (loan status):*
An initial exploration of the data revealed a significant class imbalance. There were 75,036 healthy loans (class 0) compared to only 2,500 high-risk loans (class 1), resulting in a ratio of 30 healthy loans to every 1 high-risk loan. To address this imbalance and ensure the model learns from both classes effectively, stratification was used during training. This technique ensures an even distribution of classes (0s and 1s) in both the training and testing sets. 

* ##### *Stages of the machine learning process used for this analysis:*
###### **1) Data Preprocessing:**
The first step was indentifying that the target variable being used for predictions was a categorical variable and assiging that as the y (loan status). Then using the other features as the X to train the model to find patterns to predict y (AKA - loan status). In the case of this model, there was no scaling done to the dataset but it wouldnt hurt or result in worse results. The other important aspect to check was the imbalance between the healthy loans(0) and high-risk loans(1) in loan status column which the model is being trained to predict. 

###### **2) Instantiate Model:**
Instantiate/create the model to be trained

###### **3) Fit/Train Model:**
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)` was used to split the training and testing data into a 75% training and 25% testing split of the dataset. `stratify = y` was used to account for the imbalance between in the healthy and high-risk loans in the loan status column to ensure an even distribution of classes in the training and testing sets. The model was then fit/trained using `model.fit(X_train, y_train)` on the training set. 

###### **4) Predict/Use Model:**
`test_predictions = model.predict(X_test)` was then used to see how the model would preform based on the new information (AKA: data the model hasnt seen yet) and how well it would be able to predict/classify healthy vs high-risk loans based on the freatures it was trained on. 

###### **5) Evaluate/Score Model:** 
Evaluation metrics like accuracy, precision, recall, and F1-score were used to compare model performance.

## *Algorithms Used:*
* **Logistic Regression**
* **K-Nearest Nieghbors**
* **Random Forest**

----------------------------------------
----------------------------------------

## Results

#### *Logistic Regression Model*
* **Accuracy:** 0.99 (achieved a very high overall accuracy)
* **Precision (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Recall (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **F1 Score (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Precision (High-Risk Loans-1):** 0.89 (correctly classified 89% of high-risk loans out of all the loans the model predicted as high-risk)
* **Recall (High-Risk Loans-1):** 0.87 (correctly identified 87% of high-risk loans, misclassifying 13% of high-risk loans as healthly loans)
* **F1 Score (High-Risk Loans-1):** .88 (correctly predicted 88% of the high-risk loans (high recall) while also avoiding a significant number of false positives (high precision))

#### *K-Nearest Nieghbors Model*
* **Accuracy:** 0.99 (achieved a very high overall accuracy)
* **Precision (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Recall (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **F1 Score (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Precision (High-Risk Loans-1):** 0.92 (correctly classified 92% of high-risk loans out of all the loans the model predicted as high-risk)
* **Recall (High-Risk Loans-1):** 0.88 (correctly identified 88% of high-risk loans, misclassifying 12% of high-risk loans as healthly loans)
* **F1 Score (High-Risk Loans-1):** .90 (correctly predicted 90% of the high-risk loans (high recall) while also avoiding a significant number of false positives (high precision))

#### *Random Forest Model*
* **Accuracy:** 0.99 (achieved a very high overall accuracy)
* **Precision (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Recall (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **F1 Score (Healthy Loans-0):** 1.0 (perfectly identified healthy loans)
* **Precision (High-Risk Loans-1):** 0.88 (correctly classified 88% of high-risk loans out of all the loans the model predicted as high-risk)
* **Recall (High-Risk Loans-1):** 0.88 (correctly identified 88% of high-risk loans, misclassifying 12% of high-risk loans as healthly loans)
* **F1 Score (High-Risk Loans-1):** .88 (correctly predicted 88% of the high-risk loans (high recall) while also avoiding a significant number of false positives (high precision))

----------------------------------------
----------------------------------------

## Summary

* ##### *Best performing model and recommendation:*
The best performing model overall was the K-Nearest Nieghbors model. Although all three of the models had very high accuracy of 99%, the K-Nearest Nieghbors model had higher precision (92%), higher recall (88%), and a better F1 score of (90%) than both the Logisitic Regression and Random Forest models for predicting high-risk loans. I would recommend using the K-Nearest Nieghbors model because the slight percent increases in the precision, recall, and F1 score compared to the other two models can result in significant financial benefits and reduce financial loss for a company planning to use this model for predicting credit risk. 

* ##### *Importance of performance of the model:*
In the context of loan risk assessment, prioritizing high recall for the high-risk loans(1) is critical. Misclassifying even a small percentage (13% in this case) of high-risk loans can lead to significant financial losses due to defaults. While a perfect score (100% recall and precision) for healthy loans(0) is desirable, it carries less financial weight compared to missed high-risk identifications. A borrower deemed high-risk who ultimately repays the loan represents a positive outcome for the company. Ultimately, the choice of model depends on the desired level of accuracy and the company's risk tolerance. In some cases, human oversight for high-risk predictions might be warranted.