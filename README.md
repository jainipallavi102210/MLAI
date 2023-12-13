# Access to Special Healthcare Services Among Children - Capstone Project

## Table of Contents
- [Access to Special Healthcare Services Among Children - Capstone Project](#access-to-special-healthcare-services-among-children---capstone-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data Source](#data-source)
  - [Understanding the Data](#understanding-the-data)
  - [Reading the Data](#reading-the-data)
  - [Understanding the Features](#understanding-the-features)
  - [Understanding the Task](#understanding-the-task)
  - [Feature Engineering](#feature-engineering)
  - [Train/Test Split](#traintest-split)
  - [Baseline Model](#baseline-model)
  - [Simple Model](#simple-model)
  - [Model Comparisons with default settings](#model-comparisons-with-default-settings)
  - [Improving the Model and the comparisons](#improving-the-model-and-the-comparisons)
  - [Recommendations](#recommendations)
  - [Getting Started](#getting-started)
  - [Folder Structure](#folder-structure)
    - [Running the Notebook](#running-the-notebook)
  - [Dependencies](#dependencies)
  - [Citation](#citation)

## Overview

This capstone project mainly focuses on the Children special health care services and utilizing the data from the National Survey of Children’s Health (NSCH). NSCH provides rich data on multiple, intersecting aspects of children’s lives—including physical and mental health, access to and quality of health care, and the child’s family, neighborhood, school, and social context. The National Survey of Children's Health is funded and directed by the Health Resources and Services Administration (HRSA) Maternal and Child Health Bureau (MCHB). A revised version of the survey was conducted as a mail and web-based survey by the Census Bureau in 2016, 2017, 2018, 2019, 2020, 2021 and 2022. Among other changes, the 2016 National Survey of Children’s Health started integrating two surveys: the previous NSCH and theNational Survey of Children with Special Health Care Needs (NS-CSHCN). See the MCHB website for more information on the 2016, 2017, 2018, 2019, 2020, 2021 and 2022 National Survey of Children's Health administration, methodology, survey content, and data availability. 

## Data Source

National Survey of Children’s Health, Health Resources and Services Administration, Maternal and Child Health Bureau. [here](https://mchb.hrsa.gov/data/national-surveys)

## Understanding the Data

The NSCH uses the CSHCN Screener to identify children with special health care needs. The Screener is a five item, parent-reported tool designed to reflect the federal Maternal and Child Health Bureau’s consequences-based definition of children with special health care needs. It identifies children across the range and diversity of childhood chronic conditions and special needs, allowing a more comprehensive and robust assessment of children's needs and health care system performance than is attainable by focusing on a single diagnosis or type of special need. All three parts of at least one screener question (or in the case of question 5, the two parts) must be answered “YES” in order for a child to meet CSHCN Screener criteria for having a special health care need.

More details are present on the codebooks provided

## Reading the Data

We'll load the dataset into our environment using pandas to prepare it for analysis.

## Understanding the Features

This dataset comprises various features providing information about households and children's health status. Key features include unique identifiers such as HHID (Unique HH ID), demographic information like SC_AGE_YEARS (Age of Selected Child), and indicators related to children's health needs (e.g., SC_K2Q10 - SC_K2Q23). It is essential to review the dataset for any missing values or the need for data type adjustments.

Some crucial steps in exploring this dataset include:

* Checking for missing values or unknown entries across the features.
* Examining the distribution of the target variable, such as CSHCN (Children with special health care needs), to understand its prevalence in the dataset.
* Understanding the types of data present in each column and assessing the uniqueness of categorical features if present
  
Analyzing this dataset is crucial for tasks such as data analysis and building predictive models to gain insights into factors influencing children's health and special healthcare needs.

## Understanding the Task

Predict the likelihood of children having special healthcare needs (SC_CSHCN) based on demographic factors

## Feature Engineering

In this section, we explore various features, assess their value distributions, identify outliers, evaluate their importance regarding the target variable, and determine if any features can be omitted from the analysis. We rely on visual aids like Heatmaps, histograms, and bar charts to perform these tasks.

For simplifying the training process, we eliminated features that didn't contribute to the target prediction.

## Train/Test Split

Dividing the data into training and testing sets to evaluate model performance.

## Baseline Model

A baseline model is a fundamental component in classification tasks as it provides a performance benchmark for advanced models. It serves to identify data issues, understand the problem's inherent complexity, and efficiently allocate resources. When a baseline model performs well, it signifies that the problem may not require complex algorithms, while poor performance indicates a need for more advanced modeling techniques or enhanced feature engineering. Additionally, a baseline model simplifies communication of results to stakeholders, making it a crucial step in the machine learning workflow.

For this dataset, baseline model used is LogisticRegression and baseline accuracy is around 77.312

## Simple Model 

The initial analysis involved the implementation of a basic Logistic Regression model to understand its behavior and gain insights into the dataset. This exploration uncovered an imbalance issue within the target variable. Subsequently, a detailed classification report was generated, along with a confusion matrix, providing a comprehensive view of the model's performance. 

## Model Comparisons with default settings

Comparing the performance of K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

**Logistic Regression:** The Logistic Regression model exhibited exceptional performance during training, achieving 100% accuracy. However, it slightly declined in test accuracy, reaching 99.99%. The training time for this model was relatively low, at 0.006988 seconds.

**K-Nearest Neighbors (KNN):** The KNN model demonstrated robust training accuracy of 98.81%, and its performance on the test set was also commendable, with an accuracy of 98.24%. The training time for KNN was reasonable, clocking in at 0.010437 seconds.

**Decision Tree:** The Decision Tree model showcased outstanding results, achieving 100% accuracy in both training and testing. Its training time was slightly higher than other models, at 0.023918 seconds, but the accuracy highlights its efficacy.

**Support Vector Machine (SVM):** The SVM model displayed strong training accuracy at 99.02% and maintained a high level of accuracy on the test set, scoring 98.89%. However, it required a longer training time compared to other models, with a duration of 10.79 seconds.

Each model's test accuracy performed better than the baseline accuracy of approximately 77.31%.

## Improving the Model and the comparisons

**Logistic Regression:**
- Best Hyperparameters: {'C': 0.01, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'}
- Best Train Score: 1.0
- Best Test Score: 1.0
- Top Features: 'qualcomm' (7.084334), 'mhneeds' (1.925120), 'SC_AGE_YEARS' (0.018890)

**K-Nearest Neighbors (KNN):**
- Best Hyperparameters: {'n_neighbors': 2, 'weights': 'distance'}
- Best Train Score: 0.9893876840652229
- Best Test Score: 0.9908390865639937 
- Top Features: 'qualnum' (0.0796), 'mhneeds' (0.0398), 'HHCOUNT' (0.0123), 'FAMCOUNT' (0.0116), 'rxmeds' (0.0043), 'MOMAGE' (0.0038)

**Support Vector Machine (SVM):**
- Best Hyperparameters: {'C': 0.01, 'kernel': 'linear'}
- Best Train Score: 1.0
- Best Test Score: 1.0

**Decision Tree:**
- Best Hyperparameters: {'criterion': 'gini', 'max_depth': 1, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
- Best Train Score: 1.0
- Best Test Score: 1.0
- Top Features: 'qualcomm' (1.0)

**Observations:** 
1. Logistic Regression and Decision Tree models achieved perfect accuracy on both training and test sets, suggesting potential overfitting.
2. K-Nearest Neighbors (KNN) performed well with a test accuracy of 99.08%, indicating a good generalization.
3. Support Vector Machine (SVM) also demonstrated perfect accuracy on both training and test sets.
4. Logistic Regression's top features include 'qualcomm', 'mhneeds', and 'SC_AGE_YEARS', with 'qualcomm' having the highest importance.
5. K-Nearest Neighbors (KNN) top features include 'qualnum', 'mhneeds', 'HHCOUNT', 'FAMCOUNT', 'rxmeds', and 'MOMAGE'.
6. Further analysis may be needed to address potential overfitting in Logistic Regression and Decision Tree models.

## Recommendations

Analysis may be needed to address potential overfitting in Logistic Regression and Decision Tree models.

**Regularization for Logistic Regression:**  Apply regularization techniques such as L1 or L2 regularization to penalize large coefficients and prevent overfitting. This can be achieved by adjusting the regularization parameter (C) in Logistic Regression.

**Consider Alternate Models:** Explore other machine learning algorithms that may be less prone to overfitting for your specific dataset. Experimenting with different models, such as ensemble methods or regularized linear models, can provide insights.


## Getting Started

To reproduce the analysis or contribute to the project:

1. Clone the repository to your local machine.
2. Run the provided Jupyter notebooks or scripts to reproduce the analysis.

## Folder Structure

- `data/data.csv`: Contains the raw and processed datasets.
- [capstone.ipynb](capstone.ipynb): Jupyter notebooks for data analysis and visualization.

### Running the Notebook

To run the analysis, ensure you have Python installed along with the necessary libraries. Execute the notebook sequentially to reproduce the analysis.

## Dependencies

No additional dependencies are needed to run ths project


## Citation

This DRC datasets are from the CAHMI Data Resource Center for Child and Adolescent Health.

Child and Adolescent Health Measurement Initiative (CAHMI) ([2016-2021]). [2016-2021] National Survey of Children's Health, [(SAS/SPSS/Stata)] Indicator dataset. Data Resource Center for Child and Adolescent Health supported by Cooperative Agreement U59MC27866 from the U.S. Department of Health and Human Services, Health Resources and Services Administration (HRSA), Maternal and Child Health Bureau (MCHB). Retrieved [09/03/2023] from childhealthdata.org  [here](https://www.childhealthdata.org/dataset)


