# Hands-On Application of Generative AI Models for Programming Tasks

## Introduction
The following prompts are designed to guide you through various stages of data exploration and machine learning workflows using Python. 
The prompts cover programming tasks, including data exploration, preprocessing, feature selection, model training, and evaluation, with a focus on machine learning, clustering, and deep learning.
This resource serves as a hands-on guide for leveraging AI-driven programming assistance in real-world applications.

******
### 🌐 **Generative A.I Model to be used:** ChatGPT

### 📥 **Dataset:** Download the two files [here](https://bit.ly/Practical2_Data)
### Dataset Overview

📌 **coldata.csv (Clinical & Metadata)**

+ Contains patient-level clinical, demographic, and molecular subtype data.
+ Key columns include tumor characteristics, staging, follow-up details, survival status, and molecular classifications (e.g., PAM50 subtypes).
+ The paper_BRCA_Subtype_PAM50 column provides breast cancer molecular subtypes, crucial for stratifying patients in research.
  
📌 **brca_matrix.csv (Gene-Level Data)**
+ Contains gene expression data for the same patients.
+ Each row represents a patient, and each column represents a gene

**How These Datasets Are Connected**

+ Common Identifier: Both datasets can be linked using a patient barcode or ID, ensuring that clinical information aligns with molecular profiles.
+ Integration Purpose:
  - coldata.csv provides patient metadata, clinical outcomes, and molecular subtypes.
  - brca_matrix.csv provides high-dimensional gene expression signatures.
  - Combining them allows researchers to correlate clinical outcomes with genetic and molecular alterations.

## 🔧 Task 1: Exploring the Data
 + Use ChatGPT to understand the structure and type of the datasets (the two files) that you have been provided with.
 + Examine the first few rows and statistical summary using .head() and .describe().
 + Use ChatGPT to generate Python codes to visualize the distribution of the target variable (diagnosis).

<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "I am new to Python and want to analyze a gene expression dataset for breast cancer subtyping using machine learning. What are the essential Python libraries I need to install?"
  - **Prompt 2: 💭** "I have installed the libraries. How do I load a CSV file into Python using Pandas?"
  - **Prompt 3: 💭** "I have installed the libraries. What are these files about? How do I load a CSV file into Python using Pandas?"
  - **Prompt 4: 💭** "How can I inspect the structure of my dataset? What commands should I use to check the first few rows and column types?"
  - **Prompt 5: 💭** "I want to check if there are missing values in my dataset. What Python function can help with that?"
  - **Prompt 6: 💭** "I want to understand the basic statistics of my dataset (mean, median, standard deviation). How do I generate a summary table?"
  - **Prompt 7: 💭** "I need to visualize the distribution of the target variable (breast cancer subtype). Can you explain how to do that using Seaborn or Matplotlib?"

</details>

## 🔧 Task 2: Data Processing
+ Identify data processing steps you need to do to use the dataset for a machine learning tasks such as encoding categorical data.
+ Standardize the dataset’s features to ensure they are on the same scale. Use libraries such as Scikit-learn’s StandardScaler.

<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "What is data preprocessing, and why is it important for machine learning?"
  - **Prompt 2: 💭** "My dataset contains both categorical and numerical data. What should I do to prepare it for machine learning?"
  - **Prompt 3: 💭** "How do I identify which columns are categorical and which are numerical?"
  - **Prompt 4: 💭** "What is one-hot encoding, and how can I apply it to categorical columns in my dataset?"
  - **Prompt 5: 💭** "How do I standardize numerical features so that they are on the same scale? Can you explain what StandardScaler does?"
  - **Prompt 6: 💭** "Can you generate a Python script to apply one-hot encoding to categorical features and StandardScaler to numerical features?"

</details>

## 🔧 Task 3: Feature Selection
+ Ask ChatGPT how to choose the most relevant features for model training.
+ Use Lasso (L1 regularization) for feature selection.
+ Filter the dataset to include only the selected features.


<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "What is feature selection, and why is it important in machine learning?"
  - **Prompt 2: 💭** "What are some common techniques for feature selection?"
  - **Prompt 3: 💭** "I heard about Lasso (L1 regularization). How does it help in feature selection?"
  - **Prompt 4: 💭** "Can you generate a Python script to apply Lasso regression and select the most important features?"
  - **Prompt 5: 💭** "How do I update my dataset to include only the selected features?"

</details>

## 🔧 Task 4: Splitting the Dataset

Before a machine learning model can make predictions, it must be trained on a set of data to learn an approximation function. 
Use Scikit-learn’s train_test_split Function 
+ Split the dataset into training (70%) and testing (30%) sets.
+ Set the random_state to ensure reproducibility.

<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "Why do we need to split the dataset into training and testing sets?"
  - **Prompt 2: 💭** "What is the difference between training and testing datasets?"
  - **Prompt 3: 💭** "How do I use train_test_split from Scikit-learn to divide my dataset into 70% training and 30% testing?"
  - **Prompt 4: 💭** "What is the random_state parameter, and why is it important?"
  - **Prompt 5: 💭** "Can you provide a Python script to split my dataset while ensuring reproducibility?"

</details>

## 🔧 Task 5: Classification
You will use the following machine learning models for prediction
+ [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
+ [Support vector machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
+ [Decision tree classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
+ [Random Forest Classifier](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "What is classification in machine learning?"
  - **Prompt 2: 💭** "Which classification algorithms should I consider for predicting breast cancer subtypes?"
  - **Prompt 3: 💭** "Can you explain how Logistic Regression works and provide a Python script to train a model?"
  - **Prompt 4: 💭** "How does Support Vector Machine (SVM) work, and can you provide a Python script to implement it?"
  - **Prompt 5: 💭** "What is a Decision Tree, and how does it work in classification?"
  - **Prompt 6: 💭** "How do I train and evaluate a Random Forest Classifier?"
  - **Prompt 7: 💭** "For the Decision Tree model, how does changing the random_state affect the results? Can you generate scripts for random_state = 0 and 42?"
  - **Prompt 8: 💭** "How do I evaluate my model using accuracy, precision, recall, and F1-score? Can you generate a Python script for that?"

</details>

## 🔧 Task 6: Clustering Task
Clustering is a type of unsupervised learning technique used to group data points or objects based on their similarity. The goal of clustering is to identify inherent patterns or structures in the data without prior knowledge of true labels. Clustering algorithms partition the data into groups or clusters such that data points within the same cluster are more similar to each other than to those in other clusters.

<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "What is clustering, and how is it different from classification?"
  - **Prompt 2: 💭** "Can you explain different clustering algorithms and their applications?"
  - **Prompt 3: 💭** "What is K-means clustering, and how does it work?"
  - **Prompt 4: 💭** "What is Agglomerative Clustering, and when should it be used?"
  - **Prompt 5: 💭** "Can you provide Python scripts to perform both K-means and Agglomerative Clustering with 2 clusters?"
  - **Prompt 6: 💭** "How do I evaluate clustering results using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score?"

</details>

## 🔧 Task 7: Getting Optimal Number of Clusters
Determining the optimal number of clusters is essential for effective clustering analysis. A well-chosen number of clusters ensures that data points within a cluster are similar while maintaining clear distinctions between different clusters.

<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "How can I determine the best number of clusters in K-means clustering?"
  - **Prompt 2: 💭** "Can you explain the Elbow Method and how to implement it in Python?"
  - **Prompt 3: 💭** "How does Silhouette Analysis help in choosing the right number of clusters?"
  - **Prompt 4: 💭** "What is the Gap Statistic, and how do I use it in clustering?"
  - **Prompt 5: 💭** "How can I use Hierarchical Clustering to determine the optimal number of clusters?"
  - **Prompt 6: 💭** "Can you generate Python scripts to implement and visualize the results for each method?"

</details>

## 🔧 Task 8: Deep Learning
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to automatically learn patterns from data. It extracts complex features without manual intervention. Unlike traditional machine learning, which relies on handcrafted features and structured data, deep learning can handle unstructured data like images and text, learning hierarchical representations through deep networks. 
<details>
  <summary>Prompts</summary>

  - **Prompt 1: 💭** "What is a multi-layer perceptron (MLP), and how does it work for binary classification?"
  - **Prompt 2: 💭** "Which Python libraries should I use to build a deep learning model?"
  - **Prompt 3: 💭** "How do I define an MLP architecture using Keras?"
  - **Prompt 4: 💭** "How do I split my data for training and validation in deep learning?"
  - **Prompt 5: 💭** "Can you generate a Python script to train an MLP model with a validation split of 20%, 100 epochs, and batch size of 32?"
  - **Prompt 6: 💭** "How do I evaluate my deep learning model using a classification report?"
  - **Prompt 7: 💭** "How can I tune my model’s hyperparameters using Grid Search?"

</details>
