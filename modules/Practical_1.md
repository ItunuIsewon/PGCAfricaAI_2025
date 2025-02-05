# Practical Session 1 – Hands-On Workshop (Machine Learning)

## Dataset Overview
This synthetic dataset consists of 1,000 simulated individuals with a combination of demographic, clinical, and 
genomic features relevant to population genetics and association studies. 
It contains 1,013 variables, including age, sex, and cohort information, representing different population groups. 
Clinical data include measurements such as systolic and diastolic blood pressure, LDL and HDL cholesterol levels, height, weight, and body mass index.

******
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('synthetic_data_clusters_v11.csv')
```

```python
# Select only clinical data (excluding SNP columns)
clinical_data = data.loc[:, ~data.columns.str.startswith('SNP_')]

# Display the first few rows
print(clinical_data.head())
```

```python
clinical_data.columns
```

<details>
  <summary>Output</summary>
<code>Index(['age', 'sex', 'cohort', 'systolic_BP', 'diastolic_BP', 'LDL_cholesterol', 'HDL_cholesterol', 'height', 'weight', 'BMI', 'waist_circumference', 'hip_circumference', 'WHR'], dtype='object')
</code>
</details> 

<details>
What Are Categorical Variables? Categorical variables represent discrete categories or labels, rather than numerical values. They can be classified into:
    
1. **Nominal Variables** – Categories with no inherent order (e.g., colours: "Red," "Blue," "Green").
2. **Ordinal Variables** – Categories with a meaningful order (e.g., education levels: "High School," "Bachelor’s," "Master’s," "PhD"). 

Why Convert Categorical Variables to Numerical in Machine Learning? Most machine learning algorithms require numerical input because they rely on mathematical computations like distance calculations, matrix operations, and statistical techniques. 
  
**Here’s why categorical variables must be converted:**
+ **Mathematical Operations** – Algorithms like linear regression and support vector machines (SVMs) require numerical input to perform calculations.
+ **Distance-Based Algorithms** – Models like k-NN and K-Means use Euclidean distance, which only works with numerical values.
+ **Gradient-Based Optimization** – Neural networks and gradient-boosting methods rely on numerical computations for backpropagation and optimization.
+ **Better Model Performance** – Encoding categorical variables into meaningful numerical values can improve model interpretability and accuracy. 

Common Methods to Convert Categorical Variables 
- **Label Encoding** – Assigns unique integers to categories (e.g., "Red" → 0, "Blue" → 1). 
- **One-Hot Encoding** – Converts categories into binary columns (e.g., "Red" → [1,0,0], "Blue" → [0,1,0]).
- **Ordinal Encoding** – Assigns ordered numerical values to ordinal data (e.g., "Low" → 1, "Medium" → 2, "High" → 3).
- **Target Encoding** – Replaces categories with their mean target values, useful in predictive modelling.
  
</details>


```python
clinical_data.loc[:, 'sex'] = clinical_data['sex'].map({'Female': 0, 'Male': 1})
clinical_data.loc[:, 'cohort'] = clinical_data['cohort'].map({'Ugandan': 0, 'Zulu': 1})
```

<details><summary> Why Do We Standardize Data?: </summary>
Standardization is a common preprocessing step in machine learning and data analysis. It involves scaling the data to have a specific mean and variance. Specifically, we aim to make the features have a mean of 0 and a standard deviation of 

The primary reasons for standardizing data are:

**1. Improves Model Performance:**

Many machine learning algorithms, especially those based on distance metrics (e.g., K-Nearest Neighbors, Support Vector Machines) or optimization algorithms (e.g., gradient descent in Logistic Regression, Neural Networks), work better when the features have similar scales.
Without standardization, features with larger ranges (e.g., weight in kg vs. age in years) can dominate the learning process and lead to biased or poor performance.

**2. Ensures Equal Weight:**

Standardizing ensures that all features contribute equally to the model. If one feature has a much larger scale (like income in thousands of dollars vs. age in years), the model might give more importance to that feature simply because of its magnitude.
Standardization removes this bias by transforming the features to a comparable scale.

**3. Stabilizes Gradient Descent:**

Gradient-based algorithms (e.g., Logistic Regression, Neural Networks) perform better with standardized data, as it helps prevent the algorithm from oscillating or converging slowly due to large differences in feature scales.

**4. Assumption of Many Models:**

Some machine learning models (e.g., Linear Regression, PCA, SVMs) assume that the data is centered around 0 with unit variance. Standardization ensures this assumption holds.

**What Does the StandardScaler Do?**
The StandardScaler from sklearn.preprocessing standardizes the data by transforming each feature so that it has:

Mean = 0
Standard Deviation = 1
Mathematically, for each feature 𝑥(column) in the dataset:

1. Calculate the Mean
2. Calculate the Standard Deviation
3. Transform Each Value
for more information visit [standard_scaler](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html), [geeks4geeks](https://www.geeksforgeeks.org/what-is-standardscaler/)
  
</details>

```python
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clinical_data)
```

<details><summary>Principal Component Analysis (PCA) </summary>
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics. It transforms high-dimensional data into a lower-dimensional space while preserving as much variance (information) as possible. 

**PCA helps in:** Reducing computational complexity, Removing multicollinearity and Improving visualization of high-dimensional data.


**How PCA Works**
+ **Standardization** – The data is standardized (zero mean, unit variance) to ensure all features contribute equally.
+ **Covariance Matrix Computation** – The relationships between features are analyzed using a covariance matrix.
+ **Eigenvalue & Eigenvector Computation** – The principal components (new axes) are determined from eigenvectors of the covariance matrix.
+ **Selecting Principal Components** – Components are ranked based on their explained variance, and only the most important ones are retained.
+ **Transformation** – The original data is projected onto the selected principal components.

</details>

```python
# Perform PCA
pca = PCA()
pca.fit(scaled_data)
```

<details><summary>  Cumulative Variance in PCA</summary>
Cumulative variance explains how much of the total variance in the dataset is retained when selecting a given number of principal components.

**Why Is Cumulative Variance Important?**
It helps determine how many principal components to keep for a good balance between dimensionality reduction and data retention.
Usually, we select the smallest number of components that explain a high percentage (e.g., 95%) of the variance.

**How to Calculate Cumulative Variance**
1. Compute the explained variance ratio for each principal component.
2. Compute the cumulative sum of these explained variance ratios
</details>

```python
# Cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Print cumulative variance explained
for i, variance in enumerate(cumulative_variance, start=1):
    print(f"Principal Component {i}: {variance:.4f}")
```

<details><summary>Output:</summary>
Principal Component 1: 0.2629
Principal Component 2: 0.5023
Principal Component 3: 0.6124
Principal Component 4: 0.6932
Principal Component 5: 0.7718
Principal Component 6: 0.8370
Principal Component 7: 0.8945
Principal Component 8: 0.9429
Principal Component 9: 0.9902
Principal Component 10: 0.9971
Principal Component 11: 0.9994
Principal Component 12: 0.9999
Principal Component 13: 1.0000
</details>

```python
# Plot cumulative variance explained
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Variance Explained by PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()
```

What threshold should do you think is appropriate for choosing principal components?
A useful information PCA gives is the number of clusters(groups) that are in the data

```python
# Transform the data into principal components
pca_components = pca.transform(scaled_data)

# Extract PC1 and PC2
pc1 = pca_components[:, 0]
pc2 = pca_components[:, 1]
```

```python
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Plot for Sex: Male vs Female
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pc1, pc2, 
                      c=data['sex'].apply(lambda x: 0 if x == 'Male' else 1), 
                      cmap='coolwarm', alpha=0.6)

# Create custom legend
legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female')]

plt.title('PCA: PC1 vs PC2 (Colored by Sex)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=legend_labels, loc='best')
plt.grid(True)
plt.colorbar(scatter, label='Sex')
plt.show()

# Plot for Cohort: Ugandan vs Zulu
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pc1, pc2, 
                      c=data['cohort'].apply(lambda x: 0 if x == 'Ugandan' else 1), 
                      cmap='viridis', alpha=0.6)

# Create custom legend for cohort
legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Ugandan'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Zulu')]

plt.title('PCA: PC1 vs PC2 (Colored by Cohort)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=legend_labels, loc='best')
plt.grid(True)
plt.colorbar(scatter, label='Cohort')
plt.show()
```

<details><summary>Loadings in PCA </summary>
  
In Principal Component Analysis (PCA), **loadings** represent how strongly each original variable contributes to a given principal component (PC).

Mathematically, loadings are the coefficients of the eigenvectors of the covariance or correlation matrix.
Interpretation: A high loading (positive or negative) means that the variable strongly influences that principal component.
Range: Typically between -1 and 1 (when using correlation-based PCA).

**Why Are Loadings Important?**
1. **Feature Interpretation** – Helps understand which variables are most influential in forming each principal component.
2. **Dimensionality Reduction** – Identifies which features contribute the most, allowing for variable selection.
3. **Pattern Discovery** – Reveals relationships between variables by clustering correlated features in the same principal component.
4. **Visualization** – Loadings are used in biplots, where both original variables and principal component scores are plotted together.

**What Information Do PCA Loadings Provide?**
1. **Correlation Between Variables and PCs** – Loadings indicate how much each original feature is correlated with the new principal components.
2. **Direction of Influence** – Positive or negative loadings show whether variables move together or in opposite directions.
3. **Grouping of Variables** – Variables with similar high loadings on a component are likely related.

**How Are PCA Loadings Used in a Biological Context?**
In biology and bioinformatics, PCA loadings can help:

+ **Genomics & Transcriptomics**
Identifying genes or proteins that contribute most to biological variation.
Discovering gene expression patterns that differentiate disease vs. healthy states.

+ **Metabolomics & Proteomics**
Finding key metabolites or proteins that explain differences between samples (e.g., healthy vs. diseased).
Identifying biomarkers for diagnostics.

+ **Ecology & Evolutionary Biology**
Grouping species or populations based on traits and genetic markers.
Understanding the impact of environmental variables on species distribution.

+ **Medical Research**
PCA loadings can reveal biological pathways most affected by diseases.
Helps in drug discovery by identifying the most important biochemical features.

**Example:** Using Loadings in a Biological Study
Let’s say PCA is applied to gene expression data from cancer patients. The first principal component (PC1) explains most of the variance in the dataset.

A high positive loading for a specific gene means it is highly expressed in most samples.
A high negative loading means the gene is downregulated.
If PC1 separates cancerous vs. normal tissue, genes with high loadings could be potential biomarkers.
</details>

```python
# Get the loadings (principal component coefficients)
loadings = pca.components_

# Display the loadings for each principal component
loading_df = pd.DataFrame(loadings, columns=clinical_data.columns, index=[f'PC{i+1}' for i in range(loadings.shape[0])])



# Plot the loadings for the first few principal components
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(loading_df.T, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('PCA Loadings for Each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variables')
plt.show()
```

<details><summary>PCA PLOT</summary>
From our PCA plot, we know that there are four clusters, however, we do not know what group each individual falls into. clustering algorithms can enable us to assign individuals to groups.

Clustering is a type of unsupervised learning technique used to group data points or objects based on their similarity. The goal of clustering is to identify inherent patterns or structures in the data without prior knowledge of true labels. 
Clustering algorithms partition the data into groups or clusters such that data points within the same cluster are more similar to each other than to those in other clusters.

**K-means Clustering:** K-means is a popular centroid-based clustering algorithm. It partitions the data into K clusters by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of data points assigned to each cluster. 
[sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

</details>

```python
from sklearn.cluster import KMeans

# Apply K-means clustering (for example, 3 clusters for risk stratification)
kmeans = KMeans(n_clusters=4, random_state=42)
clinical_data = clinical_data.copy()
clinical_data.loc[:, 'cluster'] = kmeans.fit_predict(scaled_data)
```

.value_counts() is a Pandas function used to count the occurrences of unique values in a Series (a single column of a DataFrame). It’s commonly used for analyzing categorical or discrete numerical data.

```python
clinical_data['cluster'].value_counts()
```

```python
clinical_data
```

<details><summary>Why Do We Need Cluster Profiles?</summary>

A cluster profile helps summarize and interpret the characteristics of each cluster after performing clustering (e.g., K-Means, Hierarchical, DBSCAN). It provides meaningful insights into the underlying patterns within the data.

**Key Reasons for Creating Cluster Profiles**
1. **Understanding Cluster Characteristics**
Helps determine what makes each cluster distinct by computing summary statistics (e.g., mean, median) for each feature.
**Example: **In a medical dataset, clusters may represent different patient groups based on clinical parameters.

2. **Feature Importance in Clustering**
Identifies which features contribute the most to the formation of clusters.
**Example:** If one cluster has high cholesterol levels, it might indicate high-risk patients.

3. **Cluster Interpretation & Labeling**
Instead of just numerical cluster labels (0, 1, 2…), profiles help describe each cluster in a meaningful way.
**Example:**
Cluster 0 → "Young & Healthy"
Cluster 1 → "Elderly with High Blood Pressure"
Cluster 2 → "Diabetic Patients"

**Validating Clustering Results**
Helps assess if the clustering makes real-world sense.
If clusters are too similar, adjust parameters (e.g., number of clusters, distance metric).

**Business & Clinical Decision-Making**
Helps researchers, or clinicians take targeted actions based on the cluster characteristics.
**Example:** Tailored treatment plans for different patient clusters.
</details>

```python
# Group data by cluster and compute summary statistics
cluster_profile = clinical_data.groupby('cluster').mean()
cluster_profile
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'cluster_profile' is your DataFrame with mean values for each cluster
# You can customize the features you want to visualize
# Convert all columns to numeric (if applicable)
cluster_profile = cluster_profile.apply(pd.to_numeric, errors='coerce')

# Plotting the cluster profiles (mean of each feature for each cluster)
plt.figure(figsize=(10, 3))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
plt.title('Cluster Profiles - Mean Feature Values by Cluster')
plt.xlabel('Features')
plt.ylabel('Clusters')
plt.show()
```
Can you identify unique patterns in the clusters based on these cluster profiles

```python
clinical_data = clinical_data.copy()
# Add PCA components to dataframe
clinical_data.loc[:,'PC1'] = pc1
clinical_data.loc[:,'PC2'] = pc2

# Plot PCA with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=clinical_data, hue=clinical_data['cluster'], palette='Set2', alpha=0.7)
plt.title('PCA Visualization of Clusters')
plt.legend(title='Cluster')
plt.show()
```

<details>
**Classification in Machine Learning**
Classification is a type of supervised learning where the goal is to predict the category or class that a given input belongs to, based on patterns learned from labeled data. The output variable (target) is categorical, which means it can take one of a limited number of distinct values (e.g., "spam" or "not spam," "disease" or "no disease").

**Training and Test Datasets**
When building a machine learning model, the data is typically split into two parts: training and test datasets.

**Training Dataset:**
This is the subset of the data used to train the model. It includes both the input features (X) and the corresponding output labels (y).
The model learns from the patterns and relationships within this dataset to make predictions.
The training process involves adjusting the model's internal parameters (e.g., weights in a neural network) to minimize prediction errors.

**Test Dataset:**
The test dataset is a separate subset of the data that the model has never seen during training.
It is used to evaluate the model's generalization ability—how well it can predict new, unseen data.
This helps determine whether the model is overfitting (memorizing training data) or underfitting (not learning enough).
A typical practice is to split the data into training and test sets, with 70-80% of the data for training and the remaining 20-30% for testing. This can be done using train-test splits or cross-validation for more robust performance assessment.

**Random Forest in Machine Learning**
Random Forest is a powerful ensemble learning algorithm used for both classification and regression tasks. It builds multiple decision trees during training and combines their outputs to improve accuracy and control overfitting. Here’s how it works:

**How Random Forest Works:**

1. **Bootstrap Sampling:**
The model randomly samples the training data with replacement to create multiple subsets of the data (bootstrap samples). Each decision tree in the forest is trained on a different subset of the data.

2. **Building Decision Trees:**
For each sample, a decision tree is built. Each node in the tree splits the data based on the most informative feature.
During tree construction, random subsets of features are considered for splitting each node (instead of considering all features), which introduces diversity among trees and reduces correlation between them.

3. **Voting for Classification:**
Once all the trees are trained, for a new input, each tree in the forest makes a prediction (classification decision).
The final prediction is made by taking the majority vote of all the individual tree predictions.

**Advantages of Random Forest:**

**1. Reduces Overfitting:** Because it averages the predictions from many trees, it reduces the risk of overfitting that might happen with a single decision tree.

**2. Handles Large Datasets:** Random Forests can handle large datasets with higher dimensionality (many features).

**3. Feature Importance:** It can calculate the importance of each feature, helping to identify which variables are the most informative.

**4. Resistant to Noise:** Because of the averaging process, Random Forest can tolerate noise in the data.


**Limitations:**

**1. Interpretability:** Random Forests are often considered a "black-box" model, meaning it's harder to interpret how the final decision is made.
**2. Computational Complexity:** Random Forest can be computationally intensive because it involves building multiple decision trees.

</details>

<details>
**Understanding 0-1-2 SNP Data Encoding in Genetics** 
  
In genomic studies, single nucleotide polymorphisms (SNPs) are variations in a single nucleotide (A, T, C, or G) at a specific position in the genome. The 0-1-2 encoding is a numerical representation of SNP genotypes that simplifies their use in statistical and machine learning models.
Each individual has two copies of each chromosome (one from each parent), meaning they inherit two alleles at every SNP position. The 0-1-2 encoding is based on the number of copies of a specific minor allele (the less frequent allele in the population):

📌 **Example SNP Genotype Encoding**
If an SNP has two possible alleles: A (major) and G (minor):

An individual with AA will be encoded as 0

An individual with AG will be encoded as 1

An individual with GG will be encoded as 2

**Why Use 0-1-2 Encoding?**
1. Numeric Representation: Machine learning and statistical models work better with numerical inputs rather than categorical A/T/C/G labels.
2. Linear Interpretability: Many models assume a linear relationship between the number of risk alleles and disease risk.
3. Efficient Storage & Computation: Instead of storing two separate allele columns, a single integer column is more compact.
   
</details>

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Extract SNP features (columns that start with 'SNP_')
snp_data = data.loc[:, data.columns.str.startswith('SNP_')]

snp_data
```

```python
# Target variable: clusters from previous K-Means
y = clinical_data['cluster']  # Use the clusters as labels
```

```python
# Standardize SNP features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(snp_data)
```

```python
# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
```

```python
# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
```

```python
# Predict on test data
y_pred = clf.predict(X_test)
```

```python
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

<details>
  
**Evaluation Metrics**
**1. Model Accuracy:** 0.8400
Accuracy indicates that the model correctly classified 84% of all instances across the 4 classes. This is a solid result, but depending on the dataset and problem, you may want to focus more on precision, recall, and F1-scores (especially if the classes are imbalanced).

**2. Classification Report**
The classification report gives a more detailed view of your model’s performance for each class.

**Precision:** The percentage of correct positive predictions out of all positive predictions for that class.
Class 0: 83% of the predictions that were labeled as Class 0 were correct.
Class 1: 97% of the predictions for Class 1 were correct, but the model struggles with recall.
Class 2: Perfect prediction (100% precision and recall), indicating your model is excellent at identifying Class 2.
Class 3: The precision of 67% means the model sometimes incorrectly predicts Class 3.

**Recall:** The percentage of actual instances of each class that were correctly predicted.
Class 1 has a lower recall (69%), meaning the model misses a fair number of Class 1 samples.
Class 3 has a recall of 89%, indicating the model does a good job of identifying Class 3.


**F1-Score:** The harmonic mean of precision and recall. It balances the two metrics, making it useful when precision and recall are imbalanced.
Class 2 has the highest F1 score (1.00), indicating near-perfect performance.
Class 1 and Class 0 have balanced F1-scores (~0.80), which suggests a trade-off between precision and recall.
Class 3 has a moderate F1-score (0.76), which could benefit from more tuning.

**3. Confusion Matrix**
The confusion matrix shows the actual vs. predicted class counts, which can help further interpret the performance:
Row 0 (Actual Class 0):
40 instances were correctly predicted as Class 0.
12 instances were misclassified as Class 3.
Row 1 (Actual Class 1):
31 instances were correctly predicted as Class 1.
3 instances were misclassified as Class 0, and 11 as Class 3.
Row 2 (Actual Class 2):
50 instances were perfectly predicted as Class 2.
Row 3 (Actual Class 3):
47 instances were correctly predicted as Class 3.
5 instances were misclassified as Class 0, and 1 as Class 1.
</details>

<details>

  **What is Hyperparameter Tuning?**
  Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance. Unlike parameters (which are learned from data, such as weights in neural networks), hyperparameters are set manually before training begins.

**Why do we need hyperparameter tuning?**
1. Improves model performance: A poorly chosen hyperparameter set can lead to underfitting or overfitting.
2. Avoids overfitting/underfitting: Helps balance model complexity.
3. Enhances generalization: Helps the model perform well on unseen data.


**What is Grid Search?**
Grid Search is a method for systematically searching through a predefined set of hyperparameters to find the best combination. It evaluates all possible combinations using cross-validation and selects the one that gives the best performance.

**How Grid Search Works:**
1. Define a hyperparameter grid: Specify possible values for each hyperparameter.
2. Train multiple models: Each combination of hyperparameters is used to train a model.
3. Evaluate models using cross-validation: The model performance is assessed on validation data.
4. Select the best combination: The best-performing set of hyperparameters is chosen.
</details>

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


snp_data = data.loc[:, data.columns.str.startswith('SNP_')]

# Target variable: clusters from previous K-Means
y = clinical_data['cluster']  # Use the clusters as labels

# Standardize SNP features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(snp_data)

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train final model with best parameters
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Evaluate the optimized model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
How does the new result compare to the result before hyperparameter tuning

```python
#pip install shap
#!pip install numba==0.52.0rc2
```

<details>
  
  **Explainable AI (XAI) & SHAP**
Explainable AI (XAI) refers to techniques and tools that help us understand and interpret machine learning models. While many ML models (e.g., deep learning, ensemble methods) are often considered "black boxes," XAI methods provide insights into how models make predictions.
One of the most powerful XAI techniques is SHAP (SHapley Additive exPlanations), which is based on game theory and helps explain how each feature contributes to a model's prediction.


**SHAP (SHapley Additive exPlanations)**
SHAP is a framework used to interpret model predictions by calculating SHAP values, which measure the impact of each feature on an individual prediction.

**Why is SHAP important?**
1. Feature Importance: Helps identify which features contribute the most to predictions.
2. Fairness & Transparency: Explains why a model makes certain decisions, improving trust.
3. Debugging Models: Identifies biases, spurious correlations, and feature dependencies.


**SHAP Values**
What are SHAP Values?
SHAP values quantify how much each feature positively or negatively impacts a specific prediction. They are based on Shapley values from cooperative game theory, which fairly distribute "credit" among features for the final prediction.

**SHAP Value Intuition:**
Positive SHAP Value → The feature increases the model’s prediction.
Negative SHAP Value → The feature decreases the model’s prediction.
SHAP Value close to 0 → The feature has little or no impact on that prediction.
Mathematically, SHAP values represent the average marginal contribution of a feature across all possible feature combinations.

**SHAP Summary Plot**
The SHAP summary plot provides a global overview of feature importance and their effects on model predictions. It shows:

1. Feature Importance: Features are sorted by importance (top features have the most impact).
2. Effect on Prediction: Each dot represents a single data point, colored by feature value.
    i. Red (higher values): Positive impact on prediction.

    ii. Blue (lower values): Negative impact on prediction.
    
3. Spread of SHAP Values: Indicates the range of impact across different samples.
</details>

```python
import shap
shap.initjs()
import matplotlib.pyplot as plt
import tqdm as notebook_tqdm

explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_scaled)

np.shape(shap_values)
```

```python
# Summarize SHAP values for the first class (or all classes)
shap.summary_plot(shap_values[:,:, 0], snp_data, max_display=10)  # For binary classification, use shap_values[0] or shap_values[1]
```
This plot reveals the impact of the first 10 features (snps)on the model, it shows that patients with higher values of these snps are likely to be classified as cluster 0

```python
shap.summary_plot(shap_values[:,:, 1], snp_data, max_display=10)
```


