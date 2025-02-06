# Practical Session 1 – Hands-On Workshop (Machine Learning)


## Dataset Overview
The following data is a synthetic data containing the the following columns:

**Columns for Anthropometric Traits:**
height (in meters), weight (in kg), BMI (calculated: weight / height²), waist_circumference (in cm), hip_circumference (in cm), and WHR (waist_circumference / hip_circumference)

**Metadata:** age (years) sex (e.g., Male/Female) cohort (e.g., Ugandan or Zulu)

**Genotype Data:** 1000 Simplified SNP columns with values representing genetic variants (e.g., 0, 1, 2).

**Cardiometabolic Traits:** systolic_BP (mmHg) diastolic_BP (mmHg) LDL_cholesterol, HDL_cholesterol (mg/dL)

******
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('synthetic_data_clusters_v18.csv')

```

```python
# Select only clinical data (excluding SNP columns)
clinical_data = data.loc[:, ~data.columns.str.startswith('SNP_')]

# Display the first few rows
print(clinical_data.head())
```

```python
len(clinical_data.columns)
```

<details>
  
  **Output:**
13
</details>

```python
# Step 1: Identify and remove outliers for all numeric columns
numeric_cols = clinical_data.select_dtypes(include=['float64', 'int64']).columns

# Function to calculate IQR and remove outliers
def remove_outliers(df, cols):
    df_no_outliers = df.copy()
    for col in cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
    return df_no_outliers

# Remove outliers from all numeric columns
```

```python
clinical_data = remove_outliers(clinical_data, numeric_cols)
```

```python
clinical_data
```

<details>

  **What Are Categorical Variables?**
  
Categorical variables represent discrete categories or labels, rather than numerical values. They can be classified into:
    
1. **Nominal Variables** – Categories with no inherent order (e.g., colours: "Red," "Blue," "Green").
2. **Ordinal Variables** – Categories with a meaningful order (e.g., education levels: "High School," "Bachelor’s," "Master’s," "PhD"). 

**Why Convert Categorical Variables to Numerical in Machine Learning?** Most machine learning algorithms require numerical input because they rely on mathematical computations like distance calculations, matrix operations, and statistical techniques. 
  
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

<details>

  **Why Do We Standardize Data?**
  
Standardization is a common preprocessing step in machine learning and data analysis. It involves scaling the data to have a specific mean and variance. Specifically, we aim to make the features have a mean of 0 and a standard deviation of 

**The primary reasons for standardizing data are:**

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

<details>

  **Principal Component Analysis (PCA)**
  
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

<details>
  
  **Cumulative Variance in PCA**
  
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

<details>
  
  **Output:**

Principal Component 1: 0.2614
Principal Component 2: 0.5016
Principal Component 3: 0.6112
Principal Component 4: 0.6920
Principal Component 5: 0.7710
Principal Component 6: 0.8368
Principal Component 7: 0.8952
Principal Component 8: 0.9440
Principal Component 9: 0.9902
Principal Component 10: 0.9971
Principal Component 11: 0.9994
Principal Component 12: 0.9999
Principal Component 13: 1.0000

</details>

```python
# Find the number of components where cumulative variance first exceeds 90%
num_components_90 = np.argmax(cumulative_variance >= 0.90) + 1  # Add 1 to match component index


# Plot cumulative variance explained
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
# Add vertical line at the number of components reaching 90%
plt.axvline(x=num_components_90, color='r', linestyle='--', label=f'{num_components_90} Components (90%)')

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


#clinical_data.loc[:, 'sex'] = clinical_data['sex'].map({ 0:'Female', 1:'Male'})

# Plot for Sex: Male vs Female
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pc1, pc2, 
                      c=clinical_data['sex'], #.apply(lambda x: 0 if x == 'Male' else 1)
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

```

```python
# Plot for Cohort: Ugandan vs Zulu
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pc1, pc2, 
                      c=clinical_data['cohort'], 
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

<details>
  
  **Loadings in PCA**
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

<details>
  
  From the PCA plot, we know that there are four clusters, however, we do not know what group each individual falls into. clustering algorithms can enable us to assign individuals to groups.

Clustering is a type of unsupervised learning technique used to group data points or objects based on their similarity. The goal of clustering is to identify inherent patterns or structures in the data without prior knowledge of true labels. 
Clustering algorithms partition the data into groups or clusters such that data points within the same cluster are more similar to each other than to those in other clusters.

**K-means Clustering:** K-means is a popular centroid-based clustering algorithm. It partitions the data into K clusters by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of data points assigned to each cluster. 
[sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

</details>

```python
from sklearn.cluster import KMeans
# Apply K-means clustering (for example, 3 clusters for risk stratification)
kmeans = KMeans(n_clusters=4, random_state=42)
clinical_data.loc[:, 'cluster'] = kmeans.fit_predict(scaled_data)
```

.value_counts() is a Pandas function used to count the occurrences of unique values in a Series (a single column of a DataFrame). It’s commonly used for analyzing categorical or discrete numerical data.

```python
clinical_data['cluster'].value_counts()
```

<details>
  
  **Why Do We Need Cluster Profiles?**
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
cluster_profile_categorical = cluster_profile


import pandas as pd

# Function to categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Healthy'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Function to categorize Systolic BP
def categorize_systolic_bp(sbp):
    if sbp < 120:
        return 'Normal'
    elif 120 <= sbp < 130:
        return 'Elevated'
    elif 130 <= sbp < 140:
        return 'Hypertension Stage 1'
    else:
        return 'Hypertension Stage 2'

# Function to categorize Diastolic BP
def categorize_diastolic_bp(dbp):
    if dbp < 80:
        return 'Normal'
    elif 80 <= dbp < 90:
        return 'Elevated'
    else:
        return 'Hypertension'

# Function to categorize LDL cholesterol
def categorize_ldl(ldl):
    if ldl < 100:
        return 'Optimal'
    elif 100 <= ldl < 130:
        return 'Near Optimal'
    elif 130 <= ldl < 160:
        return 'Borderline High'
    elif 160 <= ldl < 190:
        return 'High'
    else:
        return 'Very High'

# Function to categorize HDL cholesterol
def categorize_hdl(hdl):
    if hdl < 40:
        return 'Low'
    elif 40 <= hdl < 60:
        return 'Normal'
    else:
        return 'High'

# Function to categorize WHR
def categorize_whr(whr):
    if whr < 0.85:
        return 'Healthy'
    else:
        return 'High'


# Applying the categorization functions to the DataFrame
cluster_profile_categorical['BMI_category'] = cluster_profile_categorical['BMI'].apply(categorize_bmi)
cluster_profile_categorical['systolic_BP_category'] = cluster_profile_categorical['systolic_BP'].apply(categorize_systolic_bp)
cluster_profile_categorical['diastolic_BP_category'] = cluster_profile_categorical['diastolic_BP'].apply(categorize_diastolic_bp)
cluster_profile_categorical['LDL_category'] = cluster_profile_categorical['LDL_cholesterol'].apply(categorize_ldl)
cluster_profile_categorical['HDL_category'] = cluster_profile_categorical['HDL_cholesterol'].apply(categorize_hdl)
cluster_profile_categorical['WHR_category'] = cluster_profile_categorical['WHR'].apply(categorize_whr)
# Optional: Convert cohort to categorical (Zulu or Ugandan)
cluster_profile_categorical['cohort'] = cluster_profile_categorical['cohort'].map({0.0: 'Zulu', 1.0: 'Ugandan'})
# Example: If 'sex' is 0 for Male and 1 for Female, convert it to "Male" and "Female"
cluster_profile_categorical['sex'] = cluster_profile_categorical['sex'].map({0: 'Male', 1: 'Female'})
# Display the updated DataFrame with categories
cluster_profile_categorical[['age', 'sex', 'cohort', 'BMI_category', 'systolic_BP_category', 
          'diastolic_BP_category', 'LDL_category', 'HDL_category', 'WHR_category']]

```

```python
import seaborn as sns
import matplotlib.pyplot as plt

# List of all numeric columns in the dataframe except 'age' and 'LDL_cholesterol'
selected_columns = ['systolic_BP', 'diastolic_BP', 'LDL_cholesterol', 'height', 'weight', 
                   'waist_circumference', 'hip_circumference']

# Create a violin plot for each numeric column, grouped by the 'cluster' column
for column in selected_columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='cluster', y=column, data=clinical_data)
    plt.title(f'Violin Plot of {column} by Cluster')
    plt.show()
```

Can you identify unique patterns in the clusters based on these cluster profiles?

<details>
  
  The clusters represent distinct health profiles, each associated with varying degrees of cardiovascular and metabolic risk. Here's how we can assess the risk for each cluster:

**Cluster 0: Moderate Cardiovascular Risk, Balanced Cholesterol**
Risk of:
+ **Metabolic Syndrome:** Borderline elevated BMI and moderate LDL cholesterol levels could indicate a risk of developing metabolic syndrome (a group of conditions like high blood pressure, high cholesterol, and high blood sugar).
+ **Cardiovascular Disease:** While blood pressure is normal, the individual still has a higher LDL cholesterol level, which can contribute to atherosclerosis (plaque buildup in the arteries).
+ **Type 2 Diabetes:** The borderline overweight status (BMI 25.7) may increase the risk for insulin resistance or developing type 2 diabetes.

**Cluster 1: Higher Cholesterol and Blood Pressure, Possible Increased Risk**
Risk of:
+ **Hypertension:** The elevated systolic and diastolic blood pressure (128/84 mmHg) puts individuals in this cluster at a higher risk of hypertension, which is a significant risk factor for stroke and heart disease.
+ **Cardiovascular Disease:** Elevated blood pressure combined with high LDL cholesterol can significantly increase the risk of atherosclerosis, which could lead to coronary artery disease, heart attacks, and strokes.
+ **Obesity-Related Conditions:** The increased waist-to-hip ratio (WHR of 0.96) suggests abdominal obesity, which is strongly linked to a higher risk of type 2 diabetes, non-alcoholic fatty liver disease (NAFLD), and metabolic syndrome.
+ **Type 2 Diabetes:** Abdominal fat and higher cholesterol levels may indicate insulin resistance, which could lead to type 2 diabetes.

**Cluster 2: Similar to Cluster 1 with Slight Differences**
Risk of:
+ **Hypertension and Cardiovascular Disease:** Elevated systolic and diastolic blood pressure (130/85 mmHg) combined with high LDL cholesterol levels puts individuals at increased risk for heart disease, stroke, and other cardiovascular events.
+ **Obesity-Related Conditions:** Even with a slightly lower WHR (compared to Cluster 1), this cluster still has a BMI over 25 and higher abdominal fat, placing them at risk for metabolic syndrome, type 2 diabetes, and fatty liver disease.
+ **Type 2 Diabetes:** The presence of elevated LDL cholesterol, slightly elevated blood pressure, and higher BMI suggests a heightened risk for insulin resistance and type 2 diabetes.

  
**Cluster 3: Lower Cholesterol and Blood Pressure, More Healthy Profile**
Risk of:
+ **Metabolic Syndrome:** Despite having a relatively healthy cholesterol profile, the higher WHR (0.96) indicates abdominal obesity, which is associated with an increased risk of developing metabolic syndrome, including type 2 diabetes and cardiovascular diseases.
+ **Cardiovascular Disease:** While this cluster has relatively normal blood pressure and cholesterol, the abdominal obesity (high WHR) still suggests a moderate risk of heart disease.
+ **Type 2 Diabetes:** The higher WHR indicates a higher risk of insulin resistance and type 2 diabetes, despite the relatively normal BMI and lipid levels.

**Summary of Risks for Each Cluster:**
**Cluster 0:** Moderate cardiovascular and metabolic risk due to BMI and LDL cholesterol. Risk of metabolic syndrome and type 2 diabetes.
**Cluster 1:** High cardiovascular and metabolic risk due to high blood pressure, high LDL cholesterol, and abdominal obesity. Significant risk of heart disease, stroke, type 2 diabetes, and metabolic syndrome.
**Cluster 2:** High cardiovascular and metabolic risk, similar to Cluster 1 but with slightly different lipid and blood pressure profiles. Elevated risk for hypertension, heart disease, and diabetes.
**Cluster 3:** Moderate risk due to abdominal obesity (high WHR), despite normal cholesterol and blood pressure. Risk of cardiovascular disease, metabolic syndrome, and type 2 diabetes.

</details>


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
# Subset 'data' to have the same indices as 'clinical_data_NO'
snp_data = snp_data.loc[clinical_data.index]
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
from sklearn.linear_model import Lasso, ElasticNet
alpha = 0.01
```
```python
# Lasso Regression
lasso = Lasso(alpha= alpha,max_iter= 10000)
lasso.fit(X_train, y_train)
lasso_features = np.abs(lasso.coef_) > 0  # Select nonzero features
```


```python
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# Define SVM as the estimator for RFE
svm = SVC(kernel="linear", C=1)  # Linear kernel is important for feature selection

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=svm)  # Select top 5 features n_features_to_select=5
rfe.fit(X_train, y_train)
svm_rfe_features = np.where(rfe.support_)[0]
```

```python
# Elastic Net Regression
elastic_net = ElasticNet(alpha= alpha, l1_ratio=0.5,max_iter= 10000)  # 50% L1, 50% L2 penalty
elastic_net.fit(X_train, y_train)
elastic_net_features = np.abs(elastic_net.coef_) > 0  # Select nonzero features
```

```python
# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
```

```python
# Function to evaluate Random Forest with selected features
def evaluate_rf(selected_features, name):
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Feature Selection - Random Forest Accuracy: {accuracy:.4f}")
    return accuracy
```

```python
# Evaluate models
lasso_acc = evaluate_rf(lasso_features, "Lasso")
svm_rfe_acc = evaluate_rf(svm_rfe_features, "svm_rfe")
elastic_net_acc = evaluate_rf(elastic_net_features, "Elastic Net")
```

```python
# Compare results
plt.figure(figsize=(6, 4))
plt.bar(["Lasso", "svm-rfe", "Elastic Net"], [lasso_acc, svm_rfe_acc, elastic_net_acc], color=['blue', 'green', 'red'])
plt.ylabel("Random Forest Accuracy")
plt.title("Comparison of Feature Selection Methods")
plt.ylim(0.8, 0.9)
plt.show()
```

```python
X_train_svm_rfe_selected = X_train[:, svm_rfe_features]
X_test_svm_rfe_selected = X_test[:, svm_rfe_features]
```

```python
X_train_svm_rfe_selected.shape
```

```python
clf.fit(X_train_svm_rfe_selected, y_train)
```

```python
# Predict on test data
y_pred = clf.predict(X_test_svm_rfe_selected)
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
  
  1. **Model Accuracy:** 0.9072. This indicates that the model correctly classified 90.72% of all instances across the 4 classes. This is a strong result, but depending on the dataset and problem, you may want to focus more on precision, recall, and F1-scores (especially if the classes are imbalanced).
  2. **Classification Report:** The classification report gives a more detailed view of the model’s performance for each class.
  3. **Precision:** The percentage of correct positive predictions out of all positive predictions for that class.
Class 0 (Precision: 98%): The model is very accurate when it predicts Class 0, with 98% of those predictions being correct.
Class 1 (Precision: 100%): All predictions made for Class 1 are correct, indicating perfect precision for this class.
Class 2 (Precision: 78%): The model has room for improvement here, as it occasionally misclassifies other classes as Class 2.
Class 3 (Precision: 91%): Most predictions for Class 3 are correct, though there is still a small portion of incorrect predictions.
  4. **Recall:** The percentage of actual instances of each class that were correctly predicted.
Class 0 (Recall: 95%): The model correctly identifies 95% of all actual Class 0 instances.
Class 1 (Recall: 100%): The model never misses a Class 1 instance, showcasing perfect recall for this class.
Class 2 (Recall: 90%): The model does a fairly good job of catching most Class 2 instances, missing 10%.
Class 3 (Recall: 79%): While most Class 3 samples are identified correctly, there is a notable portion (21%) that the model fails to recognize as Class 3.

  5. **F1-Score**
The harmonic mean of precision and recall, is useful when dealing with imbalanced classes or when you want a single metric that balances both.

Class 0 (F1: 0.97): The high F1-score reflects the strong balance of precision and recall for Class 0.
Class 1 (F1: 1.00): Perfect precision and recall lead to a perfect F1-score for Class 1.
Class 2 (F1: 0.83): This moderate F1-score shows that while Class 2 performance is decent, it’s not as strong as Class 0 or Class 1.
Class 3 (F1: 0.85): The model performs reasonably well but can still improve in identifying Class 3 accurately and consistently.

**Confusion Matrix**
  The confusion matrix compares actual classes (rows) to predicted classes (columns). Each row tells you how many samples of a given actual class were predicted as each possible class. 

**Row 0 (Actual Class 0):**
42 instances correctly predicted as Class 0.
2 instances misclassified as Class 2.

**Row 1 (Actual Class 1):**
48 instances correctly predicted as Class 1.
0 misclassifications in other classes.

**Row 2 (Actual Class 2):**
45 instances correctly predicted as Class 2.
1 instance misclassified as Class 0 and 4 as Class 3.

**Row 3 (Actual Class 3):**
41 instances correctly predicted as Class 3.
11 instances misclassified as Class 2.
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


# snp_data = data.loc[:, data.columns.str.startswith('SNP_')]

# # Target variable: clusters from previous K-Means
# y = clinical_data['cluster']  # Use the clusters as labels

# # Standardize SNP features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(snp_data)

# # Split into training (80%) and test (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

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
grid_search.fit(X_train_svm_rfe_selected, y_train)

# Print the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train final model with best parameters
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_svm_rfe_selected)

# Evaluate the optimized model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Convert labels to integers
classes = np.unique(y_pred)
y_test = label_binarize(y_test, classes=classes)
y_pred = label_binarize(y_pred, classes=classes)
# ROC Curve and AUC
# Calculate ROC curve and AUC for each class (One-vs-Rest)
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(12, 6))
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate Precision-Recall and AUPR for each class
precision, recall, aupr = dict(), dict(), dict()
for i in range(len(classes)):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    aupr[i] = average_precision_score(y_test[:, i], y_pred[:, i])

# Plot Precision-Recall Curve
plt.figure(figsize=(12, 6))
for i in range(len(classes)):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {classes[i]} (AUPR = {aupr[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
```

```python
classes = np.unique(y_pred)
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
from tqdm import tqdm
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_scaled)

np.shape(shap_values)
```

```python
# Summarize SHAP values for the first class (or all classes)
shap.summary_plot(shap_values[:,:, 0], snp_data, max_display=10)  # For binary classification, use shap_values[0] or shap_values[1]

```

<details>
  
  **Patient risk stratification**
Patient risk stratification involves grouping patients based on their likelihood of developing a particular outcome, such as disease progression, complications, or treatment response. Using the synthetic data you've generated, you can apply a variety of methods to stratify patients into different risk categories. These methods can be based on both supervised and unsupervised learning approaches, depending on whether you have labelled data or are trying to discover inherent risk groups.
</details>

```python
# Create a synthetic binary outcome (1 = high risk, 0 = low risk)
# For simplicity, let's say BMI > 30 or systolic BP > 130 puts patients in the high-risk group
clinical_data['risk'] = np.where((clinical_data['BMI'] > 30) | (clinical_data['systolic_BP'] > 130), "High", "Low")

```

<details>
  In terms of health risks, both Body Mass Index (BMI) and systolic blood pressure (BP) are commonly used to assess the likelihood of developing cardiovascular diseases and other chronic conditions.

**BMI (Body Mass Index)**
BMI is a measure of body fat based on height and weight. It is used to categorize individuals into different weight groups, which can impact health risks. The categories are:

**Underweight:** BMI < 18.5
**Normal weight:** BMI 18.5 - 24.9
**Overweight:** BMI 25 - 29.9
**Obesity:** BMI ≥ 30


High risk is typically associated with:

**Overweight:** Increases the risk of conditions like type 2 diabetes, heart disease, and stroke.
**Obesity:** Carries a significantly higher risk for cardiovascular diseases, hypertension, type 2 diabetes, and certain cancers.


**Systolic Blood Pressure (BP)**
Systolic BP measures the pressure in your arteries when your heart beats. Normal BP is around 120/80 mmHg. The systolic value is the top number, representing the pressure in your arteries during a heartbeat.

**BP categories:**
Normal: < 120/80 mmHg
Elevated: 120-129 systolic and < 80 diastolic
Hypertension Stage 1: 130-139 systolic or 80-89 diastolic
Hypertension Stage 2: ≥ 140 systolic or ≥ 90 diastolic

**High risk is generally associated with:**
**Hypertension Stage 1 and 2:** Increases the risk of heart disease, stroke, kidney disease, and other complications.
High Risk (Combined)
The combination of a high BMI (overweight or obesity) and high systolic blood pressure (hypertension) significantly increases the risk of Cardiovascular diseases including heart attack, Type 2 diabetes, Kidney disease, Certain cancers, and Stroke, Together, they indicate a higher likelihood of developing serious health complications, requiring interventions like lifestyle changes or medical treatment.

</details>

```python
# Visualizing the distribution of target variable (assuming 'diagnosis' is the target column)
sns.countplot(x='risk', data=clinical_data, hue='risk', palette= "pastel", legend= True)
plt.title('Patient Stratification')
plt.xlabel('Risk groups')
plt.ylabel('Count')
plt.show()
```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


# snp_data = data.loc[:, data.columns.str.startswith('SNP_')]

# # Target variable: clusters from previous K-Means
# y = clinical_data['cluster']  # Use the clusters as labels
y = clinical_data['risk']

# # Standardize SNP features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(snp_data)


# # Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Define SVM as the estimator for RFE
svm = SVC(kernel="linear", C=1)  # Linear kernel is important for feature selection

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=svm)  # Select top 5 features n_features_to_select=5
rfe.fit(X_train, y_train)
svm_rfe_features = np.where(rfe.support_)[0]

X_train_svm_rfe_selected = X_train[:, svm_rfe_features]
X_test_svm_rfe_selected = X_test[:, svm_rfe_features]

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
grid_search.fit(X_train_svm_rfe_selected, y_train)

# Print the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train final model with best parameters
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_svm_rfe_selected)

# Evaluate the optimized model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# Convert 'y_true' and 'y_pred' from categorical to numeric (0 for 'Low' and 1 for 'High')
y_test = np.where(y_test == "High", 1, 0)
y_pred = np.where(y_pred == "High", 1, 0)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve and AUPR
precision, recall, _ = precision_recall_curve(y_test, y_pred)
aupr = average_precision_score(y_test, y_pred)

# Plot ROC Curve
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (AUPR = {aupr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()
```

```python
clinical_data["BMI"]
```

```python
import pandas as pd

# Define the BMI categories and bins based on WHO guidelines
bins = [0, 18.5, 24.9, 29.9,  40, float('inf')]  # Adding infinity to capture BMI > 40
labels = ['Underweight', 'healthy range', 'Overweight', 'Obesity', 'severe obesity']

# Create a new column for BMI groups
clinical_data['BMI_group'] = pd.cut(clinical_data['BMI'], bins=bins, labels=labels, right=False)

# Show the first few rows to check the grouping
print(clinical_data[['BMI', 'BMI_group']].head())

```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


# snp_data = data.loc[:, data.columns.str.startswith('SNP_')]

# # Target variable: clusters from previous K-Means
# y = clinical_data['cluster']  # Use the clusters as labels
y = clinical_data['BMI_group']

# # Standardize SNP features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(snp_data)


# # Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Define SVM as the estimator for RFE
svm = SVC(kernel="linear", C=1)  # Linear kernel is important for feature selection

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=svm)  # Select top 5 features n_features_to_select=5
rfe.fit(X_train, y_train)
svm_rfe_features = np.where(rfe.support_)[0]

X_train_svm_rfe_selected = X_train[:, svm_rfe_features]
X_test_svm_rfe_selected = X_test[:, svm_rfe_features]

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
grid_search.fit(X_train_svm_rfe_selected, y_train)

# Print the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train final model with best parameters
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_svm_rfe_selected)

# Evaluate the optimized model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Convert labels to integers
classes = np.unique(y_pred)
y_test = label_binarize(y_test, classes=classes)
y_pred = label_binarize(y_pred, classes=classes)
# ROC Curve and AUC
# Calculate ROC curve and AUC for each class (One-vs-Rest)
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(12, 6))
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate Precision-Recall and AUPR for each class
precision, recall, aupr = dict(), dict(), dict()
for i in range(len(classes)):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    aupr[i] = average_precision_score(y_test[:, i], y_pred[:, i])

# Plot Precision-Recall Curve
plt.figure(figsize=(12, 6))
for i in range(len(classes)):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {classes[i]} (AUPR = {aupr[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
```


