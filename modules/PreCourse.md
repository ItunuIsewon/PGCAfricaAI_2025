# Pre-course Analysis

## Table of Contents

  1. [Learning Outcomes](#Learning_Outcomes)
  2. [Setup](#Setup)
  3. [Exploratory Data Analysis](#Exploratory_Data_Analysis)


## Learning Outcomes
After completing this practical, you should be able to do the following:
  1. Set up Jupyter Notebook, install essential libraries, and test installations.
  2. Learn how to load datasets and explore it.
  3. Identify and address missing values, duplicates, and inconsistencies in the data.
  4. Use statistical summaries and visualizations (e.g., histograms, scatter plots, heatmaps).
  5. Learn about feature analysis.


## Setup 
This guide provides step-by-step instructions for installing Jupyter Notebook and the required libraries on both Windows and Ubuntu systems.


### Part 1: :hammer_and_wrench:  Jupyter Notebook
#### Windows
  1. **Install Python**
     + Download the latest version of Python from the [official website](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).
     + During installation, check the box "Add Python to PATH" and click "Install Now."
       
  2. **Install Jupyter Notebook**
     + Open the Command Prompt (Open the Run menu with Windows Key + R, then type "cmd." Press Ctrl + Shift + Enter to open as an Administrator)
     + Run the following commands:
       ```python
       pip install notebook
       ```
  3. **Verify the installation by running:**
       ```python
       jupyter notebook
       ```
  4. **Install Pip (if not already installed)**
     + If pip is not available, run:
       ```python
       python -m ensurepip --upgrade
       ```
       
#### Ubuntu
  1. **Update System Packages**
     + Open a terminal and run:
       ```sh
       sudo apt update 
       ```
       ```sh
       sudo apt upgrade 
       ```
    
  2. **Install Python and Pip**
     + Install Python 3 and pip using:
       ```sh
       sudo apt install python3 python3-pip -y 
       ```

  3. **Install Jupyter Notebook**
     + Use apt to install Jupyter Notebook: 
       ```sh
       sudo apt install jupyter-notebook 
       ```
       **_Note the hyphen between “jupyter” and “Notebook”_**
       
     + Verify the installation:
       ```sh
       jupyter notebook
       ```

  
### Part 2: :package: Installing Required Libraries
The libraries required for the workshop include:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - TensorFlow
  - PyTorch
  - biopython (optional for advanced genomics tasks)

#### Install Libraries on Windows
1. Open the Command Prompt.
2. Run the following command to install all libraries:
   ```python
     pip install pandas numpy matplotlib seaborn scikit-learn tensorflow  torch biopython
   ```

#### Install Libraries on Ubuntu
  1. Open a terminal.
  2. Run the following command to install all libraries:
     ```python
       pip3 install pandas numpy matplotlib seaborn scikit-learn tensorflow  torch biopython
     ```


### Part 3: :gear: Testing the Installation
  1. Open a terminal or Command Prompt.
  2. Start Jupyter Notebook by running:
     ```python
        jupyter notebook
     ```
  
  4. Create a new Python 3 notebook and run the following code to verify the libraries:
     ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import tensorflow as tf
        import torch
        import Bio

        print("All libraries are installed successfully!")
     ```
     
### Part 4: :hammer_and_wrench: Troubleshooting
**Common Issues and Fixing**
  - Command not found:
      + Ensure Python and pip are added to the system PATH.
      + Reinstall Python and select the option to "Add Python to PATH."
        
  - Permission errors (Ubuntu):
    + Use sudo with pip commands.
      
  - Jupyter Notebook does not open:
    + Clear browser cache or use another browser.
    + Check if the jupyter command is in your PATH by running:
      
    ```python
        which jupyter
    ```

******
:white_check_mark: You are now ready to use Jupyter Notebook with the required libraries for the workshop!


## Exploratory Data Analysis
**Dataset: The data for this practical is found in the *data/data.csv* directory.**

Import the following library

```python
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
1. pandas is used for data manipulation and analysis, providing DataFrame structures.
2. numpy is used for numerical operations, especially arrays and mathematical functions.
3. matplotlib and seaborn are used for data visualization. seaborn builds on matplotlib, offering easier and aesthetically pleasing plots.

```python
# Importing the dataset
data = pd.read_csv(r"C:\Users\Faithgokz\Desktop\PGC_Africa\data.csv")
```
The dataset is read using pandas' read_csv function, which loads CSV files into a data frame.

```python
# Printing the 1st 5 columns
data.head()
```
The first few rows are displayed to get a preview of the data.
```python
# Display dataset information to understand data types and missing values
data.info()
```
The info() method provides a concise summary of the data frame, including the number of non-null entries, data types, and memory usage. This is essential for identifying columns with missing values or inappropriate data types.

```python
# get the columns list:
data.columns
```
```python
# Target Variable:
data.diagnosis.value_counts()
```
```python
# Check for null values:
data.isnull().sum()
```
```python
# Visualizing missing data using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()
```
A heatmap is used to visualize the distribution of missing values in the dataset. The viridis colormap provides a clear representation, with non-missing values shown as one colour (purple) and missing values as another (yellow).

```python
# Visualizing the distribution of target variable (assuming 'diagnosis' is the target column)
sns.countplot(x='diagnosis', data=data, hue='diagnosis', palette= "pastel")
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()
```
A countplot is used to visualize the distribution of categorical variables. Here, the target variable 'diagnosis' is visualized to show the counts of each class (e.g., benign or malignant). This helps in understanding the class balance in the dataset.

```python
# Save diagnosis into a seperate variable
diagnosis = data['diagnosis']
features = data.drop(['diagnosis'], axis = 1)
```
```
features.head()
```
```
#drop the unnamed,missing, and Id columns:
features.drop(['Unnamed: 32', 'id','missing'], axis=1, inplace= True)
```
```
# Correlation matrix to identify relationships between numerical features
correlation_matrix = features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
The correlation matrix calculates pairwise correlations between numerical features. The heatmap visually represents these correlations, helping identify strongly correlated features, which can be useful for feature selection or elimination.

```
# Pairplot to visualize relationships between selected features
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
sns.pairplot(data[selected_features + ['diagnosis']], hue='diagnosis', palette='husl')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()
```
- A pairplot shows scatterplots and histograms of pairwise feature combinations, grouped by the target variable ('diagnosis'). This helps visualize feature separability and relationships.

- Feature separability refers to the extent to which different categories or classes of a target variable can be distinguished based on the values of one or more features.

**In the context of machine learning and data analysis:**

- Good separability means that the feature values for different classes are distinct or have minimal overlap. This makes it easier for a model to classify or predict the target variable accurately. Poor separability occurs when feature values for different classes overlap significantly, making it harder to distinguish between them. For example:

- In a scatterplot, if points belonging to two classes (e.g., "Benign" and "Malignant") form separate clusters, the features used in the plot demonstrate good separability.

- Good feature separability is critical for building accurate classification models, as it indicates that the features are informative and useful for distinguishing between classes.

```
# Boxplot to compare distributions of a feature across diagnosis categories
plt.figure(figsize=(8, 6))
sns.boxplot(x='diagnosis', y='radius_mean', data=data, hue = "diagnosis", palette='Set3')
plt.title('Comparison of Mean Radius Across Diagnosis Categories')
plt.xlabel('Diagnosis')
plt.ylabel('Mean Radius')
plt.show()
```
A boxplot is used to compare the distribution of a numerical feature (mean_radius) across categories of the target variable ('diagnosis'). This highlights potential differences in feature values based on diagnosis categories.

```
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_worst', 'texture_worst', 'perimeter_worst'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data,palette='Set3')
plt.show()
```
It is possible to plot box plots for multiple features at a time.

```
# Histogram of a numerical feature to understand its distribution
plt.figure(figsize=(8, 6))
data['area_mean'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Area Mean')
plt.xlabel('Mean Area')
plt.ylabel('Area Mean')
plt.show()
```
A histogram visualizes the distribution of a numerical feature (mean_area), providing insights into its spread, central tendency, and skewness.

- Spread: Refers to how widely the values of the feature are distributed. A wider spread indicates a more diverse range of values, while a narrow spread shows that the values are closer together.

- Central Tendency: Refers to the "center" of the data distribution, often measured by metrics like mean, median, or mode. The peak of the histogram gives an idea of the most frequent values, which often aligns with the central tendency.

- Skewness: Describes the asymmetry of the distribution. If the tail on the right side (positive values) is longer, the data is positively skewed. If the tail on the left side (negative values) is longer, the data is negatively skewed. A symmetric histogram indicates little or no skewness.

Why These Insights Matter Spread helps identify the range of feature values and whether the data is diverse or concentrated. Central Tendency provides a reference for the "average" value, useful for comparisons. Skewness can highlight potential biases in the data or suggest the need for transformations (e.g., logarithmic) to normalize the data for analysis or modeling.

```
#plot the histograms for each feature:
features.hist(figsize = (30,30), color = 'skyblue')
plt.show()
```
An histogram can also be plotted for all features at once.

```
# Scatterplot to examine the relationship between two numerical features
plt.figure(figsize=(8, 6))
sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=data, palette='coolwarm')
plt.title('Scatterplot of Mean Radius vs Mean Texture')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.show()
```
A scatterplot is used to examine the relationship between two numerical features. By adding hue for the diagnosis, the plot shows how the relationship varies across different classes.

The amount of overalap reveals how well these two features would be able to distinguish one class from the other. The lesser the overlap the better.

```
# Swarm plot to visualize the spread and clustering of features across categories
plt.figure(figsize=(10, 6))
sns.swarmplot(x='diagnosis', y='area_mean', data=data, hue='diagnosis', palette='Set2')
plt.title('Swarm Plot of Mean Area by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Mean Area')
plt.show()
```
A swarm plot provides detailed insights into the distribution and clustering of individual data points across categories, showing variability within each group.

```
# Jointplot to explore the relationship between two features along with their distributions
sns.jointplot(x='radius_mean', y='perimeter_mean', data=data, hue='diagnosis', kind='scatter', palette='coolwarm')
plt.suptitle('Jointplot of Mean Radius and Mean Perimeter', y=1.02)
plt.show()
```
A jointplot combines scatterplots with marginal histograms or KDE plots, offering a comprehensive view of the relationship between two variables and their individual distributions.

```
# Facet Grid for multi-dimensional visualizations
g = sns.FacetGrid(data, col='diagnosis', hue='diagnosis', height=4, aspect=1.5)
g.map(plt.scatter, 'texture_mean', 'smoothness_mean', alpha=0.6)
g.add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Facet Grid of Mean Texture vs Mean Smoothness by Diagnosis')
plt.show()
```
A Facet Grid splits the data into subplots based on categorical variables, facilitating the exploration of multi-dimensional relationships across subsets of data.

```
# Violin plot to analyze feature distribution and variability
plt.figure(figsize=(8, 6))
sns.violinplot(x='diagnosis', y='texture_mean', data=data, hue ='diagnosis', palette='cool')
plt.title('Violin Plot of Mean Texture by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Mean Texture')
plt.show()
```
A violin plot shows the distribution and variability of a numerical feature, combining aspects of a boxplot and KDE. This helps in visualizing density across categories.
