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
This guide provides step-by-step instructions to install Jupyter Notebook and the required libraries on both **Windows** and **Ubuntu** systems.

### Part 1: Installing Jupyter Notebook
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
  
### Part 2: Installing Required Libraries
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

### Part 3: Testing the Installation
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
     
### Part 4: Troubleshooting
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

You are now ready to use Jupyter Notebook with the required libraries for the workshop!






