# Practical Session 1 – Hands-On Workshop (Machine Learning) - R Version

### 📥 **Dataset:** Download the file [here](https://drive.google.com/drive/folders/1o_KLnkfJ-3nSStfaCQHdAQHJa2L9v3kY?usp=drive_link)

### Dataset Overview
The following data is a synthetic data containing the the following columns:

**Columns for Anthropometric Traits:**
height (in meters), weight (in kg), BMI (calculated: weight / height²), waist_circumference (in cm), hip_circumference (in cm), and WHR (waist_circumference / hip_circumference)

**Metadata:** age (years) sex (e.g., Male/Female) cohort (e.g., Ugandan or Zulu)

**Genotype Data:** 1000 Simplified SNP columns with values representing genetic variants (e.g., 0, 1, 2).

**Cardiometabolic Traits:** systolic_BP (mmHg) diastolic_BP (mmHg) LDL_cholesterol, HDL_cholesterol (mg/dL)


******
‼️**Note:** Ensure that the dataset is in your current working directory or use its full path.

### SECTION 1: Load Required Libraries
```
# Here we load various libraries for data wrangling, visualizations, and
# statistical tasks. If needed, install with install.packages("<library>").

library(tidyverse)    # A collection of packages for data manipulation and visualization
library(ggplot2)      # A grammar of graphics for plotting
library(dplyr)        # Data manipulation (filter, select, mutate, etc.)
library(factoextra)   # PCA visualization and clustering helpers
library(reshape2)     # Helps reshape data (melting, casting)
library(dplyr)
```

### SECTION 2: Read in the Data
```
# Loads a CSV file, inspects the first rows, checks dimensions, and prints structure.
# Adjust the file path to match your environment.

data <- read_csv("anthropometric_trait_gwas.csv")

head(data)    # Examine first few rows
dim(data)     # View row-column counts
str(data)     # Investigate column types
```

### SECTION 3: Subset to Clinical Data
```
# Removes columns starting with "SNP_", leaving only clinical (phenotypic) variables.

clinical_data <- data %>%
  dplyr::select(-starts_with("SNP_"))

head(clinical_data)    # Check new subset
dim(clinical_data)     # Confirm dimension change
```

### SECTION 4: Basic Exploratory Plots
```
# Loops over all numeric columns and creates histograms with density overlays,
# providing an overview of distribution shapes for each clinical variable.

numeric_columns <- clinical_data %>%
  dplyr::select(where(is.numeric)) %>%
  names()

for (col_name in numeric_columns) {
  ggplot(clinical_data, aes(x = .data[[col_name]])) +
    geom_histogram(aes(y = ..density..), binwidth = 30, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", size = 1) +
    ggtitle(paste("Distribution of", col_name)) +
    theme_minimal() +
    labs(x = col_name, y = "Density") -> p
  
  print(p)
}
```

### SECTION 5: Outlier Removal (IQR)
```

# Defines a helper function to remove outliers using the 1.5 * IQR rule
# and applies it to numeric columns in the clinical dataset.

remove_outliers <- function(df, cols) {
  df_no_outliers <- df
  for (col in cols) {
    Q1 <- quantile(df_no_outliers[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(df_no_outliers[[col]], 0.75, na.rm = TRUE)
    IQR_value <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR_value
    upper_bound <- Q3 + 1.5 * IQR_value
    
    df_no_outliers <- df_no_outliers %>%
      filter(.data[[col]] >= lower_bound & .data[[col]] <= upper_bound)
  }
  return(df_no_outliers)
}

numeric_cols <- clinical_data %>%
  dplyr::select(where(is.numeric)) %>%
  names()

clinical_data <- remove_outliers(clinical_data, numeric_cols)
dim(clinical_data)  # Check how many rows remain after outlier removal
```

### SECTION 6: Recode Categorical Variables
```
# Converts text labels for 'sex' and 'cohort' into numeric codes for further analysis.

clinical_data <- clinical_data %>%
  mutate(
    sex = case_when(
      sex == "Female" ~ 0,
      sex == "Male"   ~ 1,
      TRUE ~ NA_real_
    )
  )

clinical_data <- clinical_data %>%
  mutate(
    cohort = case_when(
      cohort == "Ugandan" ~ 0,
      cohort == "Zulu"    ~ 1,
      TRUE ~ NA_real_
    )
  )

head(clinical_data)
```

### SECTION 7: Standardize the Data
```
# Extracts numeric columns, then applies standard scaling (mean=0, sd=1).
# This helps many ML methods treat all features equally.

num_cols <- clinical_data %>%
  dplyr::select(where(is.numeric)) %>%
  names()

scaled_data <- scale(clinical_data[, num_cols])
scaled_data_df <- as.data.frame(scaled_data)  # Optional convenience data frame
```

### SECTION 8: Perform PCA
```
# Uses Principal Component Analysis to reduce dimensionality and identify
# main axes of variation in the scaled clinical data.

pca_res <- prcomp(scaled_data, center = FALSE, scale. = FALSE)  # Data already scaled

var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
cum_var_explained <- cumsum(var_explained)

for (i in seq_along(cum_var_explained)) {
  cat("Principal Component", i, ":", round(cum_var_explained[i], 4), "\n")
}

num_components_90 <- which(cum_var_explained >= 0.90)[1]
cat("\nNumber of components for >= 90% variance:", num_components_90, "\n")

plot(cum_var_explained,
     type = "o", pch = 16, xlab = "Number of Principal Components",
     ylab = "Cumulative Variance Explained", main = "Cumulative Variance by PCA")
abline(h = 0.90, col = "red", lty = 2)
abline(v = num_components_90, col = "red", lty = 2)
dev.off()
```

### SECTION 9: Scatter Plots: PC1 vs PC2
```
# Extracts the first two principal components (PC1, PC2), then plots them to see
# grouping by sex or cohort.

pca_components <- pca_res$x
pc1 <- pca_components[, 1]
pc2 <- pca_components[, 2]

plot_df <- clinical_data %>%
  mutate(PC1 = pc1, PC2 = pc2)

plot_df$sex <- factor(plot_df$sex, levels = c(0,1), labels = c("Female", "Male"))
plot_df$cohort <- factor(plot_df$cohort, levels = c(0,1), labels = c("Ugandan", "Zulu"))

# Plot by Sex
ggplot(plot_df, aes(x = PC1, y = PC2, color = sex)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "PCA: PC1 vs PC2 (Colored by Sex)")
dev.off()

# Plot by Cohort
ggplot(plot_df, aes(x = PC1, y = PC2, color = cohort)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "PCA: PC1 vs PC2 (Colored by Cohort)")
dev.off()
```

### SECTION 10: PCA Loadings (Coefficients)
```
# Identifies how each variable contributes to each principal component (the loadings),
# and visualizes them as a heatmap for easy interpretation.

loadings <- pca_res$rotation
loading_df <- as.data.frame(loadings)
colnames(loading_df) <- colnames(pca_res$rotation)
rownames(loading_df) <- rownames(pca_res$rotation)

loading_df <- loading_df %>%
  rownames_to_column(var = "Variable")

loading_melt <- loading_df %>%
  pivot_longer(
    cols = -Variable,
    names_to = "PC",
    values_to = "Loading"
  )

head(loading_melt)

ggplot(loading_melt, aes(x = PC, y = Variable, fill = Loading)) +
  geom_tile() +
  geom_text(aes(label = round(Loading, 2)), color = "black", size = 3) +  # Add text labels
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() +
  labs(
    title = "PCA Loadings Heatmap",
    x = "Principal Components",
    y = "Variables"
  )
```

### SECTION 11: K-means Clustering
```
# K-means divides the data into a specified number of clusters (e.g., 4).
# Results are attached to the clinical data, then summarized by cluster.

set.seed(42)
kmeans_res <- kmeans(scaled_data, centers = 4, nstart = 25)
clinical_data$cluster <- factor(kmeans_res$cluster)
table(clinical_data$cluster)

cluster_profile <- clinical_data %>%
  group_by(cluster) %>%
  summarise(
    across(
      where(is.numeric),
      \(x) mean(x, na.rm = TRUE)
    )
  )

cluster_profile
```

### SECTION 12: Categorize Key Variables
```
# This section classifies continuous measures (e.g., BMI, BP, LDL, etc.)
# into labeled categories for simpler interpretation.

cluster_profile_categorical <- cluster_profile %>%
  mutate(
    BMI_category = case_when(
      BMI < 18.5 ~ "Underweight",
      BMI < 24.9 ~ "Healthy",
      BMI < 29.9 ~ "Overweight",
      TRUE       ~ "Obese"
    ),
    systolic_BP_category = case_when(
      systolic_BP < 120 ~ "Normal",
      systolic_BP < 130 ~ "Elevated",
      systolic_BP < 140 ~ "Hypertension Stage 1",
      TRUE              ~ "Hypertension Stage 2"
    ),
    diastolic_BP_category = case_when(
      diastolic_BP < 80 ~ "Normal",
      diastolic_BP < 90 ~ "Elevated",
      TRUE              ~ "Hypertension"
    ),
    LDL_category = case_when(
      LDL_cholesterol < 100 ~ "Optimal",
      LDL_cholesterol < 130 ~ "Near Optimal",
      LDL_cholesterol < 160 ~ "Borderline High",
      LDL_cholesterol < 190 ~ "High",
      TRUE                  ~ "Very High"
    ),
    HDL_category = case_when(
      HDL_cholesterol < 40  ~ "Low",
      HDL_cholesterol < 60  ~ "Normal",
      TRUE                  ~ "High"
    ),
    WHR_category = case_when(
      WHR < 0.85 ~ "Healthy",
      TRUE       ~ "High"
    ),
    sex = case_when(
      sex < 0.5 ~ "Female",
      TRUE      ~ "Male"
    ),
    cohort = case_when(
      cohort < 0.5 ~ "Ugandan",
      TRUE         ~ "Zulu"
    )
  )

cluster_profile_categorical %>%
  dplyr::select(cluster, age, sex, cohort, BMI_category, systolic_BP_category, 
         diastolic_BP_category, LDL_category, HDL_category, WHR_category)
```

### SECTION 13: Violin Plots by Cluster
```
# Displays distribution of selected numeric variables across k-means clusters
# using violin and embedded boxplots.

selected_columns <- c("systolic_BP", "diastolic_BP", "LDL_cholesterol",
                      "height", "weight", "waist_circumference", "hip_circumference")

clinical_data$cluster <- factor(clinical_data$cluster)

for (column in selected_columns) {
  ggplot(clinical_data, aes(x = cluster, y = .data[[column]], fill = cluster)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
    theme_minimal() +
    labs(title = paste("Violin Plot of", column, "by Cluster"), x = "Cluster", y = column) -> p
  
  print(p)
}
```

## PART 2
### SECTION 14: Load Required Libraries
```
# This section introduces more libraries for modeling and evaluation:
# - caret: Train/test splitting, models, RFE
# - glmnet: LASSO/Elastic Net
# - randomForest: Random Forest classification or regression
# - e1071: SVM support relied upon by caret
# - pROC: ROC curve analysis
# - PRROC: Precision-recall curve analysis

library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(pROC)
library(PRROC)
```
### SECTION 15: Extract and Align SNP Data
```
# This part isolates columns prefixed with "SNP_". It also aligns row indices
# with the clinical data, then sets "y" as the target variable (clusters).

snp_data <- data %>%
  dplyr::select(starts_with("SNP_"))
snp_data <- snp_data[rownames(clinical_data), ]

y <- clinical_data$cluster
y <- factor(y)

preproc <- preProcess(snp_data, method = c("center", "scale"))
X_scaled <- predict(preproc, snp_data)
```

### SECTION 16: Train/Test Split (80/20)
```
# Uses createDataPartition for a stratified split. The resulting
# X_train, X_test, y_train, y_test are used for modeling.

set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[train_index, ]
X_test  <- X_scaled[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]
```

### SECTION 17: Feature Selection
```
# Shows three approaches: Lasso, SVM-based RFE, Elastic Net. Useful for
# dimensionality reduction or identifying critical SNP features in multi-class.

###########################
### 17.1 Lasso (alpha=1) ###
###########################
x_mat_train <- as.matrix(X_train)
lasso_fit <- glmnet(
  x = x_mat_train,
  y = y_train,
  family = "multinomial",
  alpha = 1,
  lambda = 0.01
)

coefs_list_lasso <- coef(lasso_fit)
lasso_nonzero_idx <- c()
for (class_idx in seq_along(coefs_list_lasso)) {
  coefs_current <- as.matrix(coefs_list_lasso[[class_idx]])
  nonzero <- which(abs(coefs_current[-1, ]) > 1e-10)
  lasso_nonzero_idx <- union(lasso_nonzero_idx, nonzero)
}
lasso_features_logical <- rep(FALSE, ncol(X_train))
lasso_features_logical[lasso_nonzero_idx] <- TRUE

#############################################
## 17.2 SVM-based RFE for Multi-Class
#############################################
library(caret)
library(kernlab)

library(caret)
library(kernlab)   # for svmLinear

svmFuncs <- caretFuncs

svmFuncs$fit <- function(x, y, first, last, ...) {
  cat("\n------------------------------------------------------------\n")
  cat("FIT FUNCTION: Subset size =", ncol(x), "\n")
  cat("Classes in y:\n")
  print(table(y))
  
  if (any(table(y) == 0)) {
    stop("ERROR in FIT: At least one class has zero samples in this fold!")
  }
  
  cat("Now training svmLinear...\n")
  mod <- train(x, y, method = "svmLinear", ...)
  cat("Finished training. Model class:", class(mod), "\n")
  mod
}

svmFuncs$rank <- function(object, x, y) {
  cat("\nRANK FUNCTION: Extracting variable importance.\n")
  cat("Class of 'object':", class(object), "\n")
  
  imp <- varImp(object, scale = FALSE)$importance
  cat("Variable Importance dimension:", dim(imp), "\n")
  cat("First 5 rows:\n")
  print(head(imp, 5))
  
  if (ncol(imp) > 1) {
    imp_values <- rowMeans(imp)
  } else if (ncol(imp) == 1) {
    imp_values <- imp[, 1]
  } else {
    stop("ERROR in RANK: varImp returned zero columns!")
  }
  
  df <- data.frame(
    var     = names(imp_values),
    Overall = imp_values,
    stringsAsFactors = FALSE
  )
  df <- df[order(df$Overall, decreasing = TRUE), ]
  
  cat("Returning a data frame with", nrow(df), "rows.\n")
  cat("Top 5:\n")
  print(head(df, 5))
  cat("------------------------------------------------------------\n")
  df
}

svmFuncs$pred <- function(object, x) {
  cat("\nPRED FUNCTION: generating predictions.\n")
  preds <- predict(object, newdata = x)
  cat("Predictions length:", length(preds), " Levels:", levels(preds), "\n")
  preds
}

svmFuncs$summary <- function(data, lev = NULL, model = NULL) {
  cat("\nSUMMARY FUNCTION: defaultSummary on classification.\n")
  cat("Data columns in 'data':", names(data), "\n")
  out <- defaultSummary(data, lev, model)
  print(out)
  out
}

ctrl <- rfeControl(
  functions      = svmFuncs,
  method         = "cv",
  number         = 5,
  verbose        = TRUE,
  allowParallel  = TRUE,
  returnResamp   = "all",
  saveDetails    = TRUE
)

# Convert your training data X_train to a data frame
X_train_df <- as.data.frame(X_train)
X_train_df[] <- lapply(X_train_df, as.numeric)
y_train <- factor(y_train)

cat("nrow(X_train_df) =", nrow(X_train_df), " vs length(y_train) =", length(y_train), "\n")
cat("Any NA in X_train_df? ", sum(is.na(X_train_df)), "\n")
print(table(y_train))

set.seed(123)
svmProfile <- rfe(
  x          = X_train_df,
  y          = y_train,
  sizes      = c(5, 10, 20, 50),
  rfeControl = ctrl
)

svmProfile
best_size <- svmProfile$bestSubset
cat("Best subset size:", best_size, "\n")

# The names of the best features
svm_rfe_features <- svmProfile$optVariables

# Convert to column indices
svm_rfe_idx <- match(svm_rfe_features, colnames(X_train_df))
svm_rfe_idx <- svm_rfe_idx[!is.na(svm_rfe_idx)]


X_train_svm_rfe <- X_train_df[, svm_rfe_idx, drop = FALSE]

# convert test set "X_test" to a data frame:
X_test_df <- as.data.frame(X_test)
X_test_df[] <- lapply(X_test_df, as.numeric)

X_test_svm_rfe <- X_test_df[, svm_rfe_idx, drop = FALSE]

######################################
### 17.3 Elastic Net (alpha=0.5)
######################################
elastic_net_fit <- glmnet(
  x = as.matrix(X_train_df),
  y = y_train,
  family = "multinomial",
  alpha = 0.5,
  lambda = 0.01
)

coefs_list_en <- coef(elastic_net_fit)
en_nonzero_idx <- c()
for (class_idx in seq_along(coefs_list_en)) {
  coefs_current <- as.matrix(coefs_list_en[[class_idx]])
  nonzero <- which(abs(coefs_current[-1, ]) > 1e-10)
  en_nonzero_idx <- union(en_nonzero_idx, nonzero)
}

en_features_logical <- rep(FALSE, ncol(X_train_df))
en_features_logical[en_nonzero_idx] <- TRUE
```

### SECTION 18: Random Forest Training & Evaluation
```
# Defines a helper function to train/evaluate a random forest on a given feature subset
# and compares Lasso, SVM-RFE, and Elastic Net results.

evaluate_rf <- function(feature_logic_or_idx, name) {
  
  if (is.logical(feature_logic_or_idx) && length(feature_logic_or_idx) == ncol(X_train)) {
    selected_cols <- which(feature_logic_or_idx)
  } else {
    selected_cols <- feature_logic_or_idx
  }
  
  X_train_sel <- X_train[, selected_cols, drop = FALSE]
  X_test_sel  <- X_test[, selected_cols, drop = FALSE]
  
  set.seed(42)
  rf_fit <- randomForest(x = X_train_sel, y = y_train)
  y_pred <- predict(rf_fit, newdata = X_test_sel)
  acc <- mean(y_pred == y_test)
  cat(name, "Feature Selection - Random Forest Accuracy:", round(acc, 4), "\n")
  acc
}

# Evaluate subsets
lasso_acc <- evaluate_rf(lasso_features_logical, "Lasso")
svm_rfe_acc <- evaluate_rf(svm_rfe_idx, "SVM-RFE")
en_acc <- evaluate_rf(en_features_logical, "Elastic Net")

comparison_df <- data.frame(
  Method = c("Lasso", "SVM-RFE", "Elastic Net"),
  Accuracy = c(lasso_acc, svm_rfe_acc, en_acc)
)

ggplot(comparison_df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_col(width = 0.6) +
  ylim(0, 1) +
  labs(title = "Comparison of Feature Selection Methods", y = "Random Forest Accuracy") +
  theme_minimal()
```

### SECTION 19: Final Model using SVM-RFE Features
```
# Constructs a final random forest using the subset chosen by SVM-RFE,
# then performs an additional caret-based parameter search.

X_train_svm_rfe_df <- as.data.frame(X_train_svm_rfe)
X_test_svm_rfe_df  <- as.data.frame(X_test_svm_rfe)

row.names(X_train_svm_rfe_df) <- NULL
row.names(X_test_svm_rfe_df)  <- NULL

set.seed(42)
final_rf <- randomForest(x = X_train_svm_rfe_df, y = y_train)

y_pred <- predict(final_rf, X_test_svm_rfe_df)
final_accuracy <- mean(y_pred == y_test)
cat("\nFinal Model Accuracy:", round(final_accuracy, 4), "\n")


  rf_grid <- expand.grid(
    mtry = c(2, 4, 8)  # Example: typical param to tune in Random Forest
  )
  
  fitControl <- trainControl(method = "cv", number = 5)
grid_search <- train(
  x = X_train_svm_rfe_df,
  y = y_train,
  method = "rf",
  tuneGrid = rf_grid,
  trControl = fitControl,
  metric = "Accuracy"
)

cm <- confusionMatrix(data = y_pred, reference = y_test)
cat("\nConfusion Matrix:\n")
print(cm$table)
cat("\nClassification Statistics:\n")
print(cm$overall)
print(cm$byClass)
```

### SECTION 20: Hyperparameter Tuning (Grid Search) with caret
```
# Mimics Python's GridSearchCV logic for random forest, searching over 'mtry'.

rf_grid <- expand.grid(mtry = c(2, 4, 8))
fitControl <- trainControl(method = "cv", number = 5)

set.seed(42)
grid_search <- train(
  x = X_train_svm_rfe,
  y = y_train,
  method = "rf",
  tuneGrid = rf_grid,
  trControl = fitControl,
  metric = "Accuracy"
)

cat("Best Hyperparameters:\n")
print(grid_search$bestTune)

best_rf <- grid_search$finalModel
y_pred_grid <- predict(grid_search, X_test_svm_rfe)
final_acc_grid <- mean(y_pred_grid == y_test)
cat("\nOptimized Model Accuracy:", round(final_acc_grid, 4), "\n")

cm_optimized <- confusionMatrix(y_pred_grid, y_test)
cat("\nConfusion Matrix:\n")
print(cm_optimized$table)
cat("\nClassification Stats:\n")
print(cm_optimized$overall)
print(cm_optimized$byClass)
```

### SECTION 21: ROC and Precision-Recall (Multi-Class)
```
# Handles multi-class scenario by computing one-vs-rest ROC and PR curves
# for each label. If data were binary, a direct approach would suffice.

y_prob <- predict(grid_search, X_test_svm_rfe, type = "prob")

if (nlevels(y_test) > 2) {
  all_levels <- levels(y_test)
  roc_aucs <- c()
  pr_aucs  <- c()
  
  for (this_class in all_levels) {
    actual_binary <- ifelse(y_test == this_class, 1, 0)
    predicted_prob <- y_prob[[this_class]]
    
    roc_obj <- roc(actual_binary, predicted_prob)
    auc_val <- auc(roc_obj)
    
    pr_obj <- pr.curve(scores.class0 = predicted_prob[actual_binary == 1],
                       scores.class1 = predicted_prob[actual_binary == 0],
                       curve = TRUE)
    
    roc_aucs <- c(roc_aucs, auc_val)
    pr_aucs  <- c(pr_aucs, pr_obj$auc.integral)
  }
  
  cat("\nPer-Class ROC AUC:\n")
  print(data.frame(Class=all_levels, AUC=roc_aucs))
  cat("\nPer-Class PR AUC:\n")
  print(data.frame(Class=all_levels, AUC=pr_aucs))
} else {
  roc_obj <- roc(y_test, y_prob[[2]])
  cat("\nROC AUC:", auc(roc_obj), "\n")
  
  pr_obj <- pr.curve(scores.class0 = y_prob[[2]][y_test == levels(y_test)[2]],
                     scores.class1 = y_prob[[2]][y_test == levels(y_test)[1]],
                     curve = TRUE)
  cat("PR AUC:", pr_obj$auc.integral, "\n")
}
```

### SECTION 22: Load Required Libraries
```
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)       # for SVM (caret "svmLinear" etc.)
library(pROC)        # for Generating ROC curves
library(PRROC)       # for Generating Precision-Recall curves
library(fastshap)    # for Approximating SHAP values for model interpretation
```

### SECTION 23: Example: Data Setup
```
# In practice, one might have a dataset 'data' with SNP_ columns and clinical variables.
# 'snp_data' would be numeric SNP features, 'clinical_data' would be phenotypic data (e.g., BMI).
# The lines below show an example where 'risk' or 'BMI_group' is derived.

# Suppose 'data' is your full dataset that includes SNP_ columns and clinical variables.
#   - snp_data: the SNP genotype features (numeric)
#   - clinical_data: the clinical phenotypes (including BMI, systolic_BP, etc.)

# Example structure (you would replace with your actual data load):
# data <- read_csv("anthropometric_trait_gwas.csv")
# snp_data <- data %>% select(starts_with("SNP_"))
# clinical_data <- data %>% select(-starts_with("SNP_"))

# 23.1 Create a "Risk" variable (binary: High vs. Low)
#    High if BMI > 30 or systolic_BP > 130; else Low
clinical_data <- clinical_data %>%
  mutate(risk = if_else(BMI > 30 | systolic_BP > 130, "High", "Low"))

# 23.2 Alternatively, create a BMI_group variable (multi-class)
#    Example WHO-like bins
bins <- c(0, 18.5, 24.9, 29.9, 40, Inf)
labels <- c("Underweight", "healthy range", "Overweight", "Obesity", "severe obesity")
clinical_data <- clinical_data %>%
  mutate(BMI_group = cut(BMI, breaks = bins, labels = labels, right = FALSE))
```

### SECTION 24: Simple Distribution Plot
```
# This section counts how many individuals fall into each category of interest,
# then creates simple bar plots to show the distribution.

# For the binary "risk" variable:
clinical_data %>%
  count(risk) %>%
  ggplot(aes(x = risk, y = n, fill = risk)) +
  geom_col(width = 0.5) +
  theme_minimal() +
  labs(title = "Patient Stratification by Risk",
       x = "Risk groups",
       y = "Count") +
  scale_fill_brewer(palette = "Pastel1")

# Similarly for the multi-class "BMI_group":
clinical_data %>%
  count(BMI_group) %>%
  ggplot(aes(x = BMI_group, y = n, fill = BMI_group)) +
  geom_col() +
  theme_minimal() +
  labs(title = "Distribution of Patients by BMI Group",
       x = "BMI Group",
       y = "Count") +
  scale_fill_brewer(palette = "Pastel1")

```

### SECTION 25: Choose Your Target (Binary or Multi-Class)
```
# Decide which target variable to predict:
# If using a binary 'risk', or a multi-class 'BMI_group'.

# Binary risk approach:
# y <- clinical_data$risk  # Factor with c("Low", "High")

# OR multi-class BMI group approach:
y <- clinical_data$BMI_group  # Factor with c("Underweight", "healthy range", ...)

# 25.1 Ensure your SNP data is numeric
# snp_data <- snp_data %>% mutate_all(as.numeric)

# 25.2 Standardize the SNP features
preproc <- caret::preProcess(snp_data, method = c("center", "scale"))
X_scaled <- predict(preproc, snp_data)
```

### SECTION 26: Split Train/Test
```
# We partition the data for modeling, typically 80% train and 20% test.
# Stratification can help preserve class proportions in classification tasks.

set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[train_index, ]
X_test  <- X_scaled[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]
```

### SECTION 27: SVM-based Recursive Feature Elimination (RFE)
```
# In this approach, an SVM (linear kernel) is used to rank features iteratively,
# removing the least important at each step until reaching the desired subset size.

library(caret)
library(kernlab)   # needed for "svmLinear" method if not already loaded

####################################################
## 27.1) STRATIFIED SPLIT + DROPLEVELS
####################################################
set.seed(42)

# 'y' is the factor outcome with possible levels like "severe obesity"
# 'X_scaled' is the numeric SNP feature matrix
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
# If "severe obesity" had zero total samples, caretaker might ignore that level

X_train <- X_scaled[train_idx, , drop = FALSE]
X_test  <- X_scaled[-train_idx, , drop = FALSE]

y_train <- y[train_idx]
y_test  <- y[-train_idx]

cat("Training Set Distribution (before droplevels):\n")
print(table(y_train))

# Removes any factor levels not present in the training set
y_train <- droplevels(y_train)

cat("\nTraining Set Distribution (after droplevels):\n")
print(table(y_train))

####################################################
## 27.2) ENSURE X_TRAIN IS A PLAIN NUMERIC DATA FRAME
####################################################
X_train <- as.data.frame(X_train)
X_test  <- as.data.frame(X_test)

stopifnot(all(sapply(X_train, is.numeric)))  # verify numeric columns

####################################################
## 27.3) DEFINE A CUSTOM RFE FUNCTION SET FOR MULTI-CLASS SVM
####################################################
# caret's default rank function can fail for multi-class SVM, because varImp
# returns multiple columns. The custom set here combines columns via rowMeans.

svmFuncs <- caretFuncs

svmFuncs$fit <- function(x, y, first, last, ...) {
  train(x, y, method = "svmLinear", ...)
}

svmFuncs$rank <- function(object, x, y) {
  imp_df <- varImp(object, scale = FALSE)$importance
  if (ncol(imp_df) > 1) {
    combined <- rowMeans(imp_df)
  } else {
    combined <- imp_df[, 1]
  }
  out <- data.frame(
    var     = rownames(imp_df),
    Overall = combined,
    stringsAsFactors = FALSE
  )
  out <- out[order(out$Overall, decreasing = TRUE), ]
  out
}

svmFuncs$summary <- defaultSummary


####################################################
## 27.4) DEFINE RFE CONTROL
####################################################
ctrl <- rfeControl(
  functions = svmFuncs,
  method    = "cv",
  number    = 5,
  verbose   = FALSE
)

####################################################
## 27.5) PERFORM RFE
####################################################
set.seed(42)
svmProfile <- rfe(
  x          = X_train,
  y          = y_train,
  sizes      = c(5, 10, 20, 50),
  rfeControl = ctrl
)

####################################################
## 27.6) EXTRACT FEATURES AND SUBSET
####################################################
svm_rfe_features <- svmProfile$optVariables
svm_rfe_idx <- match(svm_rfe_features, colnames(X_train))

X_train_svm_rfe_selected <- X_train[, svm_rfe_idx, drop = FALSE]
X_test_svm_rfe_selected  <- X_test[, svm_rfe_idx, drop = FALSE]

cat("\nChosen Features:\n")
print(svm_rfe_features)
```

### SECTION 28: Random Forest Hyperparameter Tuning
```
# This uses a function to search over a grid of
# Random Forest hyperparameters, specified within 'mtry'.


param_grid <- expand.grid(
  mtry          = c(2, 5, 10),
  splitrule     = "gini",
  min.node.size = c(1, 2, 4)
)

ctrl_rf <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

set.seed(42)
rf_gridsearch <- train(
  x = X_train_svm_rfe_selected,
  y = y_train,
  method    = "rf",
  trControl = ctrl_rf,
  tuneGrid  = data.frame(mtry = c(2, 5, 10)),
  metric    = "Accuracy"
)

best_clf <- rf_gridsearch$finalModel
y_pred <- predict(rf_gridsearch, X_test_svm_rfe_selected)
```

### SECTION 29: Evaluate the Model
```
# This section checks final accuracy and confusion matrix metrics.

y_test <- droplevels(y_test)
y_pred <- factor(y_pred, levels = levels(y_test))

accuracy <- mean(y_pred == y_test)
cat("Optimized Model Accuracy:", round(accuracy, 4), "\n")

cm <- confusionMatrix(y_pred, y_test)
cat("\nConfusion Matrix:\n")
print(cm$table)

cat("\nClassification Stats:\n")
print(cm$overall)
print(cm$byClass)

```

### SECTION 30: ROC & Precision-Recall (Binary or Multi-Class)
```
# A one-vs-rest loop is shown for multi-class, or direct approach for binary classification.

library(pROC)
library(PRROC)
library(ggplot2)

y_prob <- predict(rf_gridsearch, newdata = X_test_svm_rfe_selected, type = "prob")

cat("Rows in y_prob:", nrow(y_prob), "vs. length(y_test):", length(y_test), "\n")
stopifnot(nrow(y_prob) == length(y_test))

colnames(y_prob) <- c("Underweight", "healthy range", "Overweight", "Obesity")

colnames(y_prob)
levels(y_test)

all_levels <- levels(y_test)

roc_data <- data.frame(FPR = numeric(), TPR = numeric(), Class = character(), stringsAsFactors = FALSE)
pr_data  <- data.frame(Recall = numeric(), Precision = numeric(), Class = character(), stringsAsFactors = FALSE)

for (class_label in all_levels) {
  cat("\nProcessing class:", class_label, "\n")
  
  actual_binary <- ifelse(y_test == class_label, 1, 0)
  
  if (all(actual_binary == 0) || all(actual_binary == 1)) {
    cat("  No valid test distribution for class:", class_label, "\n")
    next
  }
  
  predicted_prob <- y_prob[[class_label]]
  if (is.null(predicted_prob)) {
    cat("  No column found for class:", class_label, ", skipping!\n")
    next
  }
  
  stopifnot(length(predicted_prob) == length(actual_binary))
  
  roc_obj <- roc(actual_binary, predicted_prob)
  auc_val <- auc(roc_obj)
  
  coords_df <- coords(roc_obj, x = "all", ret = c("specificity", "sensitivity"), transpose=FALSE)
  roc_df <- data.frame(
    FPR   = 1 - coords_df$specificity,
    TPR   = coords_df$sensitivity,
    Class = class_label
  )
  roc_data <- rbind(roc_data, roc_df)
  
  pr_obj <- pr.curve(
    scores.class0 = predicted_prob[actual_binary == 1],
    scores.class1 = predicted_prob[actual_binary == 0],
    curve = TRUE
  )
  pr_df <- data.frame(
    Recall    = pr_obj$curve[,1],
    Precision = pr_obj$curve[,2],
    Class     = class_label
  )
  pr_data <- rbind(pr_data, pr_df)
  
  cat(sprintf("  ROC AUC = %.3f, PR AUC = %.3f\n", auc_val, pr_obj$auc.integral))
}

if (nrow(roc_data) == 0) {
  cat("No valid ROC data to plot.\n")
} else {
  ggplot(roc_data, aes(x = FPR, y = TPR, color = Class)) +
    geom_line(size = 1) +
    coord_equal() +
    geom_abline(linetype = "dashed", color = "gray50") +
    labs(
      title = "One-vs-Rest ROC Curves (Multi-Class)",
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    theme_minimal()
}

if (nrow(pr_data) == 0) {
  cat("No valid PR data to plot.\n")
} else {
  ggplot(pr_data, aes(x = Recall, y = Precision, color = Class)) +
    geom_line(size = 1) +
    labs(
      title = "One-vs-Rest Precision-Recall Curves (Multi-Class)",
      x = "Recall",
      y = "Precision"
    ) +
    theme_minimal()
}
```

### SECTION 31: SHAP-like Interpretation using fastshap
```
# This approximates SHAP values for interpretability. One defines a 'pred_func'
# returning numeric probabilities, then calls fastshap::explain().

library(fastshap)

if (nlevels(y_train) == 2) {
  pred_func <- function(object, newdata) {
    predict(object, newdata, type = "prob")[, "High"]
  }
} else {
  target_class <- "Obesity"
  pred_func <- function(object, newdata) {
    predict(object, newdata, type = "prob")[, target_class]
  }
}

rf_model <- best_clf
train_df <- as.data.frame(X_train_svm_rfe_selected)

set.seed(999)
shap_values <- fastshap::explain(
  object        = rf_model,
  X             = train_df,
  pred_wrapper  = pred_func,
  nsim          = 50,
  feature_names = colnames(train_df)
)

head(shap_values)

shap_values_df <- as.data.frame(shap_values)

shap_importance <- shap_values_df %>%
  summarise(across(everything(), ~ mean(abs(.x)))) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "MeanAbsSHAP") %>%
  arrange(desc(MeanAbsSHAP))

head(shap_importance, 10)

ggplot(shap_importance[1:10, ], aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Mean |SHAP| Features", x = "Feature", y = "Mean(|SHAP Value|)") +
  theme_minimal()
```

### END OF PIPELINE






