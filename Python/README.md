
# PCA using NIPALS Algorithm (Python)

## Overview

This project implements Principal Component Analysis (PCA) using the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm, with a focus on flexibility and ease of use. It includes both a class-based and a module-based approach, allowing users to choose the most suitable implementation for their workflow. The project also introduces an optional visualization parameter that allows users to explore latent space distribution based on an additional variable, which is useful for identifying clusters or patterns in the data.

## Features

- **Automatic Component Selection**: By default, the number of components is determined based on the eigenvalue less than 1 rule. Users can also manually specify the number of components.
  
- **Center-Scaling**: The function centers and scales the data by default to ensure better model performance. Users can skip this step by adjusting the `to_be_scaled` parameter.

- **Alpha Parameter (Modeling Framework)**: An alpha parameter (ranging from 0 to 1) defines the confidence limit within which the model predictions are valid. A smaller alpha value provides a more constrained but accurate model, while a higher alpha broadens the model's scope at the potential cost of prediction accuracy.

- **PCA Score Visualization**: An optional third input parameter (`color_map`) allows users to visualize data distribution in latent space, providing insight into possible clustering or variable-based patterns.

- **Ease of Use**: Designed for simplicity, the functions are straightforward to implement, requiring minimal adjustments for customized analysis.

- **Missing Value Estimation**: A new feature has been added to estimate missing values in new observations based on the trends extracted using PCA.

## Usage

### Function Inputs:

- `X`: The input dataset (training data) for which the PCA model is developed.
- `num_components` (optional): The number of components to extract. By default, it uses the number of X variables.
- `color_map` (optional): A set of data for visualization purposes, helping identify potential clusters in the latent space.
- `alpha` (optional): A parameter ranging from 0 to 1 that defines the modeling framework within which the model is valid. Default is 1.

### Function Outputs:

The function returns a structured output containing:
- **Scores and Loadings**: Essential components of the PCA model.
- **Eigenvalues**: Variance explained by each principal component.
- **Centering and Scaling values**: Retained for future scaling of new data.
- **Hotelling's T²**: A multivariate measure to identify outliers in the dataset.
- **SPE (Squared Prediction Error)**: Measures the deviation between observed and predicted values.
- **SPE and T² Limits**: Set thresholds for identifying unusual data points.

### Example Usage:

#### Model Implementation as a Module:

```python
import pca_module as pca_m

# Define your dataset
X = your_data
color_map_data = some_data_for_clustering_visualization  # Optional

# Train the PCA model
pca_model = pca_m.pca_nipals(X, Num_com=3, alpha=0.95)

# Evaluate new observations using the developed PCA model
X_test = np.array([[0.9, 0.1, 0.2], [0.5, 0.4, 0.9]])
x_hat, T_score, Hotelin_T2, SPE_X = pca_m.pca_evaluation(pca_model, X_test)

print(f'x_hat={x_hat}\n', f'T_score={T_score}\n', f'Hotelin_T2={Hotelin_T2}\n', f'SPE_X={SPE_X}\n')

# Visualize PCA scores with clustering information
scores_pca = np.array([1, 2])  # Principal components for the plot
pca_m.visual_plot(pca_model, scores_pca, X_test, color_map_data)
```

#### Model Implementation as a Class:

```python
from pca_class import PCAClass as pca_c

# Create an instance of the PCA class
MyPcaModel = pca_c()

# Train the model
MyPcaModel.train(X, Num_com=3, alpha=0.95)

# Evaluate new observations using the PCA model
x_hat, T_score, Hotelin_T2, SPE_X = MyPcaModel.evaluation(X_test)

print(f'x_hat={x_hat}\n', f'T_score={T_score}\n', f'Hotelin_T2={Hotelin_T2}\n', f'SPE_X={SPE_X}\n')

# Visualize PCA scores with optional clustering variable
MyPcaModel.visual_plot(scores_pca, X_test, color_map_data)
```

#### Missing Value Estimation Example:

```python
# Generate the data
correlated_data = generate_correlated_data(N, M, alpha)

rows_for_test = 2
X_tr = correlated_data[:-rows_for_test, :]
X_tes = correlated_data[-rows_for_test:, :]
X_tes_incomplete = X_tes.copy()
columns_to_replace = [3, 1]
for col in columns_to_replace:
    X_tes_incomplete[:, col] = np.nan  # Replace with np.nan

pca_model = pca_c()
pca_model.train(X_tr, 3)

estimated_block, estimation_accuracy = pca_model.MissEstimator(X_tes_incomplete, X_tes)
print(estimation_accuracy)
```

## Advantages

- **Flexible Visualization**: The additional `color_map` parameter enhances the interpretability of PCA by enabling latent space visualization based on clustering or other data features.
- **Class and Module Flexibility**: The project allows users to choose between a class-based or module-based implementation, providing flexibility for different coding preferences.
- **Outlier Detection**: Both Hotelling's T² and SPE limits provide robust tools for identifying potential outliers in the dataset.
- **Clear and Intuitive**: Designed to be straightforward for both novice and experienced users.