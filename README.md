# PCA using NIPALS Algorithm

## Overview
This project implements Principal Component Analysis (PCA) using the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm. Designed with ease of use in mind, this implementation allows users to quickly develop PCA models for their datasets and visualize the results. The project emphasizes a structured and flexible approach to PCA, making it more applicable and user-friendly than typical algorithms found in MATLAB's regression library.

## Features
- **Automatic Component Selection**: By default, the number of components is chosen as the number of data variables.
- **Center-Scaling**: The function centers and scales the data by default. Users can choose to skip this step by setting the `to_be_scaled` parameter to `0`.
- **Modeling Framework (`alpha`)**: An `alpha` parameter (ranging from 0 to 1) defines the modeling framework within which the model is valid. A smaller alpha value restricts the model framework, increasing prediction accuracy for new observations. A higher alpha value allows a broader range of observations to be considered, though it may reduce prediction accuracy.
- **PCA Score Visualization**: Visualize PCA scores to assess if new observations are similar to the training dataset. This feature helps in identifying outliers and ensuring data consistency.
- **Ease of Use**: The code is designed to be intuitive and straightforward, allowing users to develop PCA models without deep technical knowledge.

## Usage

### Function Inputs:
- `X`: The input dataset (training data) for which the PCA model is to be developed.
- `num_components` (optional): The number of components to extract. By default, the function uses the number of X variable.
- `to_be_scaled` (optional): A binary flag (1 or 0) that determines whether the data should be centered and scaled. Default is `1` (yes).
- `alpha` (optional): A parameter ranging from 0 to 1 that defines the modeling framework within which the model is valid. Default is `1`.

### Function Output:
- **PCA Model Structure**: The function returns a structured output containing:
  - Scores and loadings
  - Eigenvalues
  - Centering and scaling values (if applied)
  - Additional metrics for new observations
- **Visualization**: Score plots that help visualize the similarity of new observations to the training dataset.

### Example Usage:
```matlab
% Load your data
data = single_block_of_your_data;

% Develop PCA model using the NIPALS algorithm
mypca = pca_nipals(data);

% Evaluate new observations using the developed PCA model
x_new = any_new_set_of_data;
[x_hat, t_point, SPE, tsquared, x_new_scaled] = pca_evaluation(mypca, x_new);

% Visualize PCA scores
pca_visual_plot(mypca, [1, 2], x_new);
