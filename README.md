# PCA using NIPALS Algorithm

## Overview

This project implements Principal Component Analysis (PCA) using the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm. It includes both MATLAB and Python implementations, providing users with flexibility to choose their preferred environment. The project aims to deliver a structured and user-friendly approach to PCA, facilitating model development and visualization.

## Features

### General Features

- **Automatic Component Selection**: Default is set to the number of data variables.
- **Center-Scaling**: Data is centered and scaled by default, with an option to skip this step.
- **Alpha Parameter**: Defines the confidence limit for model predictions, allowing control over model accuracy and scope.
- **PCA Score Visualization**: Visualize PCA scores to assess data consistency and identify outliers.

### MATLAB Implementation

- **Functionality**: Provides PCA modeling and evaluation using the NIPALS algorithm.
- **Visualization**: Tools to visualize PCA scores and data distributions.
- **Ease of Use**: Designed for straightforward implementation and analysis.

### Python Implementation

- **Class and Module Options**: Provides both class-based and module-based implementations.
- **Optional Visualization**: Includes a color_map parameter for enhanced latent space visualization.
- **Outlier Detection**: Utilizes Hotelling's T² and SPE limits for identifying potential outliers.

## Usage

### MATLAB

1. **Load Data**: 
   ```matlab
   data = single_block_of_your_data;
   ```

2. **Develop PCA Model**:
   ```matlab
   mypca = pca_nipals(data);
   ```

3. **Evaluate New Observations**:
   ```matlab
   x_new = any_new_set_of_data;
   [x_hat, t_point, SPE, tsquared, x_new_scaled] = pca_evaluation(mypca, x_new);
   ```

4. **Visualize PCA Scores**:
   ```matlab
   pca_visual_plot(mypca, [1, 2], x_new);
   ```

### Python

#### Module-Based Implementation

1. **Import and Define Dataset**:
   ```python
   import pca_module as pca_m
   X = your_data
   color_map_data = some_data_for_clustering_visualization  # Optional
   ```

2. **Train PCA Model**:
   ```python
   pca_model = pca_m.pca_nipals(X, Num_com=3, alpha=0.95)
   ```

3. **Evaluate New Observations**:
   ```python
   X_test = np.array([[0.9, 0.1, 0.2], [0.5, 0.4, 0.9]])
   x_hat, T_score, Hotelin_T2, SPE_X = pca_m.pca_evaluation(pca_model, X_test)
   ```

4. **Visualize PCA Scores**:
   ```python
   scores_pca = np.array([1, 2])
   pca_m.visual_plot(pca_model, scores_pca, X_test, color_map_data)
   ```

#### Class-Based Implementation

1. **Import and Create PCA Instance**:
   ```python
   from pca_class import PCAClass as pca_c
   MyPcaModel = pca_c()
   ```

2. **Train Model**:
   ```python
   MyPcaModel.train(X, Num_com=3, alpha=0.95)
   ```

3. **Evaluate New Observations**:
   ```python
   x_hat, T_score, Hotelin_T2, SPE_X = MyPcaModel.evaluation(X_test)
   ```

4. **Visualize PCA Scores**:
   ```python
   MyPcaModel.visual_plot(scores_pca, X_test, color_map_data)
   ```

## Advantages

- **Flexible Visualization**: Enhanced PCA visualization for clustering and data feature analysis.
- **Class and Module Flexibility**: Choose between class-based or module-based implementations.
- **Outlier Detection**: Robust tools for identifying potential outliers in the dataset.
- **Intuitive Design**: Suitable for both novice and experienced users.

