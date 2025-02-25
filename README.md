# PCA using NIPALS Algorithm

## Overview

This project implements Principal Component Analysis (PCA) using the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm. The implementation is designed to be flexible, user-friendly, and efficient, with a focus on ease of use and robust performance. The project includes both a Python implementation and a MATLAB implementation, providing users with the flexibility to choose their preferred environment.

The Python implementation has been extensively tested against the `sklearn` PCA implementation to ensure accuracy and reliability. Additionally, the project includes features for missing value estimation, outlier detection, and comprehensive visualization tools.

---

## Features

### Core Features

- **NIPALS Algorithm**: Implements PCA using the NIPALS algorithm, which is efficient for large datasets and can handle missing data.
- **Automatic Component Selection**: The number of components is automatically determined using the eigenvalue greater than 1 rule, but users can also manually specify the number of components.
- **Data Preprocessing**: Includes options for centering and scaling the data, with the ability to skip this step if desired.
- **Confidence Limits**: Uses the alpha parameter to define confidence limits for model predictions, allowing users to control the scope and accuracy of the model.
- **Missing Value Estimation**: Estimates missing values in new observations based on trends extracted from the PCA model.
- **Outlier Detection**: Utilizes Hotelling's T² and SPE (Squared Prediction Error) limits to identify potential outliers in the dataset.

### Visualization Tools

- **Score Plots**: Visualize PCA scores to assess data consistency and identify clusters or patterns.
- **Loading Plots**: Visualize the contribution of each variable to the principal components.
- **Confidence Ellipses**: Plot confidence ellipses for score plots to identify outliers and assess model validity.
- **Interactive Plots**: Includes interactive visualization options for exploring latent space and data distribution.

### Testing and Validation

- **Comprehensive Testing**: The implementation has been rigorously tested against the `sklearn` PCA implementation to ensure accuracy and reliability.
- **Unit Tests**: Includes unit tests for key functionalities, such as missing value estimation, outlier detection, and component selection.

---

## Installation

To use this project, clone the repository and ensure you have the required dependencies installed.

```bash
git clone https://github.com/Saeid-Tayebi/pca_nipals.git
cd pca-nipals
pip install -r requirements.txt
```

---

The package can also be downloaded and installed from the [Releases](https://github.com/Saeid-Tayebi/pca_nipals/releases/tag/pca_first_release) section.

## Usage

### Python Implementation

#### Importing the PCA Class

```python
from pca_nipals.pca import PcaClass as pca, pcaeval
```

#### Training the PCA Model

```python
# Generate random data
import numpy as np
Num_observation = 30
xvar = 4
X = np.random.rand(Num_observation, xvar)

# Train the PCA model
pca_model = pca().fit(X, n_component=2, alpha=0.95)
```

#### Evaluating New Observations

```python
# Generate test data
X_test = np.random.rand(10, xvar)

# Evaluate new observations
eval_result = pca_model.evaluation(X_test)
print("Reconstructed Data (xhat):", eval_result.xhat)
print("Scores (T):", eval_result.tscore)
print("Hotelling's T²:", eval_result.HT2)
print("SPE (Squared Prediction Error):", eval_result.spe)
```

#### Visualizing Results

```python
# Visualize PCA scores and statistics
pca_model.visual_plot(X_test=X_test)
```

#### Estimating Missing Values

```python
# Create incomplete data
incom_data = X_test.copy()
incom_data[0, 1] = np.nan  # Introduce missing values

# Estimate missing values
estimated_data = pca_model.MissEstimator(incom_data=incom_data)
print("Estimated Data:", estimated_data)
```

---

### MATLAB Implementation

The MATLAB implementation provides similar functionality to the Python implementation, including PCA modeling, evaluation, and visualization. Refer to the MATLAB folder for detailed usage instructions.

---

## Project Structure

```
pca-nipals/
├── pca_nipals/               # Python implementation
│   ├── pca.py                # PCA class and functions
├── tests/                    #  pytests
├── example_usage.py      # Example usage script
├── PCA_MATLAB/                   # MATLAB implementation
│   ├── pca_nipals.m          # MATLAB PCA function
│   └── example_usage.m       # Example usage script
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Testing

The project includes a comprehensive suite of unit tests to ensure the correctness and reliability of the implementation. To run the tests, navigate to the `tests` folder and execute the following command:

```bash
pytest
```

The tests cover the following functionalities:

- Component selection
- Data preprocessing
- Missing value estimation
- Outlier detection
- Comparison with `sklearn` PCA implementation

---

## Advantages

- **Flexibility**: Choose between Python and MATLAB implementations based on your preference.
- **Robustness**: Rigorously tested against `sklearn` for accuracy and reliability.
- **Ease of Use**: Simple and intuitive API for training, evaluating, and visualizing PCA models.
- **Missing Value Estimation**: Unique feature for estimating missing values in new observations.
- **Comprehensive Visualization**: Tools for exploring latent space, identifying clusters, and detecting outliers.

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
