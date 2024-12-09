# PCA - Factor Loadings TimeSeries

To calculate principal components from a correlation matrix and generate a time series of factor loadings over time from the underlying asset prices, follow these steps:

### Step 1: Prepare the Data

1. **Collect Asset Prices**: Gather historical price data for the assets you are interested in.
2. **Calculate Returns**: Compute the log returns or simple returns from the price data.

### Step 2: Compute the Correlation Matrix

1. **Standardize Returns**: Standardize the returns to have a mean of 0 and a standard deviation of 1.
2. **Correlation Matrix**: Calculate the correlation matrix from the standardized returns.

### Step 3: Perform Principal Component Analysis (PCA)

1. **Eigenvalue Decomposition**: Perform eigenvalue decomposition on the correlation matrix to obtain eigenvalues and eigenvectors.
2. **Sort Eigenvalues**: Sort the eigenvalues in descending order and arrange the corresponding eigenvectors accordingly.

### Step 4: Determine the Number of Principal Components

1. **Variance Explained**: Calculate the cumulative variance explained by the principal components.
2. **Threshold**: Set a threshold (x%) for the variance explained by the first n components.
3. **Select Components**: Determine the number of components (n) that explain at least x% of the variance.

### Step 5: Calculate Factor Loadings

1. **Factor Loadings**: The factor loadings are the eigenvectors corresponding to the selected principal components.

### Step 6: Generate Time Series of Factor Loadings

1. **Project Returns**: Project the standardized returns onto the selected eigenvectors to obtain the time series of factor loadings.

### Detailed Steps

#### Step 1: Prepare the Data

```python
import numpy as np
import pandas as pd

# Example: Load asset prices into a DataFrame
asset_prices = pd.read_csv('asset_prices.csv', index_col='Date', parse_dates=True)

# Calculate log returns
log_returns = np.log(asset_prices / asset_prices.shift(1))
log_returns = log_returns.dropna()
```

#### Step 2: Compute the Correlation Matrix

```python
# Standardize returns
standardized_returns = (log_returns - log_returns.mean()) / log_returns.std()

# Compute the correlation matrix
correlation_matrix = standardized_returns.corr()
```

#### Step 3: Perform Principal Component Analysis (PCA)

```python
# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
```

#### Step 4: Determine the Number of Principal Components

```python
# Calculate cumulative variance explained
cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

# Set threshold
threshold = 0.9  # Example: 90%

# Determine the number of components
n_components = np.argmax(cumulative_variance >= threshold) + 1
```

#### Step 5: Calculate Factor Loadings

```python
# Factor loadings are the eigenvectors corresponding to the selected components
factor_loadings = sorted_eigenvectors[:, :n_components]
```

#### Step 6: Generate Time Series of Factor Loadings

```python
# Project returns onto the selected eigenvectors
factor_scores = standardized_returns.dot(factor_loadings)

# Factor scores are the time series of factor loadings
factor_scores.columns = [f'Factor {i+1}' for i in range(n_components)]
```

### Example Output

```python
print(factor_scores.head())
```

This process will give you the time series of factor loadings over time, which can be used for further analysis or modeling.

To extend the process to calculate the residual return for each asset, which represents the idiosyncratic return series, follow these additional steps:

### Step 7: Calculate Residual Returns

1. **Reconstruct Returns**: Use the factor loadings and factor scores to reconstruct the returns explained by the principal components.
2. **Calculate Residuals**: Subtract the reconstructed returns from the original standardized returns to obtain the residual returns.

### Detailed Steps

#### Step 7: Calculate Residual Returns

```python
# Reconstruct returns using factor loadings and factor scores
reconstructed_returns = factor_scores.dot(factor_loadings.T)

# Calculate residual returns
residual_returns = standardized_returns - reconstructed_returns

# Residual returns are the idiosyncratic return series for each asset
residual_returns.columns = asset_prices.columns
```

### Full Process Summary

1. **Prepare the Data**: Collect asset prices and calculate returns.
2. **Compute the Correlation Matrix**: Standardize returns and compute the correlation matrix.
3. **Perform PCA**: Perform eigenvalue decomposition and sort eigenvalues and eigenvectors.
4. **Determine the Number of Principal Components**: Set a threshold for variance explained and select the number of components.
5. **Calculate Factor Loadings**: Obtain factor loadings from the selected eigenvectors.
6. **Generate Time Series of Factor Loadings**: Project returns onto the selected eigenvectors.
7. **Calculate Residual Returns**: Reconstruct returns and calculate residual returns.

### Example Code

```python
import numpy as np
import pandas as pd

# Step 1: Prepare the Data
asset_prices = pd.read_csv('asset_prices.csv', index_col='Date', parse_dates=True)
log_returns = np.log(asset_prices / asset_prices.shift(1))
log_returns = log_returns.dropna()

# Step 2: Compute the Correlation Matrix
standardized_returns = (log_returns - log_returns.mean()) / log_returns.std()
correlation_matrix = standardized_returns.corr()

# Step 3: Perform PCA
eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 4: Determine the Number of Principal Components
cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
threshold = 0.9  # Example: 90%
n_components = np.argmax(cumulative_variance >= threshold) + 1

# Step 5: Calculate Factor Loadings
factor_loadings = sorted_eigenvectors[:, :n_components]

# Step 6: Generate Time Series of Factor Loadings
factor_scores = standardized_returns.dot(factor_loadings)
factor_scores.columns = [f'Factor {i+1}' for i in range(n_components)]

# Step 7: Calculate Residual Returns
reconstructed_returns = factor_scores.dot(factor_loadings.T)
residual_returns = standardized_returns - reconstructed_returns
residual_returns.columns = asset_prices.columns

# Example Output
print(residual_returns.head())
```

### Explanation

- **Reconstructed Returns**: These are the returns explained by the selected principal components.
- **Residual Returns**: These are the idiosyncratic returns, representing the part of the returns that cannot be explained by the principal components.

This process will give you the residual returns for each asset, which can be used for further analysis, such as identifying asset-specific risks or evaluating the performance of individual assets independent of market factors.
