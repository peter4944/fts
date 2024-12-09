# Using Dynamic Time Warping (DTW) to Estimate Correlation Matrix for Advanced Portfolio Diversification
The sources provided outline how to use Dynamic Time Warping (DTW) to estimate the correlation matrix for use in Advanced Portfolio Diversification (APD). The technique is presented as a modification to the Hierarchical Clustering (HC) portfolio optimisation methods of Hierarchical Risk Parity (HRP) and Hierarchical Equal Risk Contribution (HERC).

DTW is used to improve the estimation of the covariance matrix, or more specifically, the correlation matrix, used as input to the HRP and HERC models. [1, 2]
DTW is a powerful algorithm for aligning two time series that may be non-linear and out of phase. [3] This makes it suitable for financial market data, which often exhibit these characteristics. [3]
DTW works by 'warping' the time dimension of sequences to enable a point-by-point comparison. This allows it to identify patterns that would be missed by traditional linear analysis. [4] In the context of APD, this allows investors to recognise leading and lagging indicators across different market segments and identify hidden temporal relationships. [5]
Steps for using DTW in APD:
Calculate the DTW distance matrix. The formula for the DTW distance between two time series, x and y, is given in the source. [6] In essence, it involves finding the path through the two time series that minimises the total distance between corresponding points. This requires specialised software implementations like the tslearn Python package. [6, 7]
Rescale the DTW distance matrix. The calculated DTW distance matrix is rescaled to a range of 0 to 1. [6]
Convert the rescaled DTW distance matrix to a correlation matrix and then to a covariance matrix. [6] This updated covariance matrix serves as input to the hierarchical clustering model (either HRP or HERC). [6]
Perform Hierarchical Clustering using the updated covariance matrix. The steps for HRP, including hierarchical clustering, quasi-diagonalisation, and recursive bisection, are described in detail in the source. [8-14]
Advantages of using DTW for APD:
Improved correlation estimates: DTW allows for the identification of more accurate correlations between assets by accounting for non-linearity and time lags in their price movements. [5, 6]
Enhanced risk management: DTW can help portfolio managers to identify potential shifts in market trends and emerging risks by detecting when assets deviate from their historical patterns. [5]
Adaptive portfolio construction: By integrating DTW into APD strategies, investment models can become more responsive to changing market conditions, allowing for better risk mitigation and growth capture. [15]
Empirical Results and Conclusion
The source reports promising results from using the DTW-modified HC model. [7] In the majority of cases, the DTW-enhanced portfolios, such as "HRP-DTW Portfolio" and "HERC-DTW Portfolio", outperform the traditional MV portfolio, particularly in recent decades. [7, 16, 17]

The improved performance is attributed to two main reasons: [18]

More diverse weight allocation: APD strategies allocate weights to assets in a more diversified manner compared to traditional MV methods.
Multiple risk minimisation methods: APD strategies can employ various risk measures beyond volatility, further enhancing their risk-adjusted performance.
The source concludes that DTW, in conjunction with HC portfolio diversification, offers a more robust and adaptable framework for portfolio construction in the modern financial landscape. [19] This approach enables investors to achieve enhanced returns while effectively managing risk exposures. [19]

## process for handling negative correlations
Since the DTW measures the similairity of 2 time series that we can normalise on a 0 to 1 scale, series with negative correlation has a higher similairty (less co-dependence) than series with positive correlation. To handle this, we can use the following steps: 


1. Calculate the DTW distance matrix between the time series af assets A vs B, and A vs -B (negative)
2. similarity = 1 - dtw_distance
2. Pick the lowest similarity measure from 1 (implies highest correlation)
3. Convert to -1 to 1 correlation metric as:
     a) if Similarity(a,-b) < Similarity(a,b)
             Correlation(a, b) = -(2 * Similarity(a, b) - 1)
        else
             Correlation(a, b) = 2 * Similarity(a, b) - 1

To elaborate slightly:

* Similarity(a, -b) < Similarity(a, b): This condition checks if asset 'a' is more similar to the inverse of asset 'b' than to the original asset 'b'.
* Correlation(a, b) = -(2 * Similarity(a, b) - 1): If 'a' is more similar to the inverse of 'b', we compute the correlation using the similarity between 'a' and 'b' and  then negate it to reflect the negative relationship.
* Correlation(a, b) = 2 * Similarity(a, b) - 1: If 'a' is more similar to the original 'b', we directly compute the correlation using their similarity.


**Step-by-Step Methodology for DTW-based Correlation Matrix:**

1. **Data Preparation:**
   - Load the price data for your *n* assets into a suitable data structure (e.g., a NumPy array or pandas DataFrame). Let's represent this as a matrix `asset_prices` where rows correspond to assets and columns correspond to time periods.

2. **Initialization:**
   - Create an empty *n x n* matrix, `correlation_matrix`, to store the DTW-derived correlations.

3. **Pairwise Comparisons:**
   - Iterate through all unique asset pairs (i, j) where *i < j* (to avoid redundant calculations since the correlation matrix is symmetric).
     - For each pair (i, j):
       - **DTW Distance Calculation:** Compute the DTW distance between:
         - The original time series of asset *i* and the original time series of asset *j*. Denote this as `dtw_distance_original`.
         - The original time series of asset *i* and the inverse (negated) time series of asset *j*. Denote this as `dtw_distance_inverse`.
       - **Similarity Selection:** Select the minimum distance between `dtw_distance_original` and `dtw_distance_inverse`. This minimum distance, `min_dtw_distance`, represents the highest similarity between asset *i* and either the original or the inverse of asset *j*.

4. **Correlation Calculation and Storage:**
   -  **Convert to Similarity:** Transform the `min_dtw_distance` to a similarity measure: 
     `similarity = 1 - min_dtw_distance`
   - **Scale to Correlation:**  Scale the similarity to the -1 to 1 range for correlation:
     `correlation = 2 * similarity - 1`
   - **Assign to Matrix:**
     - If `dtw_distance_inverse` was the minimum distance (meaning asset *i* is more similar to the *inverse* of asset *j*), then store `-correlation` in `correlation_matrix[i, j]` and `correlation_matrix[j, i]`.
     - Otherwise (asset *i* is more similar to the original asset *j*), store `correlation` in `correlation_matrix[i, j]` and `correlation_matrix[j, i]`.

5. **Diagonal Elements:**
   - Set all diagonal elements of `correlation_matrix` to 1 (as the correlation of an asset with itself is always 1).

**This procedure systematically calculates DTW-based correlations, accounting for potential negative relationships by comparing assets with both the original and inverse forms of other assets.** 



## Using Dynamic Time Warping (DTW) to Estimate Correlation Matrix for Advanced Portfolio Diversification
The sources provided outline how to use Dynamic Time Warping (DTW) to estimate the correlation matrix for use in Advanced Portfolio Diversification (APD). The technique is presented as a modification to the Hierarchical Clustering (HC) portfolio optimisation methods of Hierarchical Risk Parity (HRP) and Hierarchical Equal Risk Contribution (HERC).

DTW is used to improve the estimation of the covariance matrix, or more specifically, the correlation matrix, used as input to the HRP and HERC models. [1, 2]
DTW is a powerful algorithm for aligning two time series that may be non-linear and out of phase. [3] This makes it suitable for financial market data, which often exhibit these characteristics. [3]
DTW works by 'warping' the time dimension of sequences to enable a point-by-point comparison. This allows it to identify patterns that would be missed by traditional linear analysis. [4] In the context of APD, this allows investors to recognise leading and lagging indicators across different market segments and identify hidden temporal relationships. [5]
Steps for using DTW in APD:
Calculate the DTW distance matrix. The formula for the DTW distance between two time series, x and y, is given in the source. [6] In essence, it involves finding the path through the two time series that minimises the total distance between corresponding points. This requires specialised software implementations like the tslearn Python package. [6, 7]
Rescale the DTW distance matrix. The calculated DTW distance matrix is rescaled to a range of 0 to 1. [6]
Convert the rescaled DTW distance matrix to a correlation matrix and then to a covariance matrix. [6] This updated covariance matrix serves as input to the hierarchical clustering model (either HRP or HERC). [6]
Perform Hierarchical Clustering using the updated covariance matrix. The steps for HRP, including hierarchical clustering, quasi-diagonalisation, and recursive bisection, are described in detail in the source. [8-14]
Advantages of using DTW for APD:
Improved correlation estimates: DTW allows for the identification of more accurate correlations between assets by accounting for non-linearity and time lags in their price movements. [5, 6]
Enhanced risk management: DTW can help portfolio managers to identify potential shifts in market trends and emerging risks by detecting when assets deviate from their historical patterns. [5]
Adaptive portfolio construction: By integrating DTW into APD strategies, investment models can become more responsive to changing market conditions, allowing for better risk mitigation and growth capture. [15]
Empirical Results and Conclusion
The source reports promising results from using the DTW-modified HC model. [7] In the majority of cases, the DTW-enhanced portfolios, such as "HRP-DTW Portfolio" and "HERC-DTW Portfolio", outperform the traditional MV portfolio, particularly in recent decades. [7, 16, 17]

The improved performance is attributed to two main reasons: [18]

More diverse weight allocation: APD strategies allocate weights to assets in a more diversified manner compared to traditional MV methods.
Multiple risk minimisation methods: APD strategies can employ various risk measures beyond volatility, further enhancing their risk-adjusted performance.
The source concludes that DTW, in conjunction with HC portfolio diversification, offers a more robust and adaptable framework for portfolio construction in the modern financial landscape. [19] This approach enables investors to achieve enhanced returns while effectively managing risk exposures. [19]

Dtw

Mistral:

Great questions! Let's clarify these points in detail:

### 1. Standardizing Returns

**Standardizing returns** involves transforming the returns to have a mean of 0 and a standard deviation of 1. This process is commonly used in various types of analysis to ensure that all variables are on the same scale, which can be crucial for comparing different assets or features.

#### Use Cases for Standardizing Returns:

- **Principal Component Analysis (PCA)**: Standardizing returns is essential in PCA to ensure that each variable contributes equally to the analysis. Without standardization, variables with larger scales can dominate the principal components.
- **Clustering**: In clustering algorithms like K-means, standardizing returns helps to ensure that the distance metrics used are not biased by the scale of the variables.
- **Machine Learning Models**: Many machine learning algorithms, such as linear regression, logistic regression, and neural networks, benefit from standardized inputs to improve convergence and performance.
- **Risk Management**: In risk management, standardizing returns can help in comparing the risk profiles of different assets on a common scale.
- **Portfolio Optimization**: In mean-variance optimization (MVO), standardizing returns can help in ensuring that the optimization process is not biased by the scale of returns.

### 2. Handling Negative Codependency with DTW

Dynamic Time Warping (DTW) measures the similarity between two time series by allowing non-linear alignment. However, traditional DTW does not capture negative codependency (i.e., when one asset moves in the opposite direction to another). To address this, you can modify the DTW distance metric to account for negative codependency.

#### Steps to Measure Negative Codependency using DTW:

1. **Modify the Distance Metric**: Use a distance metric that can capture negative codependency. One approach is to use the negative of the absolute difference, which can capture both positive and negative correlations.

2. **Calculate DTW Distance with Modified Metric**: Compute the DTW distance using the modified metric.

3. **Normalize and Convert to Similarity**: Normalize the DTW distances and convert them to a similarity measure that ranges from -1 to 1.

#### Detailed Steps

##### Step 1: Modify the Distance Metric
```python
def modified_dtw_distance(series1, series2):
    distance, path = fastdtw(series1, series2, dist=lambda x, y: -abs(x - y))
    return distance
```

##### Step 2: Calculate DTW Distance with Modified Metric
```python
# Calculate DTW distances between all pairs of assets using the modified metric
dtw_distances = pdist(standardized_returns.T, metric=modified_dtw_distance)

# Convert to a square matrix
dtw_distance_matrix = squareform(dtw_distances)
```

##### Step 3: Normalize and Convert to Similarity
```python
# Normalize DTW distances to a similarity measure
max_distance = dtw_distance_matrix.max()
min_distance = dtw_distance_matrix.min()
similarity_matrix = 1 - (dtw_distance_matrix - min_distance) / (max_distance - min_distance)

# Ensure the diagonal is 1 (self-similarity)
np.fill_diagonal(similarity_matrix, 1)

# Convert similarity to a range of -1 to 1
alternative_correlation_matrix = 2 * (similarity_matrix - 0.5)
```

##### Example Output
```python
# Display the alternative correlation matrix
print(pd.DataFrame(alternative_correlation_matrix, index=asset_returns.columns, columns=asset_returns.columns).head())
```

### Explanation

- **Modified Distance Metric**: The modified distance metric `-abs(x - y)` allows DTW to capture negative codependency by considering the negative of the absolute difference.
- **Normalization**: The DTW distances are normalized to a similarity measure that ranges from 0 to 1. Then, this similarity measure is converted to a range of -1 to 1 to mimic the traditional correlation matrix.
- **Alternative Correlation Matrix**: The resulting matrix can be used in mean-variance optimization (MVO) and other analyses that require a correlation-like measure.

By following these steps, you can create an alternative correlation matrix that captures both positive and negative codependencies between asset returns using DTW. This matrix can be used in various analyses, including portfolio optimization and risk management.

Claude

Ah, excellent point! We should modify the similarity measure to capture negative relationships, similar to how traditional correlations range from -1 to 1. Let me update the implementation.​​​​​​​​​​​​​​​​

Key changes in this updated version:

1. Sign Detection:
   - Added a sliding window approach to detect directional relationships
   - For each window, we calculate if the two series move in the same or opposite directions
   - The overall sign is determined by the majority of local agreements

2. Correlation Calculation:
   - Still uses DTW distances for magnitude
   - Multiplies the similarity measure by the sign matrix
   - Results in a correlation-like measure ranging from -1 to 1

3. Visualization:
   - Updated the heatmap to center at 0 and range from -1 to 1
   - Uses the same color scheme as traditional correlation matrices

The advantages of this approach:
- Captures both positive and negative relationships
- Maintains the ability to detect lagged relationships
- More directly comparable to traditional correlation coefficients
- Better suited for financial applications where inverse relationships are important

Would you like me to:
1. Add a sensitivity analysis for the window size parameter?
2. Implement alternative methods for detecting relationship direction?
3. Add statistical significance testing?
4. Include a method to analyze the stability of the relationships over time?​​​​​​​​​​​​​​​​

import numpy as np
from scipy.spatial.distance import squareform
from fastdtw import fastdtw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_dtw_correlation_matrix(returns_df):
    """
    Calculate correlation-like matrix using DTW distances between return series,
    accounting for both positive and negative relationships.
    
    Parameters:
    returns_df (pd.DataFrame): DataFrame where columns are assets and rows are returns
    
    Returns:
    pd.DataFrame: DTW-based correlation matrix ranging from -1 to 1
    """
    n_assets = len(returns_df.columns)
    dtw_distances = np.zeros((n_assets, n_assets))
    sign_matrix = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Get return series for assets i and j
            series1 = returns_df.iloc[:, i].values
            series2 = returns_df.iloc[:, j].values
            
            # Calculate DTW distance
            distance, path = fastdtw(series1, series2)
            dtw_distances[i, j] = distance
            dtw_distances[j, i] = distance
            
            # Calculate sign of relationship using path alignment
            path = np.array(path)
            aligned_series1 = series1[path[:, 0]]
            aligned_series2 = series2[path[:, 1]]
            
            # Calculate directional agreement using sliding windows
            window_size = min(10, len(aligned_series1) // 4)  # Adaptive window size
            agreement_scores = []
            
            for k in range(len(aligned_series1) - window_size):
                window1 = aligned_series1[k:k+window_size]
                window2 = aligned_series2[k:k+window_size]
                
                # Calculate if movements are in same or opposite direction
                trend1 = np.polyfit(np.arange(window_size), window1, 1)[0]
                trend2 = np.polyfit(np.arange(window_size), window2, 1)[0]
                agreement_scores.append(np.sign(trend1 * trend2))
            
            # Overall relationship sign is based on majority of local agreements
            relationship_sign = np.sign(np.mean(agreement_scores))
            sign_matrix[i, j] = relationship_sign
            sign_matrix[j, i] = relationship_sign
    
    # Convert distances to similarities (correlation-like measure)
    max_dist = np.max(dtw_distances)
    dtw_similarities = 1 - (dtw_distances / max_dist)
    
    # Apply signs to similarities to get correlations from -1 to 1
    dtw_correlations = dtw_similarities * sign_matrix
    
    # Set diagonal to 1
    np.fill_diagonal(dtw_correlations, 1.0)
    
    # Create DataFrame with asset labels
    dtw_corr_df = pd.DataFrame(
        dtw_correlations,
        index=returns_df.columns,
        columns=returns_df.columns
    )
    
    return dtw_corr_df

def plot_comparison(returns_df, traditional_corr=True, dtw_corr=True):
    """
    Plot and compare traditional correlation matrix with DTW-based correlation matrix
    """
    if traditional_corr and dtw_corr:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    if traditional_corr:
        # Traditional correlation matrix
        trad_corr = returns_df.corr()
        sns.heatmap(trad_corr, annot=True, cmap='coolwarm', center=0, 
                    vmin=-1, vmax=1, ax=ax1)
        ax1.set_title('Traditional Correlation Matrix')
    
    if dtw_corr:
        # DTW-based correlation matrix
        dtw_corr_matrix = calculate_dtw_correlation_matrix(returns_df)
        if traditional_corr:
            sns.heatmap(dtw_corr_matrix, annot=True, cmap='coolwarm', center=0,
                        vmin=-1, vmax=1, ax=ax2)
            ax2.set_title('DTW-based Correlation Matrix')
        else:
            sns.heatmap(dtw_corr_matrix, annot=True, cmap='coolwarm', center=0,
                        vmin=-1, vmax=1, ax=ax1)
            ax1.set_title('DTW-based Correlation Matrix')
    
    plt.tight_layout()
    return fig

# Example usage with synthetic data showing both positive and negative relationships
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']
    
    # Create sample returns with both positive and negative relationships
    returns = pd.DataFrame(np.random.randn(100, 4) * 0.02, columns=assets, index=dates)
    
    # Asset B follows Asset A with lag (positive relationship)
    returns['Asset_B'] = returns['Asset_A'].shift(5) + np.random.randn(100) * 0.01
    
    # Asset C moves opposite to Asset A with lag (negative relationship)
    returns['Asset_C'] = -returns['Asset_A'].shift(10) + np.random.randn(100) * 0.01
    
    # Asset D has a more complex relationship
    returns['Asset_D'] = np.sin(np.arange(100) * 0.1) * 0.02
    
    returns = returns.dropna()
    
    # Calculate and plot both correlation matrices
    plot_comparison(returns)
    plt.show()