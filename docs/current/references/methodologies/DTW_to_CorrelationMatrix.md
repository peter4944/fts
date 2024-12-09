# Using DTW to Generate Financial Correlation Matrices

## 1. Introduction

Dynamic Time Warping (DTW) offers a powerful alternative to traditional correlation measures for financial time series analysis. While standard correlation metrics assume contemporaneous relationships between assets, DTW can detect similarities between time series even when they are out of phase or exhibit non-linear relationships.

The key advantages of DTW for financial correlation estimation include:

- Ability to handle asynchronous price movements
- Detection of lagged relationships between assets
- Robustness to non-linear relationships
- Improved estimation of dependencies during market stress periods

## 2. Dynamic Time Warping Overview

DTW works by finding an optimal alignment between two time series by warping the time dimension, while preserving essential shape characteristics. Originally developed for speech recognition, DTW has found wide application in financial markets where price movements often exhibit complex temporal relationships.

Key references for the theoretical foundations:

- Sakoe & Chiba (1978) for the original DTW algorithm
- Tsinaslanidis & Kugiumtzis (2014) for financial applications
- Bankó & Abonyi (2012) for multivariate extensions


## 3. DTW Correlation Matrix Generation Process

### Understanding Negative Correlations in DTW

DTW fundamentally measures similarity between time series, yielding a distance/similarity measure between 0 and 1. However, financial correlations range from -1 to +1, where:

- +1 indicates perfect positive correlation
- -1 indicates perfect negative correlation
- 0 indicates no correlation

The challenge is that DTW would see a perfectly negatively correlated series as very dissimilar (high distance/low similarity), while in finance, we need to identify both strong positive and negative relationships. For example, if stock A goes up 1% while stock B goes down 1% consistently, they have a strong negative correlation (-1) but DTW would show them as dissimilar.

### Step 1: Data Preprocessing and Parameters

```python
def standardize_returns(returns):
    """
    Standardize returns to have mean=0 and std=1
    """
    return (returns - np.mean(returns)) / np.std(returns)

def get_default_window_size(frequency='daily'):
    """
    Get default DTW window size based on return frequency.
    Window sizes chosen to capture meaningful lead/lag relationships
    while avoiding spurious matches.
  
    Parameters:
    -----------
    frequency : str
        'daily', 'weekly', or 'monthly'
  
    Returns:
    --------
    int : recommended window size for DTW calculation
    """
    window_sizes = {
        'daily': 20,    # Capture up to 1 month of trading lag
        'weekly': 8,    # Capture up to 2 months of trading lag
        'monthly': 3    # Capture up to 1 quarter of trading lag
    }
    return window_sizes.get(frequency, 20)  # Default to daily if not specified
```


### Step 2: Calculate DTW Distances

For each pair of assets (A, B):

1. Standardize both return series
2. Calculate two DTW distances:
   - Between A and B (captures positive correlation)
   - Between A and -B (captures negative correlation)
   - Take the minimum of these distances (highest similarity)

```python
def calculate_dtw_similarity(series_a, series_b, frequency='daily', window_size=None):
    """
    Calculate DTW similarity between two standardized return series,
    checking both original and inverted relationships
  
    Parameters:
    -----------
    series_a, series_b : array-like
        Return series to compare
    frequency : str
        Data frequency for default window size
    window_size : int, optional
        Override default window size if specified
    """
    # Standardize both series
    std_a = standardize_returns(series_a)
    std_b = standardize_returns(series_b)
  
    # Set window size
    radius = window_size if window_size is not None else get_default_window_size(frequency)
  
    # Calculate DTW distance for both original and inverted series
    distance_original, _ = fastdtw(std_a, std_b, radius=radius)
    distance_inverse, _ = fastdtw(std_a, -std_b, radius=radius)
  
    # Convert distances to similarities (0 to 1 scale)
    sim_original = 1 - (distance_original / max(distance_original, distance_inverse))
    sim_inverse = 1 - (distance_inverse / max(distance_original, distance_inverse))
  
    return sim_original, sim_inverse
```

### Step 3: Convert to Correlation Scale

Convert the similarity measures to correlations (-1 to +1 scale):

```python
def convert_to_correlation(sim_original, sim_inverse):
    """
    Convert DTW similarities to correlation scale.
    Logic:
    - If inverse similarity is higher, series are negatively correlated
    - If original similarity is higher, series are positively correlated
    """
    if sim_inverse > sim_original:
        # Negative correlation case
        return -(2 * sim_inverse - 1)
    # Positive correlation case
    return 2 * sim_original - 1
```

### Step 4: Build Complete Correlation Matrix

```python
def build_dtw_correlation_matrix(returns_df):
    """
    Build full correlation matrix from returns dataframe
    """
    n_assets = returns_df.shape[1]
    corr_matrix = np.eye(n_assets)  # Initialize with 1s on diagonal
  
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Get standardized returns for both assets
            series_i = returns_df.iloc[:, i]
            series_j = returns_df.iloc[:, j]
          
            # Calculate similarities
            sim_orig, sim_inv = calculate_dtw_similarity(series_i, series_j)
          
            # Convert to correlation
            correlation = convert_to_correlation(sim_orig, sim_inv)
          
            # Fill symmetric matrix
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
          
    return corr_matrix
```

### Why Standardization Matters

Standardization of returns is crucial for DTW-based correlation estimation for several reasons:

1. **Scale Independence**: Assets with different volatilities can be compared fairly

   - Example: A stock moving ±5% daily vs one moving ±0.5% daily
   - Without standardization, the more volatile asset would dominate the DTW distance
2. **Numerical Stability**: Standardization helps ensure:

   - DTW algorithm converges more reliably
   - Distance calculations are numerically stable
   - Consistent results across different scales
3. **Comparison Validity**: Makes DTW distances meaningful for:

   - Cross-asset comparisons
   - Different time periods
   - Different market regimes
4. **Improved Negative Correlation Detection**:

   - Makes the comparison between original and inverted series more meaningful
   - Ensures the magnitude of movements is comparable when checking for negative relationships

The standardization step should always be performed before DTW calculation, not after, as it affects the path optimization within the DTW algorithm itself.

## 4. Handling non-overlapping data




## 5. Implementation Considerations

### Recommended Libraries

- `fastdtw`: Efficient DTW implementation
- `tslearn`: Comprehensive time series tools including DTW
- `scipy`: For distance calculations and matrix operations

### Key Parameters

- Window size for DTW (controls allowed warping)
- Distance metric (typically Euclidean)
- Preprocessing method (standardization, returns calculation)

## 5. Applications in Finance

### Portfolio Optimization

- Input to mean-variance optimization
- Risk decomposition
- Hierarchical clustering for portfolio construction

### Risk Management

- Dynamic hedging relationships
- Stress testing
- Correlation breakdown detection

## 7. Limitations and Best Practices

### Limitations

- Computational intensity for large datasets
- Parameter sensitivity
- Need for careful preprocessing

### Best Practices

- Standardize returns before applying DTW
- Consider rolling window analysis for dynamic relationships
- Validate results against traditional measures
- Test robustness to different parameter choices

References:
For more detailed implementations, refer to:

- Howard & Putniņš (2024) "To lead or to lag? Measuring asynchronicity in financial time-series using DTW"
- Bankó & Abonyi (2012) "Correlation based dynamic time warping of multivariate time series"

Would you like me to expand on any particular section or add specific code examples for implementation?
