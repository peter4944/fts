# Basic Statistics Module

## 1. Overview
The base statistics module implements fundamental statistical calculations for financial time series analysis. It provides:
- Basic return statistics (mean, std, skew, kurtosis)
- Simple correlation/covariance calculations
- Core statistical utilities

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy.stats: Statistical functions

### Related Modules
- statistics/adjusted.py: Uses these base calculations for adjusted metrics
- statistics/timeseries.py: Uses these for rolling calculations
- distribution/skew_student_t.py: Uses these for distribution fitting

## 2. Methodology References
- Basic statistical formulas follow standard definitions
- Annualization follows market conventions:
  * Daily: √252
  * Weekly: √52
  * Monthly: √12
  * Quarterly: √4
  * Annual: 1

### Mathematical Foundations
#### Return Statistics
```python
# Period Statistics
μ_period = (1/T) ∑ rₜ
σ_period = √[(1/(T-1)) ∑(rₜ - μ)²]

# Annualized Statistics
μ_annual = μ_period × factor
σ_annual = σ_period × √factor

# Skewness
skew = (1/T) ∑((rₜ - μ)/σ)³

# Raw Kurtosis (normal = 3)
kurt = (1/T) ∑((rₜ - μ)/σ)⁴

# Convert to excess kurtosis if needed (normal = 0)
excess_kurt = kurt - 3
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def mean_return(returns: pd.Series, geometric: bool = False) -> float:
    """
    Calculate period (non-annualized) arithmetic or geometric mean return.

    Args:
        returns: Return series
        geometric: If True, calculate geometric mean

    Returns:
        Period mean return value

    Notes:
        - For geometric mean: exp(mean(log(1 + returns))) - 1
        - For arithmetic mean: mean(returns)
    """
    _validate_returns(returns)
    if geometric:
        return np.exp(np.mean(np.log1p(returns))) - 1
    return returns.mean()

def annualized_return(returns: pd.Series, geometric: bool = False) -> float:
    """
    Calculate annualized mean return.

    Args:
        returns: Return series
        geometric: If True, use geometric mean

    Returns:
        Annualized mean return value

    Notes:
        - Uses period mean × annualization factor
        - Factor depends on return frequency (e.g., 252 for daily)
    """
    freq = _infer_frequency(returns)
    return mean_return(returns, geometric) * ANNUALIZATION_FACTORS[freq]

def stdev(returns: pd.Series) -> float:
    """
    Calculate period (non-annualized) standard deviation.

    Args:
        returns: Return series

    Returns:
        Period standard deviation value

    Notes:
        - Uses n-1 degrees of freedom
    """
    _validate_returns(returns)
    return returns.std(ddof=1)

def volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Return series

    Returns:
        Annualized volatility value

    Notes:
        - Uses period std × sqrt(annualization factor)
        - Factor depends on return frequency (e.g., 252 for daily)
    """
    freq = _infer_frequency(returns)
    return stdev(returns) * np.sqrt(ANNUALIZATION_FACTORS[freq])

def skewness(returns: pd.Series) -> float:
    """
    Calculate return series skewness.

    Args:
        returns: Return series

    Returns:
        Skewness value

    Notes:
        - Uses Fisher-Pearson standardized moment coefficient
        - Normalized by N for consistency with other packages
    """
    _validate_returns(returns)
    return scipy.stats.skew(returns, bias=False)

def kurtosis(returns: pd.Series) -> float:
    """
    Calculate return series raw kurtosis.

    Args:
        returns: Return series

    Returns:
        Raw kurtosis value (normal = 3)

    Notes:
        - Returns raw kurtosis (normal distribution = 3)
        - For excess kurtosis, subtract 3 from result
        - Uses Fisher-Pearson standardized moment coefficient
        - Normalized by N for consistency with other packages

    Example:
        # Get raw kurtosis (normal = 3)
        k = kurtosis(returns)

        # Convert to excess kurtosis (normal = 0)
        excess_k = k - 3
    """
    _validate_returns(returns)
    return scipy.stats.kurtosis(returns, bias=False, excess=False)

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple return series.

    Args:
        returns: DataFrame of return series

    Returns:
        Correlation matrix

    Notes:
        - Uses pairwise complete observations
        - Returns symmetric matrix
    """
    _validate_returns_df(returns)
    return returns.corr(method='pearson')

def normalize_returns(returns: pd.Series) -> pd.Series:
    """
    Normalize return series to zero mean and unit variance.

    Args:
        returns: Return series

    Returns:
        Normalized series (μ=0, σ=1)

    Notes:
        - Used for distribution fitting
        - Used for DTW calculations
        - Used for other statistical comparisons
    """
    _validate_returns(returns)
    return (returns - returns.mean()) / returns.std()
```

### 3.2 Performance Considerations
- Use numpy operations for vectorized calculations
- Pre-compute common terms (mean, variance) when needed multiple times
- Cache annualization factors
- Handle large matrices efficiently

### 3.3 Error Handling
```python
def _validate_returns(returns: pd.Series, min_periods: int = 20) -> None:
    """Validate return series for statistical calculations."""
    if len(returns) < min_periods:
        raise ValidationError(
            f"Insufficient observations: {len(returns)} < {min_periods}"
        )
    if not np.isfinite(returns).all():
        raise ValidationError("Returns contain non-finite values")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic Statistical Analysis
```python
# Calculate all basic statistics
stats = {
    'period_mean': mean_return(returns),
    'annualized_mean': annualized_return(returns),
    'period_std': stdev(returns),
    'volatility': volatility(returns),
    'skewness': skewness(returns),
    'raw_kurtosis': kurtosis(returns),  # normal = 3
    'excess_kurtosis': kurtosis(returns) - 3  # normal = 0
}

# Calculate correlation matrix
corr_matrix = correlation_matrix(returns_df)
```

### 4.2 Parameter Selection
- min_periods: Default 20 for stable estimates
- annualized: True for standard financial reporting
- geometric: True for longer period returns

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known mean/variance for synthetic data
   - Comparison with numpy/scipy implementations
   - Annualization factor correctness

2. Edge Cases
   - Single value series
   - All zero returns
   - Missing values
   - Extreme values

3. Performance Tests
   - Large series (>100k observations)
   - Multiple assets (>1000 series)
   - Memory usage monitoring

## 6. Implementation Status

### Completed Features
- [x] Basic moment calculations
- [x] Correlation/covariance matrices
- [x] Input validation
- [x] Performance optimization
- [x] Error handling

### Known Limitations
- Limited to univariate series
- No handling of irregular time series
- No adjustment for autocorrelation

### Future Enhancements
- Robust moment estimators
- Multivariate moment handling
- Autocorrelation adjustment options
