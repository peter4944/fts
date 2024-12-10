# Time Series Operations Module

## 1. Overview
This module implements time series transformations and rolling window calculations:
- Rolling statistical measures
- Drawdown series calculations
- PCA factor series generation
- Time series transformations

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy: PCA calculations
- statistics.base: Basic statistical calculations

### Related Modules
- statistics/base.py: Core statistical functions
- statistics/adjusted.py: Adjusted metrics
- distribution/skew_student_t.py: Distribution fitting

## 2. Methodology References

### Background Documents
- [PCA_FactorLoadingTimeSeries.md](../../../references/methodologies/PCA_FactorLoadingTimeSeries.md)
  * Section 2.1: Factor extraction
  * Section 2.2: Rolling window considerations
  * Section 3.1: Factor series generation

### Mathematical Foundations
#### Rolling Statistics
```python
# Rolling window calculation
for i in range(len(series) - window + 1):
    window_data = series[i:i+window]
    result[i] = calculate_statistic(window_data)

# Exponential weighting
α = 2/(window + 1)  # Decay factor
ewm = series.ewm(alpha=α, adjust=False)
```

#### Drawdown Series
```python
# Cumulative returns
cum_rets = (1 + returns).cumprod()

# Running maximum
running_max = cum_rets.cummax()

# Drawdown series
drawdowns = cum_rets / running_max - 1
```

#### PCA Factors
```python
# Standardize returns
X = (returns - returns.mean()) / returns.std()

# PCA decomposition
U, S, V = np.linalg.svd(X, full_matrices=False)

# Factor returns
factor_returns = X @ V.T
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def rolling_statistics(returns: pd.Series,
                      window: int,
                      statistic: str = 'mean',
                      min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate rolling statistics for return series.

    Args:
        returns: Return series
        window: Rolling window size
        statistic: Statistic to compute ('mean', 'std', 'sharpe', etc.)
        min_periods: Minimum observations required

    Returns:
        Series of rolling statistics

    Notes:
        - Handles missing data
        - Returns NaN for insufficient observations
        - Uses calendar days for window
    """
    _validate_rolling_inputs(returns, window, min_periods)

    if statistic == 'mean':
        return returns.rolling(window, min_periods=min_periods).mean()
    elif statistic == 'std':
        return returns.rolling(window, min_periods=min_periods).std()
    elif statistic == 'sharpe':
        roll_mean = returns.rolling(window, min_periods=min_periods).mean()
        roll_std = returns.rolling(window, min_periods=min_periods).std()
        return roll_mean / roll_std
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Generate drawdown series from returns.

    Args:
        returns: Return series

    Returns:
        Series of drawdown values

    Notes:
        - Returns values <= 0 (percentage loss from peak)
        - Handles cumulative return calculation
        - Accounts for compounding
    """
    _validate_returns(returns)
    cum_rets = (1 + returns).cumprod()
    running_max = cum_rets.cummax()
    return cum_rets / running_max - 1

def pca_factor_returns(returns: pd.DataFrame,
                      n_factors: int,
                      standardize: bool = True) -> pd.DataFrame:
    """
    Generate PCA factor return series.

    Args:
        returns: DataFrame of asset returns
        n_factors: Number of factors to extract
        standardize: Whether to standardize returns

    Returns:
        DataFrame of factor return series

    Notes:
        - Standardizes input by default
        - Orders factors by explained variance
        - Returns factor returns with same index as input
    """
    _validate_returns_df(returns)

    if standardize:
        X = (returns - returns.mean()) / returns.std()
    else:
        X = returns

    U, S, V = np.linalg.svd(X, full_matrices=False)
    factor_returns = pd.DataFrame(
        X @ V.T[:, :n_factors],
        index=returns.index,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    return factor_returns
```

### 3.2 Performance Considerations
- Use efficient rolling window calculations
- Cache intermediate results (cumulative returns)
- Optimize matrix operations for PCA
- Handle large return matrices efficiently

### 3.3 Error Handling
```python
def _validate_rolling_inputs(returns: pd.Series,
                           window: int,
                           min_periods: Optional[int]) -> None:
    """Validate inputs for rolling calculations."""
    if window < 2:
        raise ValidationError("Window size must be at least 2")
    if min_periods is not None and min_periods > window:
        raise ValidationError("min_periods cannot exceed window size")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Rolling Analysis
```python
# Calculate various rolling statistics
roll_vol = rolling_statistics(returns, window=60, statistic='std')
roll_sharpe = rolling_statistics(returns, window=60, statistic='sharpe')

# Plot rolling metrics
plt.plot(roll_vol.index, roll_vol)
plt.title('60-Day Rolling Volatility')
```

#### Drawdown Analysis
```python
# Generate and analyze drawdowns
dd = drawdown_series(returns)

# Find worst drawdown
worst_dd = dd.min()
worst_dd_date = dd.idxmin()
```

#### Factor Analysis
```python
# Extract principal components
factors = pca_factor_returns(returns_df, n_factors=3)

# Analyze factor contributions
explained_var = factors.var() / returns_df.var().sum()
```

### 4.2 Parameter Selection
- window: Consider data frequency and analysis horizon
- min_periods: Usually 75% of window size
- n_factors: Based on explained variance threshold

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Rolling calculations accuracy
   - Drawdown calculations
   - PCA decomposition

2. Edge Cases
   - Small windows
   - Missing data
   - Single observation windows
   - Perfect correlation in PCA

3. Performance Tests
   - Large return series
   - Many assets for PCA
   - Long rolling windows

## 6. Implementation Status

### Completed Features
- [x] Rolling statistics framework
- [x] Drawdown calculations
- [x] PCA factor generation
- [x] Input validation

### Known Limitations
- Limited rolling statistics available
- Basic PCA implementation only
- No handling of regime changes

### Future Enhancements
- Additional rolling metrics
- Robust PCA variants
- Regime-aware calculations
- Parallel processing for large datasets
