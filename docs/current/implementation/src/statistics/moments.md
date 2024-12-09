# Statistical Moments Module

## 1. Overview
This module implements core statistical moment calculations for financial time series, including:
- Mean returns (arithmetic and geometric)
- Standard deviation/volatility
- Skewness
- Kurtosis
- Combined statistical summaries

### Related Modules
- statistics/returns.py - Return series calculations
- statistics/distribution.py - Distribution fitting
- core/validation.py - Input validation

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy.stats: Statistical functions

## 2. Methodology References

### Background Documents
- [Arithmetic_to_Geometric_returns.md](../../../references/methodologies/Arithmetic_to_Geometric_returns.md)
  * Section 1.2: Definitions of statistical moments
  * Section 2.1: Return components and drag effects
  * Section 2.2: Volatility adjustments

### Mathematical Foundations
#### Mean Return
$$ \mu_a = \frac{1}{T}\sum_{t=1}^T r_t \quad \text{(Arithmetic mean)} $$
$$ \mu_g = \exp(\frac{1}{T}\sum_{t=1}^T \ln(1 + r_t)) - 1 \quad \text{(Geometric mean)} $$

#### Volatility
$$ \sigma = \sqrt{\frac{1}{T-1}\sum_{t=1}^T (r_t - \mu)^2} $$
$$ \sigma_{ann} = \sigma \times \sqrt{\text{annualization factor}} $$

#### Higher Moments
$$ \text{Skewness} = \frac{1}{T}\sum_{t=1}^T (\frac{r_t - \mu}{\sigma})^3 $$
$$ \text{Excess Kurtosis} = \frac{1}{T}\sum_{t=1}^T (\frac{r_t - \mu}{\sigma})^4 - 3 $$

## 3. Implementation Details

### 3.1 Core Functions

#### mean_return (previously ret_mean)
```python
def mean_return(returns: pd.Series, geometric: bool = False) -> float:
    """
    Calculate arithmetic or geometric mean return.

    Args:
        returns: Return series
        geometric: If True, calculate geometric mean

    Returns:
        Mean return value

    Validation:
    - Minimum 20 observations required
    - Returns must be finite
    - Warning if extreme values (>|20%|)
    """
```

#### stdev (previously ret_volatility)
```python
def stdev(returns: pd.Series,
          annualized: bool = True,
          frequency: str = 'D') -> float:
    """
    Calculate return standard deviation (volatility).

    Args:
        returns: Return series
        annualized: If True, annualize the result
        frequency: Return frequency for annualization

    Returns:
        Standard deviation/volatility value

    Validation:
    - Minimum 20 observations
    - Finite values only
    - Valid frequency string
    """
```

### 3.2 Performance Considerations
- Use numpy operations for vectorized calculations
- Pre-compute common terms (mean, variance) when needed multiple times
- Cache annualization factors
- Optimize for large datasets (>1000 assets)

### 3.3 Error Handling
```python
def _validate_returns(returns: pd.Series, min_periods: int = 20) -> None:
    """Validate return series for moment calculations."""
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
# Calculate all moments
stats = ret_stats(returns)
print(f"Mean: {stats['mean']:.4f}")
print(f"Volatility: {stats['volatility']:.4f}")
print(f"Skewness: {stats['skewness']:.4f}")
print(f"Kurtosis: {stats['kurtosis']:.4f}")
```

#### Annualized Metrics
```python
# Daily returns to annualized metrics
ann_vol = stdev(returns, annualized=True, frequency='D')
ann_ret = mean_return(returns) * 252  # For daily returns
```

### 4.2 Parameter Selection
- frequency: Match to return series frequency ('D', 'W', 'M', 'Q', 'A')
- min_periods: Default 20, increase for more stable estimates
- geometric: Use for compounded returns over longer periods
- annualized: True for standard financial reporting

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
- [x] Annualization handling
- [x] Input validation
- [x] Performance optimization
- [x] Error handling

### Known Limitations
- Limited to univariate series
- No handling of irregular time series
- No adjustment for autocorrelation

### Future Enhancements
- Robust moment estimators
- Rolling moment calculations
- Multivariate moment handling
- Autocorrelation adjustment options
