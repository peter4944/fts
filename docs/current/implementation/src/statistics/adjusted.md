# Adjusted Statistics Module

## 1. Overview
This module implements adjusted statistical measures that account for:
- Higher moment effects (variance drag, kurtosis, skewness)
- Geometric vs arithmetic return adjustments
- Risk-adjusted performance metrics (various Sharpe ratio variants)
- Position sizing metrics (Kelly fractions)

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy.stats: Statistical functions
- statistics.base: Basic statistical calculations

### Related Modules
- statistics/base.py: Core statistical calculations
- distribution/skew_student_t.py: Distribution-specific adjustments
- statistics/timeseries.py: Rolling metrics

## 2. Methodology References

### Background Documents
- [Arithmetic_to_Geometric_returns.md](../../../references/methodologies/Arithmetic_to_Geometric_returns.md)
  * Section 2.1: Variance drag
  * Section 2.2: Higher moment adjustments
  * Section 3.1: Sharpe ratio adjustments
  * Section 4.1: Kelly criterion

### Mathematical Foundations
#### Drag Effects
```python
# Variance Drag
variance_drag = -σ²/2

# Kurtosis Drag (raw kurtosis)
kurtosis_drag = -(k-3) * σ⁴/24

# Skewness Drag
skewness_drag = -s * σ³/6

# Total Drag
total_drag = variance_drag + kurtosis_drag + skewness_drag
```

#### Geometric Return
```python
# Arithmetic to Geometric Conversion
r_g = r_a + variance_drag + kurtosis_drag + skewness_drag
```

#### Performance Metrics
```python
# Standard Sharpe Ratio
SR = (r_a - r_f) / σ_a

# Geometric Sharpe Ratio
GSR = (r_g - r_f) / σ_a

# Probabilistic Sharpe Ratio
PSR = Φ((SR - SR_b) * √(n-1))
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def variance_drag(volatility: float) -> float:
    """
    Calculate variance drag effect.

    Args:
        volatility: Annualized volatility

    Returns:
        Variance drag adjustment

    Notes:
        - Represents -σ²/2 component of geometric return
        - Always negative for non-zero volatility
    """
    return -0.5 * volatility ** 2

def kurtosis_drag(kurtosis: float, volatility: float) -> float:
    """
    Calculate kurtosis drag effect.

    Args:
        kurtosis: Raw kurtosis (normal = 3)
        volatility: Annualized volatility

    Returns:
        Kurtosis drag adjustment

    Notes:
        - Uses raw kurtosis (not excess)
        - Represents -(k-3)σ⁴/24 component
    """
    return -(kurtosis - 3) * volatility ** 4 / 24

def skew_drag(skewness: float, vol: float) -> float:
    """
    Calculate skewness drag effect.

    Args:
        skewness: Skewness parameter
        vol: Volatility

    Returns:
        Skewness drag adjustment

    Notes:
        Based on methodology:
        Skewness Drag = γ₃σ³/6
    """
    return (skewness * vol**3) / 6

def geometric_sharpe_ratio(returns: pd.Series,
                         rf_rate: float = 0.0,
                         include_higher_moments: bool = True) -> float:
    """
    Calculate geometric Sharpe ratio with optional higher moment adjustments.

    Args:
        returns: Return series
        rf_rate: Risk-free rate
        include_higher_moments: Include kurtosis/skew adjustments

    Returns:
        Geometric Sharpe ratio

    Notes:
        - Accounts for variance drag by default
        - Optionally includes higher moment effects
    """
    ann_ret = annualized_return(returns)
    vol = volatility(returns)

    # Start with variance drag
    adj_ret = ann_ret + variance_drag(vol)

    if include_higher_moments:
        k = kurtosis(returns)
        s = skewness(returns)
        adj_ret += kurtosis_drag(k, vol)
        adj_ret += skew_drag(s, vol)

    return (adj_ret - rf_rate) / vol

def adj_geometric_return(returns: pd.Series,
                         include_higher_moments: bool = True) -> float:
    """
    Calculate annualized geometric return with higher moment adjustments.

    Args:
        returns: Return series
        include_higher_moments: Include kurtosis/skew adjustments

    Returns:
        Adjusted annualized geometric return

    Notes:
        - Always includes variance drag
        - Optionally includes higher moment effects
        - Returns annualized value
    """
    # Get base annualized arithmetic return
    ann_ret = annualized_return(returns)
    vol = volatility(returns)

    # Always apply variance drag
    adj_ret = ann_ret + variance_drag(vol)

    if include_higher_moments:
        k = kurtosis(returns)  # raw kurtosis
        s = skewness(returns)
        adj_ret += kurtosis_drag(k, vol)
        adj_ret += skew_drag(s, vol)

    return adj_ret

def adj_volatility(returns: pd.Series,
                  include_higher_moments: bool = True) -> float:
    """
    Calculate volatility adjusted for higher moments.

    Args:
        returns: Return series
        include_higher_moments: Include kurtosis adjustment

    Returns:
        Adjusted annualized volatility

    Notes:
        - Adjusts for heavy tails via kurtosis
        - Adjusts for asymmetry via skewness
        - Returns annualized value
    """
    vol = volatility(returns)

    if include_higher_moments:
        k = kurtosis(returns)  # raw kurtosis
        s = skewness(returns)
        # Adjust for heavy tails
        vol *= np.sqrt(1 + ((k - 3) * vol**2) / 4 + (s**2 * vol**2) / 6)

    return vol

def adj_geometric_sharpe_ratio(returns: pd.Series,
                               rf_rate: float = 0.0,
                               include_higher_moments: bool = True) -> float:
    """
    Calculate fully adjusted geometric Sharpe ratio.

    Args:
        returns: Return series
        rf_rate: Risk-free rate
        include_higher_moments: Include all higher moment adjustments

    Returns:
        Adjusted geometric Sharpe ratio

    Notes:
        - Uses adjusted geometric return in numerator
        - Uses adjusted volatility in denominator
        - Accounts for:
          * Variance drag
          * Kurtosis effects (return and volatility)
          * Skewness effects (return only)
    """
    adj_ret = adj_geometric_return(returns, include_higher_moments)
    adj_vol = adj_volatility(returns, include_higher_moments)

    return (adj_ret - rf_rate) / adj_vol

def calculate_vol_target(returns: pd.Series,
                         rf_rate: float = 0.0) -> float:
    """
    Calculate optimal volatility target (Kelly fraction) under normal distribution.

    Args:
        returns: Return series
        rf_rate: Risk-free rate

    Returns:
        Optimal volatility target

    Notes:
        Based on methodology:
        - Under normal distribution, optimal vol target = Sharpe ratio
        - Uses arithmetic returns
    """
    ann_ret = annualized_return(returns)
    vol = volatility(returns)
    return (ann_ret - rf_rate) / vol

def calculate_adj_vol_target(returns: pd.Series,
                            rf_rate: float = 0.0) -> float:
    """
    Calculate adjusted volatility target accounting for higher moments.

    Args:
        returns: Return series
        rf_rate: Risk-free rate

    Returns:
        Adjusted optimal volatility target

    Notes:
        Based on methodology:
        - Uses adjusted geometric returns
        - Uses adjusted volatility
        - Provides more conservative target than unadjusted version
    """
    # Get adjusted geometric return
    adj_geo_ret = adj_geometric_return(returns)

    # Get adjusted volatility
    adj_vol = adj_volatility(returns)

    # Calculate adjusted target
    return (adj_geo_ret - rf_rate) / adj_vol

def calculate_normal_mtd(returns: pd.Series,
                        include_higher_moments: bool = True) -> float:
    """
    Calculate Maximum Theoretical Drawdown under normal distribution.

    Args:
        returns: Return series
        include_higher_moments: Include adjustments for higher moments

    Returns:
        Maximum Theoretical Drawdown

    Notes:
        Based on methodology:
        - Base case (no higher moments): MTD = 0.5 (50%)
        - With higher moments: MTD = σ_adj/(2×SR_adj)
    """
    if not include_higher_moments:
        return 0.5  # 50% at full Kelly

    # Get adjusted volatility and Sharpe
    adj_vol = adj_volatility(returns)
    adj_sr = adj_geometric_sharpe_ratio(returns)

    return adj_vol / (2 * adj_sr)
```

### 3.2 Performance Considerations
- Cache common calculations (volatility, moments)
- Validate inputs early
- Use vectorized operations where possible

### 3.3 Error Handling
```python
def _validate_sharpe_inputs(returns: pd.Series,
                          benchmark_sr: float,
                          min_periods: int = 60) -> None:
    """Validate inputs for Sharpe ratio calculations."""
    if len(returns) < min_periods:
        raise ValidationError(
            f"Insufficient observations for Sharpe: {len(returns)} < {min_periods}"
        )
    if not np.isfinite(benchmark_sr):
        raise ValidationError("Benchmark Sharpe must be finite")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Geometric Return Calculation
```python
# Calculate geometric return with all adjustments
vol = volatility(returns)
k = kurtosis(returns)
s = skewness(returns)

geo_ret = arithmetic_return + variance_drag(vol) \
                          + kurtosis_drag(k, vol) \
                          + skew_drag(s, vol)
```

#### Sharpe Ratio Variants
```python
# Standard vs Geometric Sharpe
sr = standard_sharpe_ratio(returns)
gsr = geometric_sharpe_ratio(returns)

# Probabilistic Sharpe
psr = probabilistic_sharpe_ratio(returns, benchmark_sr=1.0)
```

### 4.2 Parameter Selection
- include_higher_moments: True for longer horizons
- min_periods: 60+ for stable Sharpe estimates
- trials: Consider multiple testing when using PSR

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known drag effects
   - Sharpe ratio calculations
   - Higher moment impacts

2. Edge Cases
   - Zero volatility
   - Extreme kurtosis/skewness
   - Small sample sizes

3. Performance Tests
   - Large return series
   - Multiple Sharpe calculations
   - Memory usage monitoring

## 6. Implementation Status

### Completed Features
- [x] Variance drag calculations
- [x] Kurtosis/skewness adjustments
- [x] Geometric Sharpe ratio
- [x] Probabilistic Sharpe ratio

### Known Limitations
- Assumes IID returns
- No regime switching
- Limited distribution assumptions

### Future Enhancements
- Additional distribution adjustments
- Time-varying parameter support
- Regime-aware calculations
