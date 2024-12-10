# Skewed Student-t Distribution Module

## 1. Overview
This module implements skewed Student-t distribution fitting and related calculations:
- Distribution parameter estimation
- Distribution-specific moments
- Distribution-specific drag effects
- Distribution-adjusted performance metrics

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy: Distribution fitting, optimization
- statistics.base: Basic statistics and normalization
- statistics.adjusted: Standard drag calculations

### Related Modules
- statistics/base.py: Basic statistics
- statistics/adjusted.py: Standard adjustments
- statistics/timeseries.py: Time series operations
- statistics/base.py: Provides normalize_returns function

## 2. Methodology References

### Background Documents
- [Arithmetic_to_Geometric_returns.md](../../../references/methodologies/Arithmetic_to_Geometric_returns.md)
  * Section 4.1: Student-t distribution
  * Section 4.2: Skewness extension
  * Section 4.3: Moment calculations
  * Section 4.4: Drag effects

### Mathematical Foundations
#### Distribution Parameters
```python
# Skewed Student-t PDF
f(x; ν, λ) = c * (1 + y²/(ν-2))^(-(ν+1)/2)
where:
y = (x - μ)/(σ * √(1 + 3λ²))
c = normalization constant

# Parameters
ν: degrees of freedom (tail heaviness)
λ: skewness parameter
μ: location
σ: scale
```

#### Distribution Moments
```python
# Raw moments (up to order 4)
E[X] = μ + σλ * g(ν)
E[X²] = μ² + σ²(1 + 3λ²)ν/(ν-2)
E[X³] = cubic function of parameters
E[X⁴] = quartic function of parameters

# Standardized moments
skewness = f(λ, ν)
kurtosis = g(ν) * (1 + h(λ))
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def fit_skewed_t(returns: pd.Series,
                method: str = 'mle') -> Dict[str, float]:
    """
    Fit skewed Student-t distribution to return series.

    Args:
        returns: Return series
        method: Fitting method ('mle' or 'moments')

    Returns:
        Dict containing:
        - degrees_of_freedom (ν)
        - skewness_param (λ)
        - location (μ)
        - scale (σ)

    Notes:
        - MLE is more accurate but slower
        - Moments method is faster but less stable
        - Returns standardized parameters
    """
    _validate_returns(returns)

    # Use centralized normalization function
    norm_returns = normalize_returns(returns)  # From statistics.base

    if method == 'mle':
        return _fit_mle(norm_returns)
    else:
        return _fit_moments(norm_returns)

def calculate_student_t_volatility(params: Dict[str, float]) -> float:
    """
    Calculate volatility adjusted for Student-t distribution.

    Args:
        params: Distribution parameters from fit_skewed_t

    Returns:
        Adjusted annualized volatility

    Notes:
        - Accounts for heavy tails via degrees of freedom
        - Accounts for skewness via skewness parameter
        - Returns annualized value
    """
    df = params['degrees_of_freedom']
    skew = params['skewness_param']
    scale = params['scale']

    # Adjust for heavy tails and skewness
    vol = scale * np.sqrt((df/(df-2)) * (1 + 3*skew**2))
    return vol

def calculate_student_t_geometric_return(arithmetic_return: float,
                                        vol: float,
                                        params: Dict[str, float]) -> float:
    """
    Calculate geometric return adjusted for Student-t distribution.

    Args:
        arithmetic_return: Annualized arithmetic return
        vol: Base volatility
        params: Distribution parameters

    Returns:
        Adjusted geometric return

    Notes:
        Based on methodology:
        R_g,adj = R_a - σ²/2 - σ²/2 × 2/(v-2) - γ₃σ³/6
    """
    df = params['degrees_of_freedom']
    skew = params['skewness_param']

    # Calculate drag components
    variance_drag = vol**2 / 2
    tail_cost = (vol**2 / 2) * (2 / (df - 2))
    skew_cost = skew * (vol**3) / 6

    return arithmetic_return - variance_drag - tail_cost - skew_cost

def calculate_student_t_sharpe_ratio(returns: pd.Series,
                                     params: Dict[str, float],
                                     rf_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio adjusted for Student-t distribution.

    Args:
        returns: Return series
        params: Distribution parameters
        rf_rate: Risk-free rate

    Returns:
        Student-t adjusted Sharpe ratio

    Notes:
        - Uses adjusted geometric return (accounts for all drags)
        - Uses Student-t adjusted volatility
        - Provides more conservative estimate than standard Sharpe
    """
    # Get annualized arithmetic return
    mean_ret = params['location']
    ann_factor = _infer_annualization_factor(returns)
    ann_ret = mean_ret * ann_factor

    # Calculate adjusted volatility
    adj_vol = calculate_student_t_volatility(params)

    # Calculate adjusted geometric return
    adj_geo_ret = calculate_student_t_geometric_return(ann_ret, adj_vol, params)

    # Calculate adjusted Sharpe ratio
    return (adj_geo_ret - rf_rate) / adj_vol

def calculate_student_t_drag(params: Dict[str, float],
                            vol: float) -> float:
    """
    Calculate total drag for Student-t distribution.

    Args:
        params: Distribution parameters from fit_skewed_t
        vol: Base volatility

    Returns:
        Total drag adjustment

    Notes:
        Based on methodology:
        R_g,adj = R_a - σ²/2 - σ²/2 × 2/(v-2) - γ₃σ³/6
    """
    df = params['degrees_of_freedom']
    skew = params['skewness_param']

    # Calculate components
    variance_drag = vol**2 / 2
    fat_tail_drag = (vol**2 / 2) * (2 / (df - 2))
    skew_drag = skew * (vol**3) / 6

    return variance_drag + fat_tail_drag + skew_drag

def calculate_student_t_heavy_tail_drag(params: Dict[str, float],
                                        vol: float) -> float:
    """
    Calculate heavy tail component of drag.

    Args:
        params: Distribution parameters
        vol: Base volatility

    Returns:
        Heavy tail drag component

    Notes:
        Fat Tail Adjustment = σ²/2 × 2/(v-2)
    """
    df = params['degrees_of_freedom']
    return (vol**2 / 2) * (2 / (df - 2))

def calculate_student_t_kurtosis_drag(params: Dict[str, float],
                                        vol: float) -> float:
    """
    Calculate kurtosis component of drag.

    Args:
        params: Distribution parameters
        vol: Base volatility

    Returns:
        Kurtosis drag component

    Notes:
        - For Student-t, kurtosis is determined by degrees of freedom
        - This is included in heavy_tail_drag for Student-t
    """
    return calculate_student_t_heavy_tail_drag(params, vol)

def calculate_student_t_skew_drag(params: Dict[str, float],
                                    vol: float) -> float:
    """
    Calculate skewness component of drag.

    Args:
        params: Distribution parameters
        vol: Base volatility

    Returns:
        Skewness drag component

    Notes:
        Skewness Cost ≈ γ₃σ³/6
    """
    skew = params['skewness_param']
    return skew * (vol**3) / 6

def calculate_student_t_mtd(vol_target: float,
                            sr_adj: float,
                            params: Dict[str, float],
                            lambda_param: float = 0.2) -> float:
    """
    Calculate Maximum Theoretical Drawdown under Student-t.

    Args:
        vol_target: Target volatility
        sr_adj: Adjusted Sharpe ratio
        params: Distribution parameters
        lambda_param: Skewness impact parameter (default 0.2)

    Returns:
        Maximum Theoretical Drawdown

    Notes:
        Based on methodology:
        MTD = (vol_target/(2*SR_adj)) * √(v/(v-2)) * (1 - γ₃λ)
    """
    df = params['degrees_of_freedom']
    skew = params['skewness_param']

    return (vol_target / (2 * sr_adj)) * np.sqrt(df / (df - 2)) * (1 - skew * lambda_param)
```

### 3.2 Performance Considerations
- Cache fitted parameters
- Use efficient optimization methods
- Pre-compute common terms
- Handle large datasets efficiently

### 3.3 Error Handling
```python
def _validate_distribution_params(params: Dict[str, float]) -> None:
    """Validate distribution parameters."""
    if params['degrees_of_freedom'] <= 2:
        raise ValidationError("Degrees of freedom must be > 2 for finite variance")
    if not np.isfinite(params['skewness_param']):
        raise ValidationError("Skewness parameter must be finite")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Distribution Fitting
```python
# Fit distribution and get parameters
params = fit_skewed_t(returns)

# Calculate distribution-specific metrics
vol = calculate_student_t_volatility(params)
sr = calculate_student_t_sharpe_ratio(returns, params)
```

### 4.2 Parameter Selection
- method: 'mle' for accuracy, 'moments' for speed
- min_periods: 100+ for stable parameter estimates
- convergence criteria for optimization

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known parameter recovery
   - Moment calculations
   - Drag effects

2. Edge Cases
   - High/low degrees of freedom
   - Extreme skewness
   - Small sample sizes

3. Performance Tests
   - Large return series
   - Multiple fits
   - Optimization convergence

## 6. Implementation Status

### Completed Features
- [x] Distribution fitting (MLE)
- [x] Basic moment calculations
- [x] Volatility adjustments
- [x] Sharpe ratio adjustments
- [x] Student-t geometric returns
- [x] Student-t drag calculations
- [x] Maximum Theoretical Drawdown

### Not Yet Implemented
- [ ] Student-t specific drag calculations:
  * calculate_student_t_drag
  * calculate_student_t_heavy_tail_drag
  * calculate_student_t_kurtosis_drag
  * calculate_student_t_skew_drag

### Known Limitations
- Limited to univariate series
- No time-varying parameters
- Computationally intensive for large datasets
- Parameter estimation sensitivity

### Future Enhancements
- Additional fitting methods
- Time-varying parameters
- Multivariate extensions
- Performance optimizations
