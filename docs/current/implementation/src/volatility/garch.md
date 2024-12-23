# GARCH Volatility Module

## 1. Overview
This module implements GARCH volatility modeling and forecasting:
- GARCH(1,1) model fitting
- Parameter estimation
- Volatility forecasting
- Rolling predictions

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- arch: GARCH model implementation
- statistics.base: Basic calculations

### Related Modules
- volatility/har.py: Alternative volatility model
- statistics/base.py: Basic calculations
- statistics/timeseries.py: Rolling statistics

## 2. Methodology References

### Background Documents
- [VolatilityForecasting_HAR_GARCH.md](../../../references/methodologies/VolatilityForecasting_HAR_GARCH.md)
  * Section 2.1: GARCH model specification
  * Section 2.2: Parameter estimation
  * Section 2.3: Volatility forecasting

### Mathematical Foundations
```python
# GARCH(1,1) Model
σ²ₜ = ω + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁

# Log-Likelihood
L = -0.5 * Σ(log(2π) + log(σ²ₜ) + ε²ₜ/σ²ₜ)

# Forecast (1-step ahead)
σ²ₜ₊₁ = ω + α₁ε²ₜ + β₁σ²ₜ

# Multi-step Forecast
σ²ₜ₊ₖ = ω + (α₁ + β₁)ᵏ⁻¹(σ²ₜ₊₁ - ω)
```

## 3. Implementation Details

### 3.0 Class Structure
```python
# TODO: Update with proper class implementation after core/base.md
class GARCH(VolatilityModel):
    """
    GARCH volatility model implementation.
    Will inherit from base VolatilityModel class.

    Attributes:
        model: arch.univariate.GARCH model
        params: Fitted parameters
        results: Model fit results

    Notes:
        - Final implementation will use inheritance
        - Methods below will become class methods
        - Will follow composition over inheritance
    """
    pass
```

### 3.1 Core Functions
```python
class GARCHModel:
    """
    GARCH model wrapper using arch package.

    Attributes:
        model: arch.univariate.GARCH model
        params: Fitted parameters
        results: Model fit results
    """
    def __init__(self,
                 p: int = 1,
                 q: int = 1,
                 dist: str = 'normal'):
        """
        Initialize GARCH model.

        Args:
            p: ARCH order
            q: GARCH order
            dist: Error distribution ('normal', 'studentt', 'skewt')
        """
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.params = None
        self.results = None

def fit_garch(returns: pd.Series,
              p: int = 1,
              q: int = 1,
              dist: str = 'normal') -> Dict[str, float]:
    """
    Fit GARCH(p,q) model to return series.

    Args:
        returns: Return series
        p: ARCH order
        q: GARCH order
        dist: Error distribution

    Returns:
        Dictionary of parameters

    Notes:
        - Uses arch package implementation
        - Assumes zero mean
        - Returns annualized parameters
    """
    _validate_garch_inputs(returns, p, q)

    # Create and fit model
    from arch import arch_model
    model = arch_model(returns, p=p, q=q, dist=dist)
    results = model.fit(disp='off')

    return {
        'omega': results.params['omega'],
        'alpha': results.params['alpha[1]'],
        'beta': results.params['beta[1]'],
        'dist_params': results.params.get('nu', None)  # For Student-t
    }

def forecast_instantaneous(returns: pd.Series,
                           params: Dict[str, float],
                           horizon: int = 1) -> float:
    """Forecast instantaneous volatility."""
    model = arch_model(returns)
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon)

    return np.sqrt(forecast.variance.values[-1])

def forecast_rolling_volatility(returns: pd.Series,
                                window: int,
                                horizon: int = 1) -> pd.Series:
    """Generate rolling volatility forecasts."""
    forecasts = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        model = arch_model(window_data)
        results = model.fit(disp='off', update_freq=0)
        forecast = results.forecast(horizon=horizon)
        forecasts.append(np.sqrt(forecast.variance.values[-1]))

    return pd.Series(forecasts, index=returns.index[window:])

def _validate_garch_inputs(returns: pd.Series,
                           p: int,
                           q: int) -> None:
    """Validate inputs for GARCH estimation."""
    if not isinstance(returns, pd.Series):
        raise ValidationError("Returns must be a Series")
    if len(returns) < 100:
        raise ValidationError("Insufficient data for GARCH estimation")
    if p < 1 or q < 1:
        raise ValidationError("GARCH orders must be positive")

def forecast_garch_path(returns: pd.Series,
                        horizon: int,
                        annualize: bool = True) -> pd.Series:
    """
    Generate day-by-day volatility forecast path.

    Args:
        returns: Return series
        horizon: Number of days to forecast
        annualize: Whether to return annualized volatility

    Returns:
        Series of daily volatility forecasts

    Notes:
        - Uses arch package for iterative forecasting
        - Returns instantaneous (daily) volatilities
        - Can be used as input for window calculations
    """
    model = arch_model(returns)
    results = model.fit(disp='off')

    # Get forecast path
    forecast = results.forecast(horizon=horizon, method='simulation')
    daily_vol = np.sqrt(forecast.variance.values[-horizon:])

    # Create forecast dates
    last_date = returns.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='B'  # Business days
    )

    # Convert to series
    vol_path = pd.Series(daily_vol, index=forecast_dates)

    if annualize:
        vol_path *= np.sqrt(252)  # Annualize

    return vol_path

def forecast_garch_window(returns: pd.Series,
                          window: int,
                          horizon: int,
                          annualize: bool = True) -> pd.Series:
    """
    Generate rolling window volatility forecasts from GARCH path.

    Args:
        returns: Return series
        window: Window size for rolling volatility (e.g., 21 for monthly)
        horizon: How far ahead to forecast
        annualize: Whether to return annualized volatility

    Returns:
        Series of window volatility forecasts

    Notes:
        - First generates daily path
        - Then computes rolling windows
        - E.g., for 21-day vol, each point uses previous 21 daily forecasts
    """
    # Get daily volatility path
    daily_path = forecast_garch_path(returns, horizon + window - 1, annualize=False)

    # Compute rolling window volatility
    window_vol = daily_path.rolling(window).std()

    # Drop incomplete windows
    window_vol = window_vol.dropna()

    if annualize:
        window_vol *= np.sqrt(252)

    return window_vol

def _get_default_window(freq: str) -> int:
    """Get default window size based on frequency."""
    windows = {
        'D': 1,      # Daily
        'W': 5,      # Weekly
        'M': 21,     # Monthly
        'Q': 63,     # Quarterly
        'A': 252     # Annual
    }
    return windows.get(freq, 21)  # Default to monthly

### 3.2 Performance Considerations
- Cache volatility estimates
- Optimize likelihood calculation
- Handle large datasets efficiently
- Reuse parameter estimates

### 3.3 Error Handling
```python
def _validate_garch_inputs(returns: pd.Series,
                           p: int,
                           q: int) -> None:
    """Validate inputs for GARCH estimation."""
    if not isinstance(returns, pd.Series):
        raise ValidationError("Returns must be a Series")
    if len(returns) < 100:
        raise ValidationError("Insufficient data for GARCH estimation")
    if p < 1 or q < 1:
        raise ValidationError("GARCH orders must be positive")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic Volatility Forecasting
```python
# Example 1: Day-by-day volatility path
daily_vol_path = forecast_garch_path(returns, horizon=10)
print("Next 10 days volatility forecast:")
print(daily_vol_path)

# Example 2: Monthly volatility forecasts
monthly_vol = forecast_garch_window(
    returns,
    window=21,    # 21-day rolling window
    horizon=63    # 3 months ahead
)
print("3-month ahead monthly volatility forecast:")
print(monthly_vol)

# Example 3: Multiple horizons with different windows
horizons = {
    'daily': forecast_garch_path(returns, horizon=1),
    'weekly': forecast_garch_window(returns, window=5, horizon=5),
    'monthly': forecast_garch_window(returns, window=21, horizon=21),
    'quarterly': forecast_garch_window(returns, window=63, horizon=63)
}
```

### 4.2 Parameter Selection
- window: Based on data frequency
- horizon: Based on forecast needs
- p,q orders: Usually (1,1) sufficient

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known volatility patterns
   - Parameter estimation
   - Forecast accuracy

2. Edge Cases
   - High volatility periods
   - Regime changes
   - Missing data
   - Small samples

3. Performance Tests
   - Large datasets
   - Long forecast horizons
   - Multiple re-estimations

## 6. Implementation Status

### Completed Features
- [x] GARCH(1,1) estimation
- [x] Volatility forecasting
- [x] Rolling predictions
- [x] Parameter validation

### Known Limitations
- Limited to GARCH(1,1)
- No asymmetric effects
- Single volatility regime
- No distribution choice

### Future Enhancements
- Higher order models
- Asymmetric GARCH
- Student-t innovations
- Regime switching
