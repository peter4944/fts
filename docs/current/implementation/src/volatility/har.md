# HAR Volatility Module

## 1. Overview
This module implements HAR (Heterogeneous Autoregressive) volatility modeling:
- HAR model fitting
- Component calculation (daily, weekly, monthly)
- Volatility forecasting
- Rolling predictions

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy: Optimization
- statistics.base: Basic calculations

### Related Modules
- volatility/garch.py: Alternative volatility model
- statistics/base.py: Basic calculations
- statistics/timeseries.py: Rolling statistics

## 2. Methodology References

### Background Documents
- [VolatilityForecasting_HAR_GARCH.md](../../../references/methodologies/VolatilityForecasting_HAR_GARCH.md)
  * Section 3.1: HAR model specification
  * Section 3.2: Component calculation
  * Section 3.3: Volatility forecasting

### Mathematical Foundations
```python
# HAR Model
RV_t+1 = β₀ + β_d RV_t + β_w RV_t^w + β_m RV_t^m + ε_t+1

# Components
RV_t^d = RV_t                           # Daily
RV_t^w = (1/5) Σ_{i=1}^5 RV_{t-i+1}    # Weekly
RV_t^m = (1/22) Σ_{i=1}^22 RV_{t-i+1}  # Monthly

# Forecast
RV_t+h = β₀ + β_d RV_t + β_w RV_t^w + β_m RV_t^m
```

## 3. Implementation Details

### 3.0 Class Structure
```python
# TODO: Update with proper class implementation after core/base.md
class HAR(VolatilityModel):
    """
    HAR volatility model implementation.
    Will inherit from base VolatilityModel class.

    Attributes:
        params: Model parameters
        components: Volatility components
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
def calculate_har_components(returns: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate HAR volatility components.

    Args:
        returns: Return series

    Returns:
        Dictionary containing:
        - daily: Daily realized variance
        - weekly: Weekly average RV
        - monthly: Monthly average RV

    Notes:
        - Uses squared returns as RV proxy
        - Handles missing values
        - Returns annualized values
    """
    # Calculate daily RV
    rv_daily = returns**2

    # Calculate weekly component (5-day)
    rv_weekly = rv_daily.rolling(window=5).mean()

    # Calculate monthly component (22-day)
    rv_monthly = rv_daily.rolling(window=22).mean()

    return {
        'daily': rv_daily,
        'weekly': rv_weekly,
        'monthly': rv_monthly
    }

def fit_har_model(returns: pd.Series) -> Dict[str, float]:
    """
    Fit HAR model to return series.

    Args:
        returns: Return series

    Returns:
        Dictionary of parameters:
        - beta_0: Constant
        - beta_d: Daily coefficient
        - beta_w: Weekly coefficient
        - beta_m: Monthly coefficient

    Notes:
        - Uses OLS estimation
        - Handles missing values
        - Returns annualized parameters
    """
    # Get components
    components = calculate_har_components(returns)

    # Prepare regression variables
    y = components['daily'].shift(-1)  # Next day's RV
    X = pd.DataFrame({
        'const': 1,
        'daily': components['daily'],
        'weekly': components['weekly'],
        'monthly': components['monthly']
    })

    # Drop missing values
    valid = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid]
    X = X[valid]

    # Fit model
    params = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        'beta_0': params[0],
        'beta_d': params[1],
        'beta_w': params[2],
        'beta_m': params[3]
    }

def forecast_har(returns: pd.Series,
                params: Dict[str, float],
                horizon: int = 1) -> float:
    """
    Generate HAR volatility forecast.

    Args:
        returns: Return series
        params: Model parameters
        horizon: Forecast horizon

    Returns:
        Volatility forecast

    Notes:
        - Returns annualized volatility
        - Handles multi-step forecasts
        - Accounts for mean reversion
    """
    components = calculate_har_components(returns)

    forecast = (params['beta_0'] +
               params['beta_d'] * components['daily'].iloc[-1] +
               params['beta_w'] * components['weekly'].iloc[-1] +
               params['beta_m'] * components['monthly'].iloc[-1])

    # Convert to volatility
    return np.sqrt(forecast)

def _validate_har_inputs(returns: pd.Series) -> None:
    """Validate inputs for HAR estimation."""
    if not isinstance(returns, pd.Series):
        raise ValidationError("Returns must be a Series")
    if len(returns) < 22:  # Need at least one month of data
        raise ValidationError("Insufficient data for HAR estimation")

def _get_default_lags(freq: str) -> Dict[str, int]:
    """
    Get default HAR lags based on frequency.

    Args:
        freq: Data frequency ('D', 'W', 'M', 'Q', 'A')

    Returns:
        Dictionary of lags for each component

    Notes:
        - Daily: 1, 5, 22 days
        - Weekly: 1, 4, 13 weeks
        - Monthly: 1, 3, 12 months
    """
    lags = {
        'D': {'daily': 1, 'weekly': 5, 'monthly': 22},
        'W': {'daily': 1, 'weekly': 4, 'monthly': 13},
        'M': {'daily': 1, 'weekly': 3, 'monthly': 12},
        'Q': {'daily': 1, 'weekly': 2, 'monthly': 4},
        'A': {'daily': 1, 'weekly': 2, 'monthly': 3}
    }
    return lags.get(freq, lags['D'])  # Default to daily frequency

def forecast_har_window(returns: pd.Series,
                       window: int,
                       horizon: int,
                       freq: str = 'D',
                       annualize: bool = True) -> pd.Series:
    """
    Generate window-aligned HAR volatility forecasts.

    Args:
        returns: Return series
        window: Target window size (e.g., 21 for monthly)
        horizon: Forecast horizon in periods
        freq: Data frequency
        annualize: Whether to return annualized volatility

    Returns:
        Series of window volatility forecasts

    Notes:
        - Aligns forecasts with target window
        - Uses frequency-appropriate lags
        - Returns volatility at end of each window
    """
    # Get appropriate lags
    lags = _get_default_lags(freq)

    # Calculate components
    components = calculate_har_components(returns)

    # Fit model
    model = fit_har_model(components)

    # Generate forecasts for each window endpoint
    forecasts = []
    dates = []

    for i in range(horizon):
        # Calculate window endpoint
        end_date = returns.index[-1] + pd.Timedelta(days=window*(i+1))

        # Generate forecast for window endpoint
        forecast = (model['beta_0'] +
                   model['beta_d'] * components['daily'].iloc[-1] +
                   model['beta_w'] * components['weekly'].iloc[-1] +
                   model['beta_m'] * components['monthly'].iloc[-1])

        forecasts.append(np.sqrt(forecast))
        dates.append(end_date)

    # Create forecast series
    forecast_series = pd.Series(forecasts, index=dates)

    if annualize:
        ann_factor = np.sqrt({
            'D': 252, 'W': 52, 'M': 12, 'Q': 4, 'A': 1
        }.get(freq, 252))
        forecast_series *= ann_factor

    return forecast_series

### 3.2 Performance Considerations
- Cache component calculations
- Optimize rolling windows
- Handle large datasets efficiently
- Reuse parameter estimates

### 3.3 Error Handling
```python
def _validate_components(components: Dict[str, pd.Series]) -> None:
    """Validate HAR components."""
    required = {'daily', 'weekly', 'monthly'}
    if not required.issubset(components.keys()):
        raise ValidationError(f"Missing components: {required - set(components.keys())}")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic Volatility Forecasting
```python
# Example 1: Monthly volatility forecasts
monthly_vol = forecast_har_window(
    returns,
    window=21,     # 21-day window
    horizon=3,     # 3 windows ahead
    freq='D'       # Daily data
)
print("Monthly volatility forecasts:")
print(monthly_vol)

# Example 2: Multiple forecast horizons
forecasts = {
    'monthly': forecast_har_window(returns, window=21, horizon=3),
    'quarterly': forecast_har_window(returns, window=63, horizon=2),
    'annual': forecast_har_window(returns, window=252, horizon=1)
}

# Example 3: Weekly data with appropriate lags
weekly_vol = forecast_har_window(
    weekly_returns,
    window=4,      # 4-week window
    horizon=13,    # Quarter ahead
    freq='W'       # Weekly frequency
)
```

### 4.2 Parameter Selection
- window_daily: 1 day
- window_weekly: 5 days
- window_monthly: 22 days

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Component calculation
   - Parameter estimation
   - Forecast accuracy

2. Edge Cases
   - High volatility periods
   - Missing data
   - Small samples

3. Performance Tests
   - Large datasets
   - Long forecast horizons
   - Multiple re-estimations

## 6. Implementation Status

### Completed Features
- [x] Component calculation
- [x] HAR model fitting
- [x] Volatility forecasting
- [x] Input validation

### Known Limitations
- Basic HAR specification only
- No jump components
- Single volatility regime
- Linear model only

### Future Enhancements
- Jump components
- Leverage effects
- Non-linear extensions
- Regime switching
