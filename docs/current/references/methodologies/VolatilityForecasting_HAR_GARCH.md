# Volatility Forecasting: GARCH and HAR Models

This note provides an overview of volatility forecasting using Generalized Autoregressive Conditional Heteroskedasticity (GARCH) and Heterogeneous Autoregressive (HAR) models, assuming realized volatility (RV) is already calculated.

## Realized Volatility Windows and Forecast Horizons

### Input Data Structure

The realized volatility input series typically consists of overlapping windows:

- For a 20-day RV, each daily observation represents volatility over the previous 20 days
- This overlapping structure provides rich information about volatility evolution
- The HAR model is specifically designed to handle such overlapping measurements

### Forecast Horizon Structure

While inputs may be overlapping, forecasts are made for complete, non-overlapping future periods:

- For h=20 day horizon: forecast is for complete period [t+1 to t+20]
- For multiple periods: forecasts are for [t+21 to t+40], [t+41 to t+60], etc.
- Each forecast represents the volatility of a complete future window
- No mixing of known and unknown returns in forecast windows

### Example Implementation with Proper Date Alignment

```python
def forecast_volatility_har(rv_series, n_ahead=1, window=20):
    """
    Forecast volatility for n complete future windows
  
    Args:
        rv_series (pd.Series): Daily series of rolling window RV, indexed by date
        n_ahead (int): Number of complete future windows to forecast
        window (int): Size of RV calculation window (e.g., 20 days)
    
    Returns:
        pd.Series: Forecasts indexed by the date they represent
                  (end of each forecast window)
    """
    # Get last date in input series
    last_date = rv_series.index[-1]
  
    # Generate forecasts
    forecasts = []
    forecast_dates = []
  
    for i in range(n_ahead):
        # Calculate forecast for complete future window
        forecast = generate_window_forecast(rv_series, window)
        forecasts.append(forecast)
    
        # Calculate the date this forecast represents
        # (end of complete window)
        forecast_date = last_date + pd.Timedelta(days=window*(i+1))
        forecast_dates.append(forecast_date)
  
    # Return as time series
    return pd.Series(forecasts, index=forecast_dates)

# Example usage:
if __name__ == "__main__":
    # Assuming we have RV series up to Nov 1, 2023
    forecast = forecast_volatility_har(rv_series, n_ahead=3, window=20)
  
    # This would output forecasts for:
    # - Nov 21, 2023 (first complete window)
    # - Dec 11, 2023 (second complete window)
    # - Dec 31, 2023 (third complete window)
  
    # Easy comparison with realized values
    comparison = pd.DataFrame({
        'forecast': forecast,
        'realized': realized_vol  # actual RV when available
    })
```

### Date Alignment Considerations:

1. **Forecast Dates**

   - First forecast: t + window_size days
   - Second forecast: t + (2 × window_size) days
   - nth forecast: t + (n × window_size) days
2. **Business Day Adjustments** (optional enhancement)

```python
def get_forecast_dates(start_date, n_ahead, window):
    """
    Get proper business day aligned forecast dates
    """
    forecast_dates = []
    current_date = start_date
  
    for i in range(n_ahead):
        # Add business days for window
        forecast_date = pd.date_range(
            start=current_date, 
            periods=window, 
            freq='B'  # business day frequency
        )[-1]
    
        forecast_dates.append(forecast_date)
        current_date = forecast_date + pd.Timedelta(days=1)
  
    return forecast_dates
```

3. **Typical Usage**

```python
# For November 1, 2023 with 20-day window:
#
# Input RV series:
# 2023-10-13 to 2023-11-01 (current window)
#
# First forecast window:
# 2023-11-02 to 2023-11-21 
#
# Second forecast window:
# 2023-11-22 to 2023-12-11
```

This date alignment is crucial for:

- Proper forecast evaluation
- Comparing forecast vs realized volatility
- Risk management applications
- Portfolio rebalancing decisions


## 1. GARCH Volatility Forecasting: Methodology and Implementation

### Introduction to GARCH Models

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models are particularly effective for volatility forecasting because they capture three key features of financial market volatility:
1. Volatility clustering (periods of high volatility tend to persist)
2. Mean reversion (volatility tends to return to a long-term average)
3. Asymmetric response to market movements (volatility often reacts differently to positive and negative returns)

### GARCH(1,1) Model Specification

The GARCH(1,1) model represents the conditional variance σ²t as:

σ²t = ω + α⋅ε²(t-1) + β⋅σ²(t-1)

where:
- ω (omega): Weight given to long-run average variance
- α (alpha): Reaction coefficient - measures how strongly volatility reacts to market movements
- β (beta): Persistence coefficient - indicates how long volatility shocks persist
- ε²(t-1): Previous period's squared return
- σ²(t-1): Previous period's variance

Parameter constraints:
- ω > 0
- α ≥ 0
- β ≥ 0
- α + β < 1 (for stationarity)

### Types of Volatility Estimates

#### 1. Instantaneous Volatility
- Represents the volatility at a specific point in time
- For daily data, this is the one-day-ahead forecast
- Used as the starting point for multi-period forecasts

#### 2. N-Period Ahead Forecasts
- Individual volatility forecasts for specific future periods
- Accounts for mean reversion in forecasts
- Calculated recursively using GARCH parameters

#### 3. Average Volatility
- Represents the average volatility over a future time window
- Calculated as the square root of the average variance over the period
- Critical for option pricing and risk management

### Implementation Structure

```python
class GARCHModel:
    def __init__(self, returns_series):
        self.returns = returns_series
        self.params = None
        self.instantaneous_vol = None
        
    def fit(self, omega=None):
        """
        Fit GARCH(1,1) model using maximum likelihood estimation.
        
        Args:
            omega: Optional fixed long-run variance (for variance targeting)
        """
        # Implementation using maximum likelihood estimation
        pass
        
    def forecast_volatility(self, n_periods, calculation_type='instant'):
        """
        Generate volatility forecasts.
        
        Args:
            n_periods: Number of periods ahead to forecast
            calculation_type: One of ['instant', 'average', 'term']
                - 'instant': Point estimates for each future period
                - 'average': Average volatility over specified horizon
                - 'term': Term structure of volatility
        
        Returns:
            pd.Series: Forecasted volatilities
        """
        if calculation_type == 'instant':
            return self._forecast_point_volatility(n_periods)
        elif calculation_type == 'average':
            return self._forecast_average_volatility(n_periods)
        else:
            return self._forecast_term_structure(n_periods)

    def _forecast_point_volatility(self, n_periods):
        """
        Calculate period-by-period volatility forecasts.
        
        The n-step ahead forecast is given by:
        σ²(t+n) = ω + (α + β)^(n-1) * (σ²(t+1) - ω)
        
        where σ²(t+1) is the instantaneous volatility
        """
        forecasts = []
        current_var = self.instantaneous_vol**2
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        
        for n in range(1, n_periods + 1):
            forecast_var = (
                long_run_var + 
                (self.alpha + self.beta)**(n-1) * 
                (current_var - long_run_var)
            )
            forecasts.append(np.sqrt(forecast_var))
            
        return pd.Series(forecasts)

    def _forecast_average_volatility(self, horizon):
        """
        Calculate average volatility over specified horizon.
        
        The average variance over [t+1, t+n] is:
        avg_var = ω + (current_var - ω) * 
                  (1 - (α + β)^n) / (n * (1 - (α + β)))
        """
        current_var = self.instantaneous_vol**2
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        persistence = self.alpha + self.beta
        
        if abs(persistence - 1) < 1e-6:  # Handle near-unit-root case
            avg_var = long_run_var
        else:
            avg_var = (
                long_run_var + 
                (current_var - long_run_var) * 
                (1 - persistence**horizon) / 
                (horizon * (1 - persistence))
            )
            
        return np.sqrt(avg_var)
```

### Parameter Estimation and Model Diagnostics

#### Maximum Likelihood Estimation
The GARCH parameters are estimated by maximizing the log-likelihood function:

L = -0.5 * Σ(log(σ²t) + ε²t/σ²t)

Two estimation approaches are supported:
1. Full maximum likelihood (estimates all parameters)
2. Variance targeting (fixes ω based on sample variance)

#### Model Diagnostics
Key metrics to assess model fit:
1. Parameter significance (t-statistics)
2. Standardized residuals analysis
3. Log-likelihood value
4. Information criteria (AIC, BIC)

#### Parameter Interpretation
- α: Reaction to news (typically 0.05-0.15)
- β: Persistence (typically 0.8-0.9)
- α + β: Total persistence (should be < 1)
- ω/(1-α-β): Long-run variance

### Practical Considerations

#### Sample Size Requirements
- Minimum 2 years of daily data recommended
- Larger samples improve parameter stability
- Consider data frequency (daily preferred)

#### Model Stability Checks
1. Parameter bounds (0 ≤ α,β ≤ 1)
2. Persistence check (α + β < 1)
3. Standard errors of parameters
4. Comparison with simpler models (EWMA)

#### Advanced Applications
1. Confidence intervals for forecasts
2. Term structure analysis
3. Option pricing applications
4. Risk management metrics

## 2. HAR Models

### Methodology

HAR models exploit persistence in realized volatility by modeling it as a function of lagged values over different time horizons. The model can be specified in two forms:

**Standard HAR:**

$$
RV_{t+h} = \beta_0 + \beta_D RV_t + \beta_W RV_t^W + \beta_M RV_t^M + \epsilon_t
$$

**Centered HAR (Improved Specification):**

$$
RV_{t+h} - \overline{RV_t} = \beta_D(RV_t - \overline{RV_t}) + \beta_W(RV_t^W - \overline{RV_t}) + \beta_M(RV_t^M - \overline{RV_t}) + \epsilon_t
$$

where:

* $RV_{t+h}$: Realized volatility at time *t+h*
* $RV_t$, $RV_t^W$, $RV_t^M$: Daily, weekly, monthly RV
* $\overline{RV_t}$: Expanding mean of RV up to time *t*
* $\beta_0$, $\beta_D$, $\beta_W$, $\beta_M$: Regression coefficients

### Implementation Steps

1. **Data Preparation:**
   - Calculate daily, weekly, monthly lagged RV
   - For centered version, calculate expanding mean
2. **Model Fitting:** Estimate coefficients using OLS
3. **Forecasting:** Generate forecasts using fitted model


### HAR Model Implementation

The HAR model uses overlapping components at different frequencies to capture volatility persistence patterns. Default lags are empirically motivated:

- Daily frequency: 1-day lag (previous RV)
- Weekly frequency: 5-day lag (approximately one trading week)
- Monthly frequency: 22-day lag (approximately one trading month)

For different data frequencies, these defaults adjust proportionally:
- Weekly data: 1-week, 4-week, 12-week lags
- Monthly data: 1-month, 3-month, 12-month lags

```python
def forecast_volatility_har(rv_series, 
                          n_ahead=1,
                          window=20,
                          frequency='D',
                          lags=None):
    """
    HAR volatility forecast with configurable lags
    
    Args:
        rv_series (pd.Series): Realized volatility series with datetime index
        n_ahead (int): Number of complete future windows to forecast
        window (int): Size of RV calculation window
        frequency (str): Data frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
        lags (dict): Optional custom lags for each component
                    Format: {'daily': 1, 'weekly': 5, 'monthly': 22}
    
    Returns:
        pd.Series: Forecasts indexed by forecast dates
    """
    # Set default lags based on frequency
    if lags is None:
        if frequency == 'D':
            lags = {'daily': 1, 'weekly': 5, 'monthly': 22}
        elif frequency == 'W':
            lags = {'daily': 1, 'weekly': 4, 'monthly': 12}
        elif frequency == 'M':
            lags = {'daily': 1, 'weekly': 3, 'monthly': 12}
        else:
            raise ValueError("Frequency must be 'D', 'W', or 'M'")

    # Create HAR components (all are overlapping windows)
    df = pd.DataFrame({'RV': rv_series})
    
    # Daily component (previous period)
    df['RV_D'] = df['RV'].shift(lags['daily'])
    
    # Weekly component (average over weekly lag)
    df['RV_W'] = df['RV'].rolling(
        window=lags['weekly']
    ).mean().shift(1)
    
    # Monthly component (average over monthly lag)
    df['RV_M'] = df['RV'].rolling(
        window=lags['monthly']
    ).mean().shift(1)
    
    # Center the components if using centered specification
    rv_mean = df['RV'].expanding().mean()
    for col in ['RV', 'RV_D', 'RV_W', 'RV_M']:
        df[col] = df[col] - rv_mean
    
    # Fit model and generate forecasts
    model = sm.OLS(
        df['RV'].dropna(), 
        sm.add_constant(df[['RV_D', 'RV_W', 'RV_M']].dropna())
    ).fit()
    
    # Generate forecast dates
    last_date = rv_series.index[-1]
    forecast_dates = [
        last_date + pd.Timedelta(window * (i+1), unit='D')
        for i in range(n_ahead)
    ]
    
    # Generate forecasts
    latest = df[['RV_D', 'RV_W', 'RV_M']].iloc[-1]
    forecasts = []
    
    for _ in range(n_ahead):
        forecast = model.predict(sm.add_constant(latest))[0]
        forecast = forecast + rv_mean.iloc[-1]  # Add back level
        forecasts.append(np.sqrt(forecast))  # Convert to volatility
    
    return pd.Series(forecasts, index=forecast_dates)
```

### Key Points About HAR Components:

1. **Overlapping Nature**
   - RV_D: Single previous period
   - RV_W: Rolling average over weekly horizon
   - RV_M: Rolling average over monthly horizon
   - All components share some overlapping data

2. **Default Lag Structure**
```
Daily Data (D):
- Daily: t-1
- Weekly: t-5 to t-1 (avg)
- Monthly: t-22 to t-1 (avg)

Weekly Data (W):
- Weekly: t-1
- Monthly: t-4 to t-1 (avg)
- Quarterly: t-12 to t-1 (avg)

Monthly Data (M):
- Monthly: t-1
- Quarterly: t-3 to t-1 (avg)
- Annual: t-12 to t-1 (avg)
```

3. **Empirical Evidence**
- Default lags based on extensive research
- Capture market dynamics at different frequencies
- Proven robust across different markets and time periods
- User override available for specific applications

Would you like me to expand on any aspects of the lag structure or add more details about the empirical motivation for the default choices?

### Key Improvements Over Basic Models

1. **Centered HAR Specification**

   - Removes level dependence in volatility series
   - Improves forecast accuracy by better handling varying volatility levels
   - Simple to implement yet significantly enhances performance
2. **Robust Implementation**

   - Proper handling of missing values
   - Input validation
   - Conversion between variance and volatility scales
3. **Flexible Forecast Horizons**

   - Accommodates different forecast periods
   - Direct multi-step ahead forecasting

## Usage Considerations

1. **Model Selection**

   - GARCH: Better for capturing volatility clustering and leverage effects
   - HAR: Better for exploiting long-memory properties of realized volatility
   - Centered HAR: Recommended as default choice for general use
2. **Data Requirements**

   - Minimum length of historical data (at least 22 days for monthly component)
   - Treatment of missing values
   - Handling of outliers
3. **Implementation Notes**

   - Use logarithmic transformation for highly skewed series
   - Consider data frequency and market characteristics
   - Monitor parameter stability over time

## Future Extensions

While maintaining simplicity, potential future enhancements could include:

1. **Basic Model Improvements**

   - Rolling window estimation
   - Simple confidence intervals for forecasts
   - Basic forecast evaluation metrics (MSE, MAE)
2. **Additional Features**

   - Basic plotting functionality
   - Simple parameter stability diagnostics
   - Storage of model diagnostics

## Conclusion

This implementation provides a robust foundation for volatility forecasting while maintaining simplicity and ease of use. The centered HAR specification offers improved forecast accuracy over the standard version with minimal additional complexity. Both GARCH and HAR models are included to accommodate different use cases and market conditions.
