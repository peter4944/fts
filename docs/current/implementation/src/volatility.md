# Volatility Module Implementation

## 1. Overview

The volatility module implements:
1. Realized volatility calculations
2. HAR (Heterogeneous Autoregressive) model
3. GARCH model fitting and forecasting

## 2. Realized Volatility

```python
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from ..core.validation import ValidationError
from ..core import ANNUALIZATION_FACTORS, get_annualization_factor

class RealizedVolatility:
    """Calculate realized volatility measures."""

    @staticmethod
    def realized_volatility(returns: pd.Series,
                            frequency: str = 'D') -> float:
        """
        Calculate realized volatility for given frequency.

        Args:
            returns: Return series
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

        Returns:
            Annualized realized volatility
        """
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be pandas Series")

        if frequency not in ANNUALIZATION_FACTORS:
            raise ValidationError(
                f"Unsupported frequency: {frequency}. "
                f"Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
            )

        # Calculate period volatility
        period_vol = np.sqrt(np.sum(returns ** 2) / len(returns))

        # Annualize based on frequency
        annualization_factor = get_annualization_factor(frequency)
        return period_vol * annualization_factor

    @staticmethod
    def rolling_volatility(returns: pd.Series,
                          window: int = 21,
                          min_periods: Optional[int] = None,
                          frequency: str = 'D') -> pd.Series:
        """
        Calculate rolling realized volatility.

        Args:
            returns: Return series
            window: Rolling window size
            min_periods: Minimum periods required
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

        Returns:
            Series of annualized rolling volatilities
        """
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be pandas Series")

        if frequency not in ANNUALIZATION_FACTORS:
            raise ValidationError(
                f"Unsupported frequency: {frequency}. "
                f"Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
            )

        # Calculate rolling sum of squared returns
        squared_returns = returns ** 2
        rolling_sum = squared_returns.rolling(
            window=window,
            min_periods=min_periods
        ).sum()

        # Convert to volatility
        rolling_vol = np.sqrt(rolling_sum / window)

        # Annualize based on frequency
        annualization_factor = get_annualization_factor(frequency)
        rolling_vol *= annualization_factor

        return rolling_vol

    @staticmethod
    def har_components(returns: pd.Series,
                      frequency: str = 'D') -> Dict[str, pd.Series]:
        """
        Calculate HAR components (daily, weekly, monthly RV).

        Args:
            returns: Return series
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

        Returns:
            Dictionary with period, weekly, monthly annualized RV series
        """
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be pandas Series")

        if frequency not in ANNUALIZATION_FACTORS:
            raise ValidationError(
                f"Unsupported frequency: {frequency}. "
                f"Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
            )

        # Calculate squared returns
        squared_returns = returns ** 2

        # Period RV (e.g., daily, weekly, etc.)
        period_rv = squared_returns

        # Longer period averages based on frequency
        periods_per_week = {
            'D': 5,     # 5 days per week
            'W': 1,     # Already weekly
            'M': 0.25,  # ~1/4 month per week
            'Q': 0.08,  # ~1/12 quarter per week
            'A': 0.02   # ~1/52 year per week
        }

        periods_per_month = {
            'D': 22,    # ~22 days per month
            'W': 4,     # ~4 weeks per month
            'M': 1,     # Already monthly
            'Q': 0.33,  # ~1/3 quarter per month
            'A': 0.08   # ~1/12 year per month
        }

        # Calculate window sizes based on frequency
        week_window = max(1, round(periods_per_week[frequency]))
        month_window = max(1, round(periods_per_month[frequency]))

        # Calculate components
        weekly_rv = squared_returns.rolling(window=week_window).mean()
        monthly_rv = squared_returns.rolling(window=month_window).mean()

        # Annualize all components
        annualization_factor = get_annualization_factor(frequency)
        period_rv *= annualization_factor
        weekly_rv *= annualization_factor
        monthly_rv *= annualization_factor

        return {
            'period_rv': period_rv,
            'weekly_rv': weekly_rv,
            'monthly_rv': monthly_rv
        }
```

## 3. HAR Model Implementation

```python
import statsmodels.api as sm
from typing import Dict, Any, Optional, List

class HARModel:
    """Heterogeneous Autoregressive (HAR) model."""

    # Default lags by frequency (from methodology paper)
    DEFAULT_LAGS = {
        'D': {'period': 1, 'week': 5, 'month': 22},
        'W': {'period': 1, 'week': 4, 'month': 12},
        'M': {'period': 1, 'week': 3, 'month': 12}
    }

    def __init__(self,
                 min_periods: int = 30,
                 frequency: str = 'D',
                 window: int = 21,
                 lags: Optional[Dict[str, int]] = None):
        """
        Initialize HAR model.

        Args:
            min_periods: Minimum periods required for fitting
            frequency: Return series frequency ('D', 'W', 'M')
            window: Size of realized volatility calculation window
            lags: Optional custom lags for components
                  Format: {'period': 1, 'week': 5, 'month': 22}
        """
        self.min_periods = min_periods
        self.frequency = frequency
        self.window = window
        self.lags = lags or self.DEFAULT_LAGS[frequency]
        self._coefficients = None
        self._residuals = None
        self._r_squared = None

    def _get_forecast_dates(self,
                            last_date: pd.Timestamp,
                            n_ahead: int) -> List[pd.Timestamp]:
        """
        Get proper forecast dates based on frequency and window.

        Args:
            last_date: Last date in input series
            n_ahead: Number of periods ahead to forecast

        Returns:
            List of forecast dates (end of each forecast window)
        """
        forecast_dates = []
        current_date = last_date

        for i in range(n_ahead):
            # Add business days for window
            forecast_date = pd.date_range(
                start=current_date,
                periods=self.window,
                freq='B'  # business day frequency
            )[-1]

            forecast_dates.append(forecast_date)
            current_date = forecast_date + pd.Timedelta(days=1)

        return forecast_dates

    def fit(self, rv_components: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Fit HAR model using components.

        Args:
            rv_components: Dictionary with period, weekly, monthly RV

        Returns:
            Dictionary with model results
        """
        # Validate inputs
        required_keys = {'period_rv', 'weekly_rv', 'monthly_rv'}
        if not all(k in rv_components for k in required_keys):
            raise ValidationError(f"rv_components must contain: {required_keys}")

        # Create lagged predictors using configured lags
        X = pd.DataFrame({
            'period_rv': rv_components['period_rv'].shift(self.lags['period']),
            'weekly_rv': rv_components['weekly_rv'].shift(self.lags['week']),
            'monthly_rv': rv_components['monthly_rv'].shift(self.lags['month'])
        }).dropna()

        if len(X) < self.min_periods:
            raise ValidationError(
                f"Insufficient data after lag creation: {len(X)} < {self.min_periods}"
            )

        # Target variable
        y = rv_components['period_rv'].loc[X.index]

        # Fit model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        # Store results
        self._coefficients = model.params
        self._residuals = model.resid
        self._r_squared = model.rsquared

        return {
            'coefficients': model.params,
            'r_squared': model.rsquared,
            't_stats': model.tvalues,
            'p_values': model.pvalues,
            'residuals': model.resid
        }

    def forecast(self,
                 rv_components: Dict[str, pd.Series],
                 n_ahead: int = 1) -> pd.Series:
        """
        Generate volatility forecasts for future windows.

        Args:
            rv_components: Dictionary with period, weekly, monthly RV
            n_ahead: Number of complete future windows to forecast

        Returns:
            Series of volatility forecasts indexed by forecast dates
        """
        if self._coefficients is None:
            raise RuntimeError("Model must be fit before forecasting")

        # Get last date in input series
        last_date = max(series.index[-1] for series in rv_components.values())

        # Generate forecast dates
        forecast_dates = self._get_forecast_dates(last_date, n_ahead)

        # Create prediction matrix
        X = pd.DataFrame({
            'period_rv': rv_components['period_rv'],
            'weekly_rv': rv_components['weekly_rv'],
            'monthly_rv': rv_components['monthly_rv']
        })

        # Add constant
        X = sm.add_constant(X)

        # Generate forecasts for each future window
        forecasts = pd.Series(
            np.dot(X, self._coefficients),
            index=forecast_dates
        )

        return forecasts
```

## 4. GARCH Model Implementation

```python
from arch import arch_model
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from ..core.validation import ValidationError, ProcessingError

class GARCHModel:
    """
    GARCH(1,1) model implementation for volatility forecasting.

    The model follows the specification:
    σ²(t) = ω + α⋅ε²(t-1) + β⋅σ²(t-1)

    where:
    - ω (omega): Long-run variance weight
    - α (alpha): News impact coefficient
    - β (beta): Volatility persistence coefficient
    """

    def __init__(self, p: int = 1, q: int = 1, frequency: str = 'D'):
        """
        Initialize GARCH model.

        Args:
            p: GARCH lag order (default 1)
            q: ARCH lag order (default 1)
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')
        """
        if p != 1 or q != 1:
            raise ValueError("Only GARCH(1,1) is currently supported")

        if frequency not in ANNUALIZATION_FACTORS:
            raise ValueError(
                f"Unsupported frequency: {frequency}. "
                f"Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
            )

        self.p = p
        self.q = q
        self.frequency = frequency
        self._annualization_factor = get_annualization_factor(frequency)
        self._model = None
        self._result = None

    def fit(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH model to return series.

        Args:
            returns: Return series

        Returns:
            Dict containing:
            - parameters: Model parameters (omega, alpha, beta)
            - persistence: Alpha + beta
            - long_term_vol: Annualized long-term volatility
            - half_life: Volatility half-life
            - log_likelihood: Model log-likelihood
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
        """
        # Create and fit model
        self._model = arch_model(
            returns,
            vol='Garch',
            p=self.p,
            q=self.q
        )

        try:
            self._result = self._model.fit(disp='off')
        except Exception as e:
            raise ProcessingError(f"GARCH fitting failed: {str(e)}")

        # Extract parameters
        omega = self._result.params['omega']
        alpha = self._result.params['alpha[1]']
        beta = self._result.params['beta[1]']
        persistence = alpha + beta

        # Calculate derived quantities
        long_term_var = omega / (1 - persistence)
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

        return {
            'parameters': {
                'omega': omega,
                'alpha': alpha,
                'beta': beta
            },
            'persistence': persistence,
            'long_term_vol': np.sqrt(long_term_var) * self._annualization_factor,
            'half_life': half_life,
            'log_likelihood': self._result.loglikelihood,
            'aic': self._result.aic,
            'bic': self._result.bic
        }

    def _forecast_instantaneous(self,
                              n_periods: int,
                              start_var: Optional[float] = None
                              ) -> np.ndarray:
        """
        Generate instantaneous volatility forecasts.

        Args:
            n_periods: Number of periods to forecast
            start_var: Starting variance (if None, uses last fitted value)

        Returns:
            Array of variance forecasts
        """
        if self._result is None:
            raise RuntimeError("Model must be fit before forecasting")

        # Get parameters
        omega = self._result.params['omega']
        alpha = self._result.params['alpha[1]']
        beta = self._result.params['beta[1]']
        persistence = alpha + beta
        long_term_var = omega / (1 - persistence)

        # Get starting variance
        if start_var is None:
            start_var = self._result.conditional_volatility[-1]**2

        # Generate forecasts
        forecasts = np.zeros(n_periods)
        current_var = start_var

        for t in range(n_periods):
            # Formula: σ²(t+n) = ω + (α + β)^(n-1) * (σ²(t+1) - ω/(1-α-β))
            forecasts[t] = (
                long_term_var +
                persistence**t * (current_var - long_term_var)
            )

        return forecasts

    def forecast_rolling_volatility(self,
                                   n_periods: int,
                                   window_size: int
                                   ) -> pd.Series:
          """
          Forecast rolling window volatility using blended historical and forecast values.

          Args:
              n_periods: Number of periods ahead to forecast
              window_size: Size of forecast window

          Returns:
              Series of annualized rolling window volatility forecasts
          """
          if self._result is None:
              raise RuntimeError("Model must be fit before forecasting")

          # Get historical instantaneous volatilities from fitted model
          hist_vols = self._result.conditional_volatility[-window_size+1:]

          # Generate instantaneous variance forecasts
          var_forecasts = self._forecast_instantaneous(
              n_periods=n_periods,
              start_var=self._result.conditional_volatility[-1]**2
          )

          # Convert to volatilities
          vol_forecasts = np.sqrt(var_forecasts)

          # Create dates for forecasts
          start_date = self._result.data.index[-1]

          forecast_dates = pd.date_range(
              start=start_date + pd.Timedelta(days=1),
              periods=n_periods,
              freq='B'
          )

          # Combine historical and forecast volatilities
          all_vols = pd.concat([
              hist_vols,
              pd.Series(vol_forecasts, index=forecast_dates)
          ])

          # Calculate rolling window volatility and annualize
          rolling_vol = np.sqrt(
              all_vols.pow(2)  # Convert to variances
              .rolling(window=window_size)  # Take rolling window
              .mean()  # Average the variances
          ) * self._annualization_factor

          # Return only the forecast period
          return rolling_vol[forecast_dates]

    @property
    def long_term_volatility(self) -> float:
        """
        Calculate annualized long-term volatility from model parameters.

        Returns:
            Annualized long-term volatility
        """
        if self._result is None:
            raise RuntimeError("Model must be fit first")

        omega = self._result.params['omega']
        alpha = self._result.params['alpha[1]']
        beta = self._result.params['beta[1]']

        return np.sqrt(omega / (1 - alpha - beta)) * self._annualization_factor
```

## 5. Validation and Error Handling

```python
class VolatilityModelError(Exception):
    """Base class for volatility model errors."""
    pass

def validate_returns(returns: pd.Series) -> None:
    """Validate return series for volatility calculations."""
    if not isinstance(returns, pd.Series):
        raise ValidationError("Returns must be pandas Series")

    if returns.empty:
        raise ValidationError("Return series is empty")

    if not np.isfinite(returns).all():
        raise ValidationError("Return series contains invalid values")

def validate_model_parameters(p: int, q: int) -> None:
    """Validate GARCH model parameters."""
    if not all(isinstance(x, int) and x > 0 for x in [p, q]):
        raise ValidationError("p and q must be positive integers")
```

## 6. Usage Examples

```python
# Calculate realized volatility
rv = RealizedVolatility()
daily_vol = rv.realized_volatility(returns, frequency='D')
weekly_vol = rv.realized_volatility(weekly_returns, frequency='W')
monthly_vol = rv.realized_volatility(monthly_returns, frequency='M')

# Calculate rolling volatilities
rolling_daily_vol = rv.rolling_volatility(returns, window=21, frequency='D')
rolling_weekly_vol = rv.rolling_volatility(weekly_returns, window=13, frequency='W')
rolling_monthly_vol = rv.rolling_volatility(monthly_returns, window=6, frequency='M')

# Calculate HAR components for different frequencies
daily_components = rv.har_components(returns, frequency='D')
weekly_components = rv.har_components(weekly_returns, frequency='W')
monthly_components = rv.har_components(monthly_returns, frequency='M')

# Fit HAR models
daily_har = HARModel(frequency='D')
daily_results = daily_har.fit(daily_components)
daily_forecasts = daily_har.forecast(daily_components, n_ahead=3)

weekly_har = HARModel(frequency='W')
weekly_results = weekly_har.fit(weekly_components)
weekly_forecasts = weekly_har.forecast(weekly_components, n_ahead=2)

# Fit GARCH model
garch = GARCHModel(p=1, q=1, frequency='D')
garch_results = garch.fit(returns)

# Get annualized long-term volatility estimate
long_term_vol = garch.long_term_volatility

# Forecast 120 days of annualized rolling 21-day volatility
rolling_vol_forecasts = garch.forecast_rolling_volatility(
    n_periods=120,
    window_size=21
)

# Forecast with historical volatilities for blending
rolling_vol_forecasts_with_hist = garch.forecast_rolling_volatility(
    n_periods=120,
    window_size=21,
)
```

For detailed methodology, see [VolatilityForecasting_HAR_GARCH.md](../../references/methodologies/VolatilityForecasting_HAR_GARCH.md)
