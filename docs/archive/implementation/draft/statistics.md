# Statistics Module Implementation

## 1. Basic Statistics

```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from ..core import ANNUALIZATION_FACTORS, get_annualization_factor

class ReturnStatistics:
    """Calculate return series statistics."""

    @staticmethod
    def mean_return(returns: pd.Series) -> float:
        """Calculate arithmetic mean return."""
        return returns.mean()

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """Calculate total compound return over the series."""
        return (1 + returns).prod() - 1

    @staticmethod
    def stdev(returns: pd.Series) -> float:
        """Calculate return series standard deviation."""
        return returns.std()

    @staticmethod
    def skewness(returns: pd.Series) -> float:
        """Calculate return series skewness."""
        return returns.skew()

    @staticmethod
    def kurtosis(returns: pd.Series) -> float:
        """Calculate return series excess kurtosis."""
        return returns.kurtosis()

    @staticmethod
    def annualized_return(returns: pd.Series,
                           frequency: str = 'D') -> float:
        """
        Calculate annualized return.

        Args:
            returns: Return series
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

        Returns:
            Annualized return
        """
        mean_ret = ReturnStatistics.mean_return(returns)
        ann_factor = ANNUALIZATION_FACTORS[frequency]  # No sqrt for returns
        return (1 + mean_ret) ** ann_factor - 1

    @staticmethod
    def volatility(returns: pd.Series,
                  frequency: str = 'D') -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Return series
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

        Returns:
            Annualized volatility
        """
        return returns.std() * get_annualization_factor(frequency)

    @staticmethod
    def sharpe_ratio(returns: pd.Series,
                    rf_rate: float = 0.0,
                    frequency: str = 'D',
                    method: str = 'standard') -> float:
        """
        Calculate annualized Sharpe ratio variants.

        Args:
            returns: Return series
            rf_rate: Risk-free rate (annualized)
            frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')
            method: One of ['standard', 'deflated', 'probabilistic',
                          'geometric', 'adjusted_geometric']

        Returns:
            Annualized Sharpe ratio
        """
        if method not in ['standard', 'deflated', 'probabilistic',
                         'geometric', 'adjusted_geometric']:
            raise ValueError(f"Unsupported Sharpe ratio method: {method}")

        # Convert annual rf_rate to period rate
        period_factor = 1 / ANNUALIZATION_FACTORS[frequency]
        period_rf = (1 + rf_rate) ** period_factor - 1

        # Calculate period excess returns
        excess_returns = returns - period_rf

        if method == 'standard':
            sr = excess_returns.mean() / excess_returns.std()

        elif method == 'deflated':
            n = len(returns)
            sr_standard = excess_returns.mean() / excess_returns.std()
            deflator = 1 - (0.5 * ((sr_standard ** 2) / (n - 1)))
            sr = sr_standard * deflator

        elif method == 'probabilistic':
            sr_standard = excess_returns.mean() / excess_returns.std()
            n = len(returns)
            skew = returns.skew()
            kurt = returns.kurtosis()
            sr = ReturnStatistics._probabilistic_sr(
                sr_standard, n, skew, kurt
            )

        elif method == 'geometric':
            geo_mean = (1 + excess_returns).prod() ** (1/len(excess_returns)) - 1
            sr = geo_mean / excess_returns.std()

        else:  # adjusted_geometric
            geo_mean = (1 + excess_returns).prod() ** (1/len(excess_returns)) - 1
            adj_std = ReturnStatistics._calculate_adjusted_std(returns)
            sr = geo_mean / adj_std

        # Annualize the Sharpe ratio
        return sr * np.sqrt(ANNUALIZATION_FACTORS[frequency])

    @staticmethod
    def _probabilistic_sr(sr: float,
                         n: int,
                         skew: float,
                         kurt: float) -> float:
        """Calculate probabilistic Sharpe ratio adjustment."""
        var_term = (1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
        if var_term <= 0:
            raise ValueError("Invalid combination of SR, skewness and kurtosis")
        return stats.norm.cdf((np.sqrt(n - 1) * sr) / np.sqrt(var_term))

    @staticmethod
    def _calculate_adjusted_std(returns: pd.Series) -> float:
        """Calculate adjusted standard deviation for geometric Sharpe ratio."""
        # Implementation of adjusted std calculation
        pass

class CorrelationCalculator:
    """Calculate correlations using various methods."""

    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame,
                                   method: str = 'pearson',
                                   min_periods: Optional[int] = None
                                   ) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            returns: DataFrame of return series
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum number of overlapping periods required

        Returns:
            Correlation matrix
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(f"Unsupported correlation method: {method}")

        return returns.corr(method=method, min_periods=min_periods)

    @staticmethod
    def rolling_correlation(series1: pd.Series,
                          series2: pd.Series,
                          window: int = 63,
                          min_periods: Optional[int] = None,
                          method: str = 'pearson'
                          ) -> pd.Series:
        """Calculate rolling correlation between two series."""
        df = pd.DataFrame({'x': series1, 'y': series2})
        return df['x'].rolling(
            window=window,
            min_periods=min_periods
        ).corr(df['y'], method=method)

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series,
                                rf_rate: float = 0.0,
                                frequency: str = 'D') -> float:
        """Calculate annualized Sharpe ratio."""
        ann_factor = get_annualization_factor(frequency)
        excess_returns = returns - rf_rate
        return (excess_returns.mean() * ann_factor) / (returns.std() * ann_factor)
```

## 2. Distribution Analysis

```python
class DistributionAnalyzer(ValidationMixin):
    """Analyze return distributions."""

    def fit_distribution(self,
                        returns: pd.Series,
                        dist_type: str = 'skewed-t'
                        ) -> Dict[str, Any]:
        """Fit distribution to returns."""
        # Validate inputs
        self.validate_returns(returns)
        if dist_type not in ['normal', 'skewed-t', 'stable']:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        if dist_type == 'normal':
            return self._fit_normal(returns)
        elif dist_type == 'skewed-t':
            return self._fit_skewed_t(returns)
        elif dist_type == 'stable':
            return self._fit_stable(returns)

    def _fit_normal(self, returns: pd.Series) -> Dict[str, Any]:
        """Fit normal distribution."""
        self.validate_returns(returns)
        mu, sigma = norm.fit(returns)

        # Calculate goodness of fit
        _, p_value = normaltest(returns)

        return {
            'parameters': {'mu': mu, 'sigma': sigma},
            'p_value': p_value,
            'distribution': 'normal'
        }

    def _fit_skewed_t(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit skewed Student's t distribution.

        Uses MLE to fit parameters:
        - location (μ)
        - scale (σ)
        - shape (α)
        - degrees of freedom (ν)
        """
        from scipy.stats import t
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            mu, sigma, alpha, nu = params
            y = (returns - mu) / sigma
            w = alpha * y

            # Log-likelihood for skewed-t
            ll = np.sum(
                t.logpdf(y, nu) +
                t.logcdf(w, nu + 1) +
                np.log(2) - np.log(sigma)
            )
            return -ll

        # Initial guess based on moments
        init_params = [
            returns.mean(),  # mu
            returns.std(),   # sigma
            returns.skew(),  # alpha
            6.0             # nu (degrees of freedom)
        ]

        # Bounds for parameters
        bounds = [
            (None, None),  # mu
            (1e-6, None),  # sigma > 0
            (None, None),  # alpha
            (2.1, 100.0)   # nu > 2 for finite variance
        ]

        # Fit distribution
        result = minimize(
            neg_log_likelihood,
            init_params,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if not result.success:
            warnings.warn("Skewed-t fitting did not converge")

        mu, sigma, alpha, nu = result.x

        # Calculate goodness of fit
        _, p_value = normaltest(returns)  # Basic normality test

        return {
            'parameters': {
                'location': mu,
                'scale': sigma,
                'shape': alpha,
                'df': nu
            },
            'p_value': p_value,
            'distribution': 'skewed-t',
            'convergence': result.success,
            'log_likelihood': -result.fun
        }

    def _fit_stable(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit stable distribution.

        Parameters:
        - α (stability parameter) ∈ (0,2]
        - β (skewness) ∈ [-1,1]
        - γ (scale) > 0
        - δ (location)
        """
        from scipy.stats import levy_stable

        # Fit stable distribution
        alpha, beta, loc, scale = levy_stable.fit(returns)

        # Calculate goodness of fit using KS test
        ks_stat, p_value = levy_stable.kstest(
            returns,
            'levy_stable',
            args=(alpha, beta, loc, scale)
        )

        return {
            'parameters': {
                'alpha': alpha,  # stability
                'beta': beta,    # skewness
                'location': loc,
                'scale': scale
            },
            'p_value': p_value,
            'ks_statistic': ks_stat,
            'distribution': 'stable'
        }
```

## 3. Risk Metrics

```python
class RiskMetrics:
    """Calculate various risk metrics."""

    @staticmethod
    def calculate_var(returns: pd.Series,
                     confidence: float = 0.95,
                     method: str = 'historical'
                     ) -> float:
        """Calculate Value at Risk."""
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            z_score = norm.ppf(confidence)
            return -(returns.mean() + z_score * returns.std())
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    @staticmethod
    def calculate_cvar(returns: pd.Series,
                      confidence: float = 0.95,
                      method: str = 'historical'
                      ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = RiskMetrics.calculate_var(returns, confidence, method)
        return -returns[returns <= -var].mean()

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series,
                             rf_rate: float = 0.0,
                             periods: int = 252,
                             modified: bool = True
                             ) -> float:
        """
        Calculate Sharpe Ratio.

        If modified=True, adjusts for skewness and kurtosis.
        """
        excess_returns = returns - rf_rate
        if not modified:
            return np.sqrt(periods) * (excess_returns.mean() / excess_returns.std())

        # Modified Sharpe Ratio
        sr = np.sqrt(periods) * (excess_returns.mean() / excess_returns.std())
        skew = excess_returns.skew()
        kurt = excess_returns.kurtosis()

        modified_sr = sr * (1 + (skew/6) * sr - ((kurt-3)/24) * sr**2)
        return modified_sr

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               rf_rate: float = 0.0,
                               periods: int = 252) -> float:
        """
        Calculate Sortino Ratio using downside deviation.

        Args:
            returns: Return series
            rf_rate: Risk-free rate
            periods: Number of periods per year
        """
        excess_returns = returns - rf_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))

        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf

        return np.sqrt(periods) * excess_returns.mean() / downside_std

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series,
                               rf_rate: float = 0.0,
                               periods: int = 252) -> float:
        """
        Calculate Calmar Ratio (annualized return / max drawdown).
        """
        excess_returns = returns - rf_rate
        ann_return = excess_returns.mean() * periods
        max_dd = ReturnStatistics.calculate_drawdown(returns)['max_drawdown']

        if max_dd == 0:
            return np.inf if ann_return > 0 else -np.inf

        return -ann_return / max_dd

    @staticmethod
    def calculate_omega_ratio(returns: pd.Series,
                              threshold: float = 0.0,
                              periods: int = 252) -> float:
        """
        Calculate Omega Ratio.

        Ratio of probability-weighted gains vs losses relative to threshold.
        """
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return np.inf if gains > 0 else 0

        return gains / losses
```

## 4. Regression Analysis

```python
class RegressionAnalysis:
    """Statistical regression analysis for backfill."""

    @staticmethod
    def fit_ols(y: pd.Series,
                X: pd.DataFrame,
                min_periods: int = 30) -> Dict[str, Any]:
        """
        Fit OLS regression for backfill.

        Args:
            y: Target return series
            X: Explanatory return series
            min_periods: Minimum overlapping periods required

        Returns:
            Dictionary containing:
            - coefficients
            - residuals
            - r_squared
            - t_stats
            - p_values
        """
        # Ensure sufficient overlap
        common_idx = y.index.intersection(X.index)
        if len(common_idx) < min_periods:
            raise ValueError(f"Insufficient overlapping periods: {len(common_idx)} < {min_periods}")

        # Align data
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Add constant
        X = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X).fit()

        return {
            'coefficients': model.params,
            'residuals': model.resid,
            'r_squared': model.rsquared,
            't_stats': model.tvalues,
            'p_values': model.pvalues,
            'std_errors': model.bse
        }
```

## 5. Volatility Models

```python
class VolatilityModels(ValidationMixin):
    """Volatility model implementations."""

    @staticmethod
    def realized_volatility(returns: pd.Series,
                           annualize: bool = True,
                           trading_days: int = 252) -> float:
        """Calculate realized volatility."""
        VolatilityModels.validate_returns(returns)
        if not isinstance(trading_days, int) or trading_days <= 0:
            raise ValueError("trading_days must be positive integer")

        vol = np.sqrt(np.sum(returns**2))
        if annualize:
            vol *= np.sqrt(trading_days)
        return vol

    @staticmethod
    def har_components(returns: pd.Series,
                      daily_lags: int = 1,
                      weekly_lags: int = 5,
                      monthly_lags: int = 22) -> Dict[str, pd.Series]:
        """Calculate HAR model components."""
        VolatilityModels.validate_returns(returns)
        for name, lag in [('daily_lags', daily_lags),
                         ('weekly_lags', weekly_lags),
                         ('monthly_lags', monthly_lags)]:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError(f"{name} must be positive integer")

        if len(returns) < monthly_lags:
            raise ValueError(f"Insufficient data: {len(returns)} < {monthly_lags}")

        # Calculate daily RV
        daily_rv = returns**2

        # Calculate weekly RV (5-day average)
        weekly_rv = daily_rv.rolling(window=weekly_lags).mean()

        # Calculate monthly RV (22-day average)
        monthly_rv = daily_rv.rolling(window=monthly_lags).mean()

        return {
            'daily_rv': daily_rv,
            'weekly_rv': weekly_rv,
            'monthly_rv': monthly_rv
        }

    @staticmethod
    def fit_har(rv_components: Dict[str, pd.Series],
                min_periods: int = 30) -> Dict[str, Any]:
        """Fit HAR model using components."""
        # Validate inputs
        if not isinstance(rv_components, dict):
            raise TypeError("rv_components must be dictionary")
        required_keys = {'daily_rv', 'weekly_rv', 'monthly_rv'}
        if not all(k in rv_components for k in required_keys):
            raise ValueError(f"rv_components must contain: {required_keys}")

        if not isinstance(min_periods, int) or min_periods <= 0:
            raise ValueError("min_periods must be positive integer")

        # Create lagged predictors
        X = pd.DataFrame({
            'daily_rv': rv_components['daily_rv'].shift(1),
            'weekly_rv': rv_components['weekly_rv'].shift(1),
            'monthly_rv': rv_components['monthly_rv'].shift(1)
        }).dropna()

        if len(X) < min_periods:
            raise ValueError(f"Insufficient data after lag creation: {len(X)} < {min_periods}")

        # Target variable
        y = rv_components['daily_rv'].loc[X.index]

        # Fit model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        return {
            'coefficients': model.params,
            'r_squared': model.rsquared,
            't_stats': model.tvalues,
            'p_values': model.pvalues,
            'residuals': model.resid
        }

    @staticmethod
    def fit_garch(returns: pd.Series,
                  p: int = 1,
                  q: int = 1) -> Dict[str, Any]:
        """Fit GARCH(p,q) model."""
        VolatilityModels.validate_returns(returns)
        if not all(isinstance(x, int) and x > 0 for x in [p, q]):
            raise ValueError("p and q must be positive integers")

        if len(returns) < (p + q + 1) * 10:  # Rule of thumb for minimum sample size
            raise ValueError(f"Insufficient data for GARCH({p},{q})")

        model = arch_model(returns, vol='Garch', p=p, q=q)
        try:
            result = model.fit(disp='off')
        except Exception as e:
            raise ProcessingError(f"GARCH fitting failed: {str(e)}")

        return {
            'parameters': result.params,
            'conditional_volatility': result.conditional_volatility,
            'residuals': result.resid,
            'log_likelihood': result.loglikelihood,
            'aic': result.aic,
            'bic': result.bic
        }
```

## 6. Return Adjustments

```python
class ValidationMixin:
    """Validation methods for statistical calculations."""

    @staticmethod
    def validate_volatility(volatility: float) -> None:
        """Validate volatility input."""
        if not isinstance(volatility, (int, float)):
            raise TypeError("Volatility must be numeric")
        if volatility < 0:
            raise ValueError("Volatility cannot be negative")

    @staticmethod
    def validate_returns(returns: pd.Series) -> None:
        """Validate return series."""
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be pandas Series")
        if returns.empty:
            raise ValueError("Return series is empty")
        if not np.isfinite(returns).all():
            raise ValueError("Return series contains invalid values")

    @staticmethod
    def validate_degrees_of_freedom(df: float) -> None:
        """Validate degrees of freedom for Student-t."""
        if not isinstance(df, (int, float)):
            raise TypeError("Degrees of freedom must be numeric")
        if df <= 2:
            raise ValueError("Degrees of freedom must be > 2 for finite variance")

class ReturnAdjustments(ValidationMixin):
    """Arithmetic to geometric return conversions and adjustments."""

    @staticmethod
    def variance_drag(volatility: float) -> float:
        """Calculate variance drag from volatility."""
        ReturnAdjustments.validate_volatility(volatility)
        return (volatility ** 2) / 2

    @staticmethod
    def kurtosis_drag(excess_kurtosis: float, volatility: float) -> float:
        """Calculate kurtosis drag."""
        return (excess_kurtosis * volatility ** 4) / 24

    @staticmethod
    def skew_drag(skewness: float, volatility: float) -> float:
        """Calculate skewness drag."""
        return (skewness * volatility ** 3) / 6

    @staticmethod
    def geometric_return(arithmetic_return: float,
                        volatility: float,
                        skewness: float = 0.0,
                        excess_kurtosis: float = 0.0) -> float:
        """Convert arithmetic to geometric return with full adjustments."""
        ReturnAdjustments.validate_volatility(volatility)
        if not all(isinstance(x, (int, float)) for x in [arithmetic_return, skewness, excess_kurtosis]):
            raise TypeError("All inputs must be numeric")

        geo_return = arithmetic_return - ReturnAdjustments.variance_drag(volatility)

        if skewness != 0:
            geo_return -= ReturnAdjustments.skew_drag(skewness, volatility)
        if excess_kurtosis != 0:
            geo_return -= ReturnAdjustments.kurtosis_drag(excess_kurtosis, volatility)

        return geo_return

    @staticmethod
    def adjusted_volatility(volatility: float,
                           excess_kurtosis: float,
                           skewness: float) -> float:
        """
        Calculate adjusted volatility accounting for higher moments.

        Formula from background note:
        σ_adj = σ * sqrt(1 + (γ₄-3)σ²/4 + γ₃²σ²/6)
        """
        kurtosis_term = (excess_kurtosis * volatility ** 2) / 4
        skewness_term = (skewness ** 2 * volatility ** 2) / 6
        return volatility * np.sqrt(1 + kurtosis_term + skewness_term)

    @staticmethod
    def student_t_volatility(volatility: float, df: float) -> float:
        """
        Calculate volatility under Student's t distribution.

        Formula: σ_t = σ * sqrt(ν/(ν-2))
        where ν is degrees of freedom
        """
        return volatility * np.sqrt(df / (df - 2))

    @staticmethod
    def skew_adjusted_student_t_volatility(volatility: float,
                                         df: float,
                                         skewness: float,
                                         market_impact: float) -> float:
        """
        Calculate skew-adjusted Student's t volatility.

        Formula: σ_t_adj = σ_t * (1 + γ₃λ)
        where λ is market impact factor
        """
        student_t_vol = ReturnAdjustments.student_t_volatility(volatility, df)
        return student_t_vol * (1 + skewness * market_impact)

    @staticmethod
    def kelly_fraction_normal(return_value: float,
                             volatility: float,
                             rf_rate: float = 0.0) -> float:
        """
        Calculate Kelly fraction under normal distribution.

        Basic Kelly fraction equals the Sharpe ratio.
        """
        return (return_value - rf_rate) / volatility

    @staticmethod
    def kelly_fraction_student_t(arithmetic_return: float,
                                 volatility: float,
                                 excess_kurtosis: float,
                                 skewness: float,
                                 df: float,
                                 market_impact: float,
                                 rf_rate: float = 0.0) -> float:
        """
        Calculate Kelly fraction under Student-t distribution.

        Formula: f*_t = f*_g,adj × (ν-2)/(ν+1) × (1 - γ₃λ)
        where:
        - ν is degrees of freedom
        - γ₃ is skewness
        - λ is market impact factor
        """
        # Calculate adjusted geometric Kelly fraction
        adj_geo_return = ReturnAdjustments.geometric_return(
            arithmetic_return, volatility, skewness, excess_kurtosis
        )
        adj_vol = ReturnAdjustments.adjusted_volatility(
            volatility, excess_kurtosis, skewness
        )
        f_adj = (adj_geo_return - rf_rate) / adj_vol

        # Apply Student-t adjustment
        student_t_adj = (df - 2) / (df + 1)

        # Apply skewness adjustment
        skew_adj = 1 - skewness * market_impact

        return f_adj * student_t_adj * skew_adj
```

## 7. Performance Metrics

```python
class PerformanceMetrics(ValidationMixin):
    """Performance metrics with adjustments for higher moments."""

    @staticmethod
    def probabilistic_sharpe_ratio(observed_sr: float,
                                  benchmark_sr: float,
                                  n_samples: int,
                                  skewness: float,
                                  kurtosis: float) -> float:
        """Calculate Probabilistic Sharpe Ratio."""
        # Validate inputs
        if not isinstance(n_samples, int) or n_samples <= 1:
            raise ValueError("n_samples must be integer > 1")
        if not all(isinstance(x, (int, float)) for x in [observed_sr, benchmark_sr, skewness, kurtosis]):
            raise TypeError("Statistical inputs must be numeric")

        # Calculate PSR components
        n_term = np.sqrt(n_samples - 1)
        sr_diff = observed_sr - benchmark_sr

        # Check denominator
        var_term = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr ** 2
        if var_term <= 0:
            raise ValueError("Invalid combination of SR, skewness and kurtosis")

        return norm.cdf((n_term * sr_diff) / np.sqrt(var_term))
```
