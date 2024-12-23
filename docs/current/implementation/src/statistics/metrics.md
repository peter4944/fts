# Statistical Metrics Module

## Overview
Provides statistical calculations and adjustments for financial time series.

## Implementation

```python
from typing import Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum
import warnings

from fts.core.validation import ValidationRules
from fts.core.constants import ANNUALIZATION_FACTORS
from fts.core.base import convert_returns
from fts.core.transformations.returns import ReturnTransformations

class ReturnType(Enum):
    ARITHMETIC = "arithmetic"
    LOG = "log"

class ReturnStatistics:
    """Calculate statistics for single or multiple return series."""

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        return_type: ReturnType = ReturnType.LOG,
        frequency: str = 'daily',
        annualized: bool = True
    ):
        """Initialize with return series."""
        self.frequency = frequency
        self.return_type = return_type
        self.annualized = annualized
        self._validate_inputs(returns)

        # Store data efficiently
        self._dtype = np.float32
        self._returns_array = returns.to_numpy(dtype=self._dtype)
        self._index = returns.index
        self._columns = returns.columns if isinstance(returns, pd.DataFrame) else None
        self._n_obs = len(returns)

        # Initialize cache
        self._cache = {}

        # Compute and cache basic statistics in one pass
        self._compute_basic_stats()

        # Convert and cache return types (vectorized)
        self._compute_return_types()

    def _validate_inputs(self, returns: Union[pd.Series, pd.DataFrame]) -> None:
        """Validate input data."""
        ValidationRules.validate_returns(returns)
        ValidationRules.validate_frequency(self.frequency)

    def _compute_basic_stats(self) -> None:
        """Compute all basic statistics in one pass."""
        mean = np.mean(self._returns_array, axis=0)
        diff = self._returns_array - mean
        var = np.mean(diff**2, axis=0)
        std = np.sqrt(var)
        skew = np.mean(diff**3, axis=0) / var**1.5
        kurt = np.mean(diff**4, axis=0) / var**2 - 3

        self._period_stats = {
            'mean': mean,
            'std': std,
            'var': var,
            'skew': skew,
            'kurt': kurt
        }

    def _compute_return_types(self) -> None:
        """Convert between log and arithmetic returns."""
        if self.return_type == ReturnType.LOG:
            self._log_array = self._returns_array
            self._arith_array = np.expm1(self._returns_array)  # faster than exp(x)-1
        else:
            self._arith_array = self._returns_array
            self._log_array = np.log1p(self._returns_array)  # faster than log(1+x)

    @property
    def annualized_stats(self) -> Dict[str, np.ndarray]:
        """Get cached annualized statistics."""
        if 'annualized' not in self._cache:
            ann_factor = ANNUALIZATION_FACTORS[self.frequency]

            # Compute annualized returns
            log_annual = self._period_stats['mean'] * ann_factor
            arith_annual = (1 + self._period_stats['mean']) ** ann_factor - 1

            # Compute annualized volatility
            vol_annual = self._period_stats['std'] * np.sqrt(ann_factor)

            self._cache['annualized'] = {
                'returns': {
                    'arithmetic': arith_annual,
                    'log': log_annual
                },
                'volatility': vol_annual
            }
        return self._cache['annualized']

    @property
    def drag_components(self) -> Dict[str, np.ndarray]:
        """Get cached drag calculations."""
        if 'drags' not in self._cache:
            vol = self.annualized_stats['volatility']

            # Compute all drags vectorized
            drags = {
                'variance_drag': -(vol ** 2) / 2,
                'kurtosis_drag': -(self._period_stats['kurt'] * vol ** 4) / 24,
                'skew_drag': -(self._period_stats['skew'] * vol ** 3) / 6
            }

            # Add total and observed drags
            drags['total_estimated_drag'] = sum(drags.values())
            drags['observed_drag'] = (
                self.geometric_return() -
                self.annualized_stats['returns']['arithmetic']
            )

            self._cache['drags'] = drags

        return self._cache['drags']

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._cache.clear()

    def to_pandas(self, data: np.ndarray) -> Union[pd.Series, pd.DataFrame]:
        """Convert numpy array back to pandas with original index/columns."""
        if self._columns is not None:
            return pd.DataFrame(data, index=self._index, columns=self._columns)
        return pd.Series(data, index=self._index)

    # Rest of the methods updated to use cached computations...

    # Statistical measures (per period)
    @property
    def period_mean(self) -> float:
        """Per period mean return."""
        return self._period_stats['mean']

    @property
    def period_std(self) -> float:
        """Per period standard deviation."""
        return self._period_stats['std']

    @property
    def skewness(self) -> float:
        """Return skewness."""
        return self._period_stats['skew']

    @property
    def excess_kurtosis(self) -> float:
        """Return excess kurtosis."""
        return self._period_stats['kurt']

    # Annualized financial metrics
    @property
    def annualized_return(self) -> Dict[str, Union[float, pd.Series]]:
        """Get annualized returns (vectorized)."""
        if not self.annualized:
            return {
                'arithmetic': self._arith_array.mean(),
                'log': self._log_array.mean()
            }

        ann_factor = ANNUALIZATION_FACTORS[self.frequency]

        # Vectorized calculations
        log_annual = self._log_array.mean() * ann_factor
        arith_annual = (1 + self._arith_array.mean()) ** ann_factor - 1

        return {
            'arithmetic': arith_annual,
            'log': log_annual
        }

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        if not self.annualized:
            return self._period_stats['std']
        return self._period_stats['std'] * np.sqrt(ANNUALIZATION_FACTORS[self.frequency])

    def geometric_return(
        self,
        include_higher_moments: bool = False
    ) -> float:
        """
        Calculate annualized geometric return.

        Args:
            include_higher_moments: Whether to include skew and kurtosis adjustments

        Returns:
            Annualized geometric return

        Notes:
            - For arithmetic returns: Includes variance drag
            - For log returns: Direct annualization
            - Higher moments only applicable for arithmetic returns
        """
        if self.return_type == ReturnType.LOG:
            if include_higher_moments:
                warnings.warn("Higher moment adjustments not applicable for log returns")
            return self.annualized_return

        # For arithmetic returns
        geo_return = self.annualized_return - (self.volatility ** 2) / 2

        if include_higher_moments:
            geo_return -= (self.excess_kurtosis * self.volatility ** 4) / 24
            geo_return -= (self.skewness * self.volatility ** 3) / 6

        return geo_return

    def adjusted_volatility(
        self,
        include_higher_moments: bool = False
    ) -> float:
        """
        Calculate adjusted volatility.

        Args:
            include_higher_moments: Whether to include higher moment adjustments

        Returns:
            Adjusted annualized volatility

        Notes:
            - Higher moment adjustments only applicable for arithmetic returns
        """
        if self.return_type == ReturnType.LOG and include_higher_moments:
            warnings.warn("Higher moment adjustments not applicable for log returns")
            return self.volatility

        vol = self.volatility
        if include_higher_moments:
            kurtosis_term = (self.excess_kurtosis * vol ** 2) / 4
            skewness_term = (self.skewness ** 2 * vol ** 2) / 6
            vol *= np.sqrt(1 + kurtosis_term + skewness_term)

        return vol

    def summary_statistics(self) -> Union[Dict[str, float], pd.DataFrame]:
        """
        Get comprehensive statistics.

        Returns:
            DataFrame of statistics for multiple series
            Dict of statistics for single series
        """
        stats = {
            'n_observations': self._n_obs,
            'period_mean': self._period_stats['mean'],
            'period_std': self._period_stats['std'],
            'skewness': self._period_stats['skew'],
            'excess_kurtosis': self._period_stats['kurt'],
            'annualized_return_arithmetic': self.annualized_return['arithmetic'],
            'annualized_return_log': self.annualized_return['log'],
            'volatility': self.volatility,
            'geometric_return': self.geometric_return()
        }

        # For DataFrame input, return DataFrame of stats
        if isinstance(self._returns_array, np.ndarray) and self._returns_array.ndim > 1:
            return pd.DataFrame(stats)
        # For Series input, return dict
        return stats

    def sharpe_ratio(
        self,
        risk_free_rate: float = 0.0,
        use_geometric: bool = True
    ) -> Union[float, pd.Series]:
        """Calculate Sharpe ratio (vectorized)."""
        if use_geometric:
            excess_return = self.geometric_return() - risk_free_rate
        else:
            excess_return = self.annualized_return['arithmetic'] - risk_free_rate

        return excess_return / self.volatility
```

## Usage Examples

### Basic Statistics
```python
# Calculate return statistics
stats = ReturnStatistics(returns, frequency='daily')
print(f"Annualized mean: {stats.mean:.2%}")
print(f"Annualized vol: {stats.volatility:.2%}")
print(f"Geometric mean: {stats.geometric_return():.2%}")

# Get full summary
summary = stats.summary_statistics()
```

### Adjusted Metrics
```python
# Get adjusted geometric return
adj_geo = stats.geometric_return(include_higher_moments=True)

# Get adjusted volatility
adj_vol = stats.adjusted_volatility(include_higher_moments=True)
```
