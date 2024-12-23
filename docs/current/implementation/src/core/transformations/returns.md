"""
Return series transformations and standardizations.

Implements:
1. Z-score standardization
2. Min-Max scaling
3. Rank-based transformations
4. Box-Cox transformation
5. Winsorization
6. Excess return standardization
7. Alpha calculation

Notes:
- All methods operate on multiple series (DataFrame)
- Uses alignment module for series alignment
- Vectorized operations throughout
- Proper handling of both log and arithmetic returns
"""


```python
from typing import Dict, Union, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from ..base import ReturnType
from ...data.alignment import align_series

class ReturnTransformations:
    """
    Transform and standardize return series.

    Methods handle:
    - Basic standardizations (z-score, min-max)
    - Rank transformations
    - Distribution transformations (Box-Cox)
    - Robust transformations (Winsorization)
    - Financial transformations (excess returns, alpha)
    """

    @staticmethod
    def zscore(
        returns: pd.DataFrame,
        cross_sectional: bool = False,
        demean: bool = True
    ) -> pd.DataFrame:
        """
        Z-score standardization.

        Args:
            returns: Return series (assets in columns)
            cross_sectional: If True, standardize across assets at each time point
                           If False, standardize each asset's time series
            demean: Whether to subtract mean

        Returns:
            Standardized returns

        Notes:
            - For cross_sectional=True: Standardizes returns across assets at each time point
            - For cross_sectional=False: Standardizes each asset's time series independently
            - If demean=False: Only divides by standard deviation
        """
        if cross_sectional:
            if demean:
                means = returns.mean(axis=1)
                demeaned = returns.sub(means, axis=0)
            else:
                demeaned = returns

            stds = returns.std(axis=1)
            stds = np.where(stds == 0, 1, stds)
            return demeaned.div(stds, axis=0)
        else:
            if demean:
                means = returns.mean()
                demeaned = returns.sub(means)
            else:
                demeaned = returns

            stds = returns.std()
            stds = np.where(stds == 0, 1, stds)
            return demeaned.div(stds)

    @staticmethod
    def minmax(
        returns: pd.DataFrame,
        feature_range: Tuple[float, float] = (0, 1),
        cross_sectional: bool = False
    ) -> pd.DataFrame:
        """
        Min-max scaling to [0,1] or custom range.

        Args:
            returns: Return series (assets in columns)
            feature_range: (min, max) range to scale to
            cross_sectional: If True, scale across assets at each time point
                            If False, scale each asset's time series

        Returns:
            Scaled returns in specified range

        Notes:
            - For cross_sectional=True: Scales returns across assets at each time point
            - For cross_sectional=False: Scales each asset's time series independently
            - Handles edge case where min == max by returning mid-point of feature_range
        """
        min_val, max_val = feature_range

        if cross_sectional:
            data_min = returns.min(axis=1)
            data_max = returns.max(axis=1)

            equal_vals = (data_max == data_min)
            if equal_vals.any():
                warnings.warn("Some time points have equal min/max values")
                mid_point = (min_val + max_val) / 2
                data_range = np.where(equal_vals, 1, data_max - data_min)
            else:
                data_range = data_max - data_min

            scaled = returns.sub(data_min, axis=0).div(data_range, axis=0)
            return scaled * (max_val - min_val) + min_val
        else:
            data_min = returns.min()
            data_max = returns.max()

            equal_vals = (data_max == data_min)
            if equal_vals.any():
                warnings.warn("Some assets have equal min/max values")
                mid_point = (min_val + max_val) / 2
                data_range = np.where(equal_vals, 1, data_max - data_min)
            else:
                data_range = data_max - data_min

            scaled = returns.sub(data_min).div(data_range)
            return scaled * (max_val - min_val) + min_val

    @staticmethod
    def rank_transform(
        returns: pd.DataFrame,
        method: str = 'average',
        gaussian: bool = False,
        cross_sectional: bool = True
    ) -> pd.DataFrame:
        """
        Rank-based transformation.

        Args:
            returns: Return series (assets in columns)
            method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
            gaussian: If True, apply inverse normal to ranks
            cross_sectional: If True, rank across assets at each time point
                            If False, rank each asset's time series

        Returns:
            Rank transformed returns

        Notes:
            - For cross_sectional=True: Ranks returns across assets at each time point
            - For cross_sectional=False: Ranks each asset's time series independently
            - Gaussian transformation uses inverse normal CDF
        """
        if cross_sectional:
            ranked = returns.rank(axis=1, method=method)
            if gaussian:
                n_assets = returns.shape[1]
                scaled_ranks = (ranked - 0.5) / n_assets
                return pd.DataFrame(
                    stats.norm.ppf(scaled_ranks),
                    index=returns.index,
                    columns=returns.columns
                )
            return ranked
        else:
            ranked = returns.rank(method=method)
            if gaussian:
                n_obs = returns.shape[0]
                scaled_ranks = (ranked - 0.5) / n_obs
                return pd.DataFrame(
                    stats.norm.ppf(scaled_ranks),
                    index=returns.index,
                    columns=returns.columns
                )
            return ranked

    @staticmethod
    def box_cox(
        returns: pd.DataFrame,
        shift: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Box-Cox transformation.

        Args:
            returns: Return series (assets in columns)
            shift: Value to add to make returns positive
                   If None, automatically determined

        Returns:
            Tuple of:
            - Transformed returns
            - Dict of lambda parameters for each series

        Notes:
            - Automatically shifts data to be positive if needed
            - Handles each series independently
            - Returns lambda parameters for potential inverse transform
            - Warns if transformation may not be appropriate
        """
        if shift is None:
            min_vals = returns.min()
            shift = max(0, -(min_vals.min()) + 1e-6)

        if shift > 0:
            warnings.warn(f"Shifting data by {shift} to ensure positivity")

        shifted_returns = returns + shift

        if (shifted_returns <= 0).any().any():
            raise ValueError("Data must be strictly positive for Box-Cox transform")

        transformed = pd.DataFrame(index=returns.index, columns=returns.columns)
        lambdas = {}

        for col in returns.columns:
            try:
                transformed_data, lambda_param = stats.boxcox(shifted_returns[col])
                transformed[col] = transformed_data
                lambdas[col] = lambda_param
            except Exception as e:
                warnings.warn(f"Failed to transform {col}: {str(e)}")
                transformed[col] = np.log(shifted_returns[col])
                lambdas[col] = 0

        return transformed, lambdas

    @staticmethod
    def winsorize(
        returns: pd.DataFrame,
        limits: Tuple[float, float] = (0.05, 0.05),
        cross_sectional: bool = True
    ) -> pd.DataFrame:
        """
        Winsorize extreme values.

        Args:
            returns: Return series (assets in columns)
            limits: (lower, upper) percentiles to winsorize
            cross_sectional: If True, winsorize across assets at each time point
                            If False, winsorize each asset's time series

        Returns:
            Winsorized returns

        Notes:
            - For cross_sectional=True: Winsorizes each time point independently
            - For cross_sectional=False: Winsorizes each asset independently
            - Preserves original index and column names
        """
        lower, upper = limits

        if cross_sectional:
            winsorized = pd.DataFrame(index=returns.index, columns=returns.columns)

            for idx in returns.index:
                row = returns.loc[idx]
                lower_bound = np.percentile(row, lower * 100)
                upper_bound = np.percentile(row, (1 - upper) * 100)
                winsorized.loc[idx] = np.clip(row, lower_bound, upper_bound)

            return winsorized
        else:
            winsorized = pd.DataFrame(index=returns.index, columns=returns.columns)

            for col in returns.columns:
                series = returns[col]
                lower_bound = np.percentile(series, lower * 100)
                upper_bound = np.percentile(series, (1 - upper) * 100)
                winsorized[col] = np.clip(series, lower_bound, upper_bound)

            return winsorized

    @staticmethod
    def excess_returns(
        returns: pd.DataFrame,
        risk_free_rate: Union[float, pd.Series],
        return_type: ReturnType = ReturnType.LOG
    ) -> pd.DataFrame:
        """
        Calculate excess returns over risk-free rate.

        Args:
            returns: Return series (assets in columns)
            risk_free_rate: Either constant rate (as decimal, e.g., 0.02 for 2%)
                           or time series of rates (as decimals)
            return_type: Type of returns for proper calculation

        Returns:
            Excess returns for each asset

        Notes:
            - For LOG returns: Simple subtraction of rates
            - For ARITHMETIC returns: (1 + r_asset)/(1 + r_f) - 1
            - Handles both constant and time-varying risk-free rates
            - Automatically aligns if risk_free_rate is time series
            - Converts percentage rates to decimals if needed
        """
        if isinstance(risk_free_rate, (float, int)):
            rf = risk_free_rate / 100 if abs(risk_free_rate) > 1 else risk_free_rate

            if return_type == ReturnType.LOG:
                excess = returns.sub(rf)
            else:
                excess = (1 + returns).div(1 + rf) - 1
        else:
            aligned_returns, aligned_rf = align_series(returns, risk_free_rate)

            if (abs(aligned_rf) > 1).any():
                aligned_rf = aligned_rf / 100

            if return_type == ReturnType.LOG:
                excess = aligned_returns.sub(aligned_rf, axis=0)
            else:
                excess = (1 + aligned_returns).div(1 + aligned_rf, axis=0) - 1

        return excess

    @staticmethod
    def alpha(
        returns: pd.DataFrame,
        benchmark: pd.Series,
        return_type: ReturnType = ReturnType.LOG
    ) -> pd.DataFrame:
        """
        Calculate alpha returns relative to benchmark.

        Args:
            returns: Return series (assets in columns)
            benchmark: Benchmark return series
            return_type: Type of returns for proper calculation

        Returns:
            Alpha returns for each asset

        Notes:
            - Similar to excess returns but using benchmark series
            - For LOG returns: Simple subtraction
            - For ARITHMETIC returns: (1 + r_asset)/(1 + r_bench) - 1
            - Handles alignment automatically
            - Preserves original index and column names
        """
        aligned_returns, aligned_benchmark = align_series(returns, benchmark)

        if return_type == ReturnType.LOG:
            alpha = aligned_returns.sub(aligned_benchmark, axis=0)
        else:
            alpha = (1 + aligned_returns).div(1 + aligned_benchmark, axis=0) - 1

        return alpha
```

## Usage Examples

### Basic Transformations
```python
import pandas as pd
from fts.core.transformations.returns import ReturnTransformations

# Sample data
returns = pd.DataFrame({
    'asset1': [0.01, -0.02, 0.03, -0.01],
    'asset2': [-0.01, 0.02, -0.02, 0.01]
}, index=pd.date_range('2024-01-01', periods=4))

# Z-score standardization
# Cross-sectional (across assets at each time point)
z_cross = ReturnTransformations.zscore(returns, cross_sectional=True)

# Time series (each asset independently)
z_time = ReturnTransformations.zscore(returns, cross_sectional=False)

# Min-max scaling to [0,1]
scaled = ReturnTransformations.minmax(returns, feature_range=(0, 1))
```

### Rank Transformations
```python
# Cross-sectional ranking with Gaussian transformation
ranked = ReturnTransformations.rank_transform(
    returns,
    method='average',
    gaussian=True,
    cross_sectional=True
)

# Time series ranking
ranked_ts = ReturnTransformations.rank_transform(
    returns,
    cross_sectional=False
)
```

### Distribution Transformations
```python
# Box-Cox transformation
transformed, lambdas = ReturnTransformations.box_cox(returns)
print(f"Lambda parameters: {lambdas}")

# Winsorize outliers
winsorized = ReturnTransformations.winsorize(
    returns,
    limits=(0.05, 0.05),  # 5% on each tail
    cross_sectional=True
)
```

### Financial Transformations
```python
# Excess returns over constant risk-free rate
rf_rate = 0.02  # 2% annual rate
excess = ReturnTransformations.excess_returns(
    returns,
    risk_free_rate=rf_rate,
    return_type=ReturnType.LOG
)

# Excess returns over time-varying risk-free rate
rf_series = pd.Series([0.02, 0.025, 0.02, 0.015], index=returns.index)
excess_tv = ReturnTransformations.excess_returns(
    returns,
    risk_free_rate=rf_series,
    return_type=ReturnType.LOG
)

# Alpha relative to benchmark
benchmark = pd.Series([0.01, -0.01, 0.02, -0.005], index=returns.index)
alpha = ReturnTransformations.alpha(
    returns,
    benchmark=benchmark,
    return_type=ReturnType.LOG
)
```

### Chaining Transformations
```python
# Example: Winsorize -> Z-score -> Rank transform
returns_processed = (
    ReturnTransformations.winsorize(returns)
    .pipe(ReturnTransformations.zscore, cross_sectional=True)
    .pipe(ReturnTransformations.rank_transform, gaussian=True)
)
```
