# Core Base Module

## 1. Overview
This module provides base classes and core functionality:
- Base classes for time series data
- Return series operations
- Price series operations
- Drawdown calculations
- Data validation

### Core Dependencies
```python
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import warnings

from .validation import ValidationRules
from .errors import ValidationError
from .transformations.returns import ReturnTransformations
```

### Type Definitions
```python
class FrequencyType(Enum):
    """Supported time series frequencies."""
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    ANNUAL = 'A'

class ReturnType(Enum):
    """Types of return calculations."""
    ARITHMETIC = "arithmetic"  # (P1 - P0) / P0
    GEOMETRIC = "geometric"   # P1 / P0 - 1
    LOG = "log"              # ln(P1 / P0)

class ValidationLevel(Enum):
    """Levels of validation checking."""
    NONE = "none"      # No validation
    BASIC = "basic"    # Basic type and value checks
    STRICT = "strict"  # Comprehensive validation
```

## 2. Base Classes

### 2.1 TimeSeries Base Class
```python
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any

class TimeSeries(ABC):
    """
    Abstract base class for time series data.

    Attributes:
        data: Time series data
        metadata: Series metadata
        frequency: Data frequency (FrequencyType)
        validation_level: Level of validation to apply

    Notes:
        - Enforces common interface
        - Handles validation
        - Provides utility methods
    """
    def __init__(
        self,
        data: pd.Series,
        metadata: Optional[Dict] = None,
        frequency: FrequencyType = FrequencyType.DAILY,
        validation_level: ValidationLevel = ValidationLevel.BASIC
    ):
        self._data = data
        self._metadata = metadata or {}
        self._frequency = frequency
        self._validation_level = validation_level
        self.validate()

    @abstractmethod
    def validate(self) -> bool:
        """Validate time series data."""
        if self._validation_level == ValidationLevel.NONE:
            return True

        if not isinstance(self._data, pd.Series):
            raise ValidationError("Data must be pandas Series")

        if not isinstance(self._data.index, pd.DatetimeIndex):
            raise ValidationError("Index must be DatetimeIndex")

        if self._validation_level == ValidationLevel.STRICT:
            # Check for sorted index
            if not self._data.index.is_monotonic_increasing:
                raise ValidationError("Index must be sorted")

            # Check for duplicates
            if self._data.index.has_duplicates:
                raise ValidationError("Index contains duplicates")

        return True

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return self._data.copy()

    @abstractmethod
    def resample(self, freq: FrequencyType) -> 'TimeSeries':
        """Resample time series to new frequency."""
        # Validate frequency conversion
        valid_conversions = {
            FrequencyType.DAILY: {FrequencyType.WEEKLY, FrequencyType.MONTHLY},
            FrequencyType.MONTHLY: {FrequencyType.QUARTERLY, FrequencyType.ANNUAL},
            FrequencyType.QUARTERLY: {FrequencyType.ANNUAL}
        }

        if (freq != self._frequency and
            freq not in valid_conversions.get(self._frequency, set())):
            raise ValidationError(
                f"Invalid frequency conversion from {self._frequency} to {freq}"
            )

        return NotImplemented

    # Properties
    @property
    def start_date(self) -> datetime:
        """Start date of the series."""
        return self._data.index[0]

    @property
    def end_date(self) -> datetime:
        """End date of the series."""
        return self._data.index[-1]

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self._data)

    # Arithmetic operations
    def __add__(self, other: Union['TimeSeries', float]) -> 'TimeSeries':
        """Add two series or add constant."""
        if isinstance(other, TimeSeries):
            raise NotImplementedError("Operation requires alignment. Use data.alignment module.")
        return type(self)(self._data + other)

    def __mul__(self, other: Union['TimeSeries', float]) -> 'TimeSeries':
        """Multiply two series or multiply by constant."""
        if isinstance(other, TimeSeries):
            raise NotImplementedError("Operation requires alignment. Use data.alignment module.")
        return type(self)(self._data * other)

    def __truediv__(self, other: Union['TimeSeries', float]) -> 'TimeSeries':
        """Divide two series or divide by constant."""
        if isinstance(other, TimeSeries):
            raise NotImplementedError("Operation requires alignment. Use data.alignment module.")
        return type(self)(self._data / other)
```

### 2.2 ReturnSeries Class
```python
class TransformationType(Enum):
    """Types of transformations applied to returns."""
    STANDARDIZED = "standardized"
    MINMAX_SCALED = "minmax_scaled"
    RANK_TRANSFORMED = "rank_transformed"
    BOX_COX = "box_cox"
    WINSORIZED = "winsorized"
    EXCESS = "excess_returns"
    ALPHA = "alpha"

class ReturnSeries(TimeSeries):
    """
    Class for handling return series data.

    Attributes:
        data: Return series data
        metadata: Series metadata
        frequency: Data frequency
        return_type: Type of returns (ARITHMETIC, GEOMETRIC, LOG)

    Methods:
        validate: Validate return series data
        to_prices: Convert returns to price series
    """
    def __init__(
        self,
        data: pd.Series,
        return_type: ReturnType = ReturnType.LOG,
        metadata: Optional[Dict] = None,
        frequency: FrequencyType = FrequencyType.DAILY,
        validation_level: ValidationLevel = ValidationLevel.BASIC,
        transformations: Optional[List[Dict[str, Any]]] = None
    ):
        self._return_type = return_type
        self._transformations = transformations or []
        super().__init__(data, metadata, frequency, validation_level)

    @property
    def transformations(self) -> List[Dict[str, Any]]:
        """Get history of transformations applied to series."""
        return self._transformations.copy()

    def _add_transformation(
        self,
        transform_type: TransformationType,
        params: Dict[str, Any]
    ) -> None:
        """Record a transformation."""
        self._transformations.append({
            'type': transform_type,
            'params': params,
            'timestamp': datetime.now()
        })

    def validate(self) -> bool:
        """Validate return series data."""
        super().validate()

        if self._validation_level >= ValidationLevel.BASIC:
            # Check for non-finite values
            if not np.isfinite(self._data).all():
                raise ValidationError("Returns contain non-finite values")

            # Return type specific validation
            if self._return_type == ReturnType.ARITHMETIC:
                if (self._data <= -1).any():
                    raise ValidationError("Arithmetic returns cannot be <= -1")
            elif self._return_type == ReturnType.GEOMETRIC:
                if (self._data <= -1).any():
                    raise ValidationError("Geometric returns cannot be <= -1")

        return True

    def to_prices(
        self,
        anchor_price: float = 100.0,
        anchor_index: Optional[Union[int, datetime, str]] = None,
        direction: str = 'forward'
    ) -> 'PriceSeries':
        """
        Convert returns to price series.

        Args:
            anchor_price: Price value to anchor the series to
            anchor_index: Index position to anchor price at
                         If None, uses first position for forward,
                         last position for backward
            direction: Direction to generate prices ('forward' or 'backward')

        Returns:
            PriceSeries with reconstructed prices

        Notes:
            - For forward: P(t) = P(t-1) * (1 + r(t)) [arithmetic]
                      P(t) = P(t-1) * exp(r(t)) [log]
            - For backward: P(t-1) = P(t) / (1 + r(t)) [arithmetic]
                       P(t-1) = P(t) / exp(r(t)) [log]
        """
        # Validate inputs
        if direction not in ['forward', 'backward']:
            raise ValueError("direction must be 'forward' or 'backward'")

        # Get anchor position
        if anchor_index is None:
            anchor_pos = 0 if direction == 'forward' else len(self._data) - 1
        else:
            if isinstance(anchor_index, (str, datetime)):
                anchor_pos = self._data.index.get_loc(anchor_index)
            else:
                anchor_pos = anchor_index

        # Initialize price array
        prices = np.zeros_like(self._data, dtype=np.float64)
        prices[anchor_pos] = anchor_price

        if direction == 'forward':
            # Forward accumulation
            if self._return_type == ReturnType.LOG:
                # For log returns: P(t) = P(0) * exp(sum(r[0:t]))
                cumsum = self._data.cumsum()
                prices = anchor_price * np.exp(cumsum - cumsum[anchor_pos])
            else:
                # For arithmetic returns: P(t) = P(0) * prod(1 + r)
                for i in range(anchor_pos + 1, len(prices)):
                    prices[i] = prices[i-1] * (1 + self._data.iloc[i])
                for i in range(anchor_pos - 1, -1, -1):
                    prices[i] = prices[i+1] / (1 + self._data.iloc[i+1])
        else:
            # Backward accumulation
            if self._return_type == ReturnType.LOG:
                # For log returns: P(t-1) = P(t) / exp(r(t))
                cumsum = self._data.cumsum()
                prices = anchor_price * np.exp(-(cumsum - cumsum[anchor_pos]))
            else:
                # For arithmetic returns: P(t-1) = P(t) / (1 + r(t))
                for i in range(anchor_pos - 1, -1, -1):
                    prices[i] = prices[i+1] / (1 + self._data.iloc[i+1])
                for i in range(anchor_pos + 1, len(prices)):
                    prices[i] = prices[i-1] * (1 + self._data.iloc[i])

        return PriceSeries(
            pd.Series(prices, index=self._data.index),
            self._metadata
        )

    def convert_return_type(self, to_type: ReturnType) -> 'ReturnSeries':
        """
        Convert returns to a different return type.

        Args:
            to_type: Target return type

        Returns:
            ReturnSeries with converted return type

        Notes:
            Converts through prices to ensure accuracy:
            - First converts current returns to prices
            - Then converts prices to desired return type
        """
        if self._return_type == to_type:
            return self

        # Convert to prices (using initial price of 1.0)
        prices = self.to_prices(initial_price=1.0)._data

        # Convert to desired return type
        if to_type == ReturnType.ARITHMETIC:
            new_data = prices.pct_change()
        elif to_type == ReturnType.GEOMETRIC:
            new_data = prices / prices.shift(1) - 1
        else:  # LOG returns
            new_data = np.log(prices / prices.shift(1))

        return type(self)(
            new_data.dropna(),
            return_type=to_type,
            metadata=self._metadata,
            frequency=self._frequency
        )

    def rolling_stats(self, window: int) -> Dict[StatisticType, pd.Series]:
        """Calculate rolling statistics."""
        roll = self._data.rolling(window=window)
        return {
            StatisticType.MEAN: roll.mean(),
            StatisticType.VOLATILITY: roll.std() * np.sqrt(252),  # Annualized
            StatisticType.SKEWNESS: roll.skew(),
            StatisticType.KURTOSIS: roll.kurt()
        }

    @property
    def return_type(self) -> ReturnType:
        """Get the type of returns."""
        return self._return_type

    def standardize(self, cross_sectional: bool = False, demean: bool = True) -> 'ReturnSeries':
        """Z-score standardization wrapper."""
        std_returns = ReturnTransformations.zscore(
            self._data,
            cross_sectional=cross_sectional,
            demean=demean
        )
        new_series = type(self)(
            std_returns,
            return_type=self.return_type,
            metadata=self._metadata,
            transformations=self._transformations
        )
        new_series._add_transformation(
            TransformationType.STANDARDIZED,
            {'cross_sectional': cross_sectional, 'demean': demean}
        )
        return new_series

    def minmax_scale(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        cross_sectional: bool = False
    ) -> 'ReturnSeries':
        """Min-max scaling wrapper."""
        scaled = ReturnTransformations.minmax(
            self._data,
            feature_range=feature_range,
            cross_sectional=cross_sectional
        )
        return type(self)(
            scaled,
            return_type=self.return_type,
            metadata=self._metadata
        )

    def rank_transform(
        self,
        method: str = 'average',
        gaussian: bool = False,
        cross_sectional: bool = True
    ) -> 'ReturnSeries':
        """Rank transformation wrapper."""
        ranked = ReturnTransformations.rank_transform(
            self._data,
            method=method,
            gaussian=gaussian,
            cross_sectional=cross_sectional
        )
        return type(self)(
            ranked,
            return_type=self.return_type,
            metadata=self._metadata
        )

    def box_cox_transform(
        self,
        shift: Optional[float] = None
    ) -> Tuple['ReturnSeries', Dict[str, float]]:
        """Box-Cox transformation wrapper."""
        transformed, lambdas = ReturnTransformations.box_cox(
            self._data,
            shift=shift
        )
        return (
            type(self)(
                transformed,
                return_type=self.return_type,
                metadata=self._metadata
            ),
            lambdas
        )

    def winsorize(
        self,
        limits: Tuple[float, float] = (0.05, 0.05),
        cross_sectional: bool = True
    ) -> 'ReturnSeries':
        """Winsorize extreme values wrapper."""
        winsorized = ReturnTransformations.winsorize(
            self._data,
            limits=limits,
            cross_sectional=cross_sectional
        )
        return type(self)(
            winsorized,
            return_type=self.return_type,
            metadata=self._metadata
        )

    def excess_returns(
        self,
        risk_free_rate: Union[float, pd.Series]
    ) -> 'ReturnSeries':
        """Calculate excess returns over risk-free rate."""
        excess = ReturnTransformations.excess_returns(
            self._data,
            risk_free_rate=risk_free_rate,
            return_type=self.return_type
        )
        return type(self)(
            excess,
            return_type=self.return_type,
            metadata=self._metadata
        )

    def alpha(
        self,
        benchmark: pd.Series
    ) -> 'ReturnSeries':
        """Calculate alpha returns relative to benchmark."""
        alpha_returns = ReturnTransformations.alpha(
            self._data,
            benchmark=benchmark,
            return_type=self.return_type
        )
        return type(self)(
            alpha_returns,
            return_type=self.return_type,
            metadata=self._metadata
        )

    def resample(self, freq: FrequencyType) -> 'ReturnSeries':
        """
        Resample return series to new frequency.

        Notes:
            - For arithmetic returns: Compounds returns within period
            - For log returns: Sums returns within period
            - Not supported for transformed returns

        Raises:
            ValueError: If series has been transformed
        """
        if self._transformations:
            raise ValueError(
                "Cannot resample transformed returns. "
                "Consider resampling original returns or prices first."
            )

        super().resample(freq)  # Validate frequency

        if self._return_type == ReturnType.LOG:
            # Sum log returns within period
            resampled = self._data.resample(freq.value).sum()
        else:
            # Compound arithmetic returns within period
            # (1 + r1)(1 + r2)...(1 + rn) - 1
            resampled = (1 + self._data).resample(freq.value).prod() - 1

        return type(self)(
            resampled,
            return_type=self._return_type,
            metadata=self._metadata,
            frequency=freq
        )
```

### 2.3 PriceSeries Class
```python
class PriceSeries(TimeSeries):
    """
    Class for handling price series data.

    Attributes:
        data: Price series data
        metadata: Series metadata
        frequency: Data frequency

    Methods:
        validate: Validate price series data
        to_returns: Convert to return series
        normalize: Normalize prices to a starting value
    """
    def __init__(
        self,
        data: pd.Series,
        metadata: Optional[Dict] = None,
        frequency: FrequencyType = FrequencyType.DAILY,
        validation_level: ValidationLevel = ValidationLevel.BASIC
    ):
        super().__init__(data, metadata, frequency, validation_level)

    def validate(self) -> bool:
        """Validate price series data."""
        super().validate()

        if self._validation_level >= ValidationLevel.BASIC:
            # Check for non-positive prices
            if (self._data <= 0).any():
                raise ValidationError("Prices must be positive")

            # Check for missing values
            if self._data.isnull().any():
                raise ValidationError("Price series contains missing values")

        if self._validation_level == ValidationLevel.STRICT:
            # Check for extreme price changes
            pct_changes = self._data.pct_change()
            mean, std = pct_changes.mean(), pct_changes.std()
            outliers = np.abs(pct_changes - mean) > 5 * std
            if outliers.any():
                warnings.warn(f"Found {outliers.sum()} potential price jump outliers")

        return True

    def to_returns(self, geometric: bool = False) -> ReturnSeries:
        """
        Convert prices to return series.

        By default, returns arithmetic returns. For geometric returns,
        use geometric=True.

        Args:
            geometric: If True, calculate geometric returns
                      If False, calculate arithmetic returns

        Returns:
            ReturnSeries with either arithmetic or geometric returns
        """
        if geometric:
            # Geometric returns: P(t)/P(t-1) - 1
            returns = self._data / self._data.shift(1) - 1
            return_type = ReturnType.GEOMETRIC
        else:
            # Arithmetic returns: (P(t) - P(t-1))/P(t-1)
            returns = self._data.pct_change()
            return_type = ReturnType.ARITHMETIC

        return ReturnSeries(
            returns.dropna(),
            return_type=return_type,
            metadata=self._metadata,
            frequency=self._frequency
        )

    def normalize(
        self,
        value: float = 100.0
    ) -> 'PriceSeries':
        """
        Normalize prices to a starting value.

        Args:
            value: Target starting value

        Returns:
            Normalized PriceSeries
        """
        norm_prices = self._data / self._data.iloc[0] * value
        return type(self)(
            norm_prices,
            metadata=self._metadata,
            frequency=self._frequency
        )

    def resample(
        self,
        freq: FrequencyType,
        method: str = 'last'
    ) -> 'PriceSeries':
        """
        Resample price series to new frequency.

        Args:
            freq: Target frequency
            method: How to sample prices ('last', 'first', 'mean', 'max', 'min')
        """
        super().resample(freq)  # Validate frequency

        if method == 'last':
            if freq == FrequencyType.WEEKLY:
                resampled = self._data.resample('W-FRI').last()
            else:
                resampled = self._data.resample(freq.value).last()
        elif method == 'first':
            resampled = self._data.resample(freq.value).first()
        elif method in ['mean', 'max', 'min']:
            resampled = getattr(self._data.resample(freq.value), method)()
        else:
            raise ValueError(f"Unsupported resampling method: {method}")

        return type(self)(resampled, self._metadata, freq)
```

### 2.4 DrawdownSeries Class
```python
class DrawdownSeries:
    """
    Calculate and analyze drawdown series.

    Handles:
    - Drawdown calculation from returns/prices
    - Maximum drawdown
    - Drawdown statistics
    - Vectorized operations for multiple series
    """

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame],
        is_returns: bool = True,
        return_type: ReturnType = ReturnType.LOG
    ):
        """
        Initialize with price or return series.

        Args:
            data: Price or return series (Series/DataFrame)
            is_returns: Whether input is returns (True) or prices (False)
            return_type: Type of returns if is_returns=True

        Notes:
            - Handles both single and multiple series
            - Supports both log and arithmetic returns
            - Optimized for vectorized operations
        """
        self._validate_inputs(data)

        # Store data efficiently
        self._dtype = np.float32
        self._data = data.to_numpy(dtype=self._dtype)
        self._index = data.index
        self._columns = data.columns if isinstance(data, pd.DataFrame) else None

        # Convert returns to prices if needed
        if is_returns:
            self._prices = self._returns_to_prices(self._data, return_type)
        else:
            self._prices = self._data

        # Initialize cache
        self._cache = {}

        # Compute drawdown series
        self._compute_drawdowns()

    def _validate_inputs(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Validate input data."""
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError("Data must be pandas Series or DataFrame")
        if data.isnull().any().any():
            raise ValueError("Data contains null values")

    def _returns_to_prices(
        self,
        returns: np.ndarray,
        return_type: ReturnType
    ) -> np.ndarray:
        """Convert returns to price series."""
        if return_type == ReturnType.LOG:
            return np.exp(np.cumsum(returns, axis=0))
        else:
            return (1 + returns).cumprod(axis=0)

    def _compute_drawdowns(self) -> None:
        """
        Compute drawdown series vectorized.

        Formula:
        drawdown = price / running_max - 1

        Note: Uses vectorized operations for efficiency
        """
        running_max = np.maximum.accumulate(self._prices, axis=0)
        self._drawdowns = self._prices / running_max - 1

    @property
    def drawdown_series(self) -> Union[pd.Series, pd.DataFrame]:
        """Get drawdown series with original index/columns."""
        return self.to_pandas(self._drawdowns)

    @property
    def max_drawdown(self) -> Union[float, pd.Series]:
        """Get maximum drawdown (most negative value)."""
        if 'max_dd' not in self._cache:
            self._cache['max_dd'] = np.min(self._drawdowns, axis=0)
        return self.to_pandas(self._cache['max_dd'])

    def drawdown_stats(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate comprehensive drawdown statistics.

        Returns:
            Dict containing:
            - max_drawdown: Maximum drawdown
            - avg_drawdown: Average drawdown
            - drawdown_volatility: Volatility of drawdowns
            - time_underwater: Percentage of time in drawdown
            - max_drawdown_duration: Longest drawdown period
        """
        if 'dd_stats' not in self._cache:
            # Vectorized calculations
            max_dd = np.min(self._drawdowns, axis=0)
            avg_dd = np.mean(self._drawdowns, axis=0)
            dd_vol = np.std(self._drawdowns, axis=0)

            # Time underwater (vectorized)
            underwater = self._drawdowns < 0
            time_underwater = np.mean(underwater, axis=0)

            # Max drawdown duration (vectorized)
            dd_duration = np.zeros_like(self._drawdowns)
            current_duration = np.zeros(self._drawdowns.shape[1])

            for i in range(len(self._drawdowns)):
                current_duration[underwater[i]] += 1
                current_duration[~underwater[i]] = 0
                dd_duration[i] = current_duration

            max_duration = np.max(dd_duration, axis=0)

            self._cache['dd_stats'] = {
                'max_drawdown': max_dd,
                'avg_drawdown': avg_dd,
                'drawdown_volatility': dd_vol,
                'time_underwater': time_underwater,
                'max_drawdown_duration': max_duration
            }

        return {k: self.to_pandas(v) for k, v in self._cache['dd_stats'].items()}

    def to_pandas(self, data: np.ndarray) -> Union[pd.Series, pd.DataFrame]:
        """Convert numpy array back to pandas with original index/columns."""
        if self._columns is not None:
            if data.ndim == 1:
                return pd.Series(data, index=self._columns)
            return pd.DataFrame(data, index=self._index, columns=self._columns)
        if data.ndim == 1:
            return pd.Series(data, index=self._index)
        return pd.Series(data.flatten(), index=self._index)
```

## 3. Usage Guidelines

### 3.1 Common Use Cases
```python
# Basic time series operations
prices = PriceSeries(data, frequency=FrequencyType.DAILY)
returns = prices.to_returns()
drawdowns = DrawdownSeries(prices, is_returns=False)

# Frequency conversion
monthly = daily_returns.resample(FrequencyType.MONTHLY)
```
