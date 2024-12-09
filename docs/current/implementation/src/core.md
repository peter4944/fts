# Core Module Implementation

## 1. Data Structures

### 1.1 Base Classes

```python
@dataclass(frozen=True)
class ReturnSeriesMethod(Enum):
    """Method for generating return series."""
    ALL_INCLUDING_GAPS = 'all_including_gaps'  # Include all periods, NaN allowed per series
    ALL_OVERLAPPING = 'all_overlapping'    # Only periods where all series have data
    SYNCHRONIZED_AVERAGE = 'synchronized_average'  # Distribute returns across gaps
    PAIRWISE_OVERLAPPING = 'pairwise_overlapping'  # Use overlapping periods per pair
    PAIRWISE_AVERAGE = 'pairwise_average'  # Distribute returns across gaps per pair

class TimeSeries:
    """Base time series container."""

    def __init__(self, data: pd.Series, metadata: Optional[Dict] = None):
        self.data = data
        self.metadata = metadata or {}
        self._validate()

    def _validate(self) -> None:
        """Validate time series data."""
        if not isinstance(self.data, pd.Series):
            raise ValidationError("Data must be pandas Series")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValidationError("Index must be DatetimeIndex")

class ReturnSeries(TimeSeries):
    """Return series with specific calculations."""

    def __init__(self,
                 data: pd.Series,
                 method: ReturnSeriesMethod = ReturnSeriesMethod.ALL_INCLUDING_GAPS,
                 metadata: Optional[Dict] = None):
        super().__init__(data, metadata)
        self.method = method

    @classmethod
    def from_price_series(cls,
                         prices: pd.Series,
                         method: ReturnSeriesMethod = ReturnSeriesMethod.ALL_INCLUDING_GAPS,
                         geometric: bool = True) -> 'ReturnSeries':
        """Convert price series to returns."""
        if geometric:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return cls(returns, method=method)

    def standardize(self) -> 'ReturnSeries':
        """
        Standardize returns to mean=0, std=1.
        Required for many statistical calculations.
        """
        data = (self.data - self.data.mean()) / self.data.std()
        return ReturnSeries(data, self.method, self.metadata)

    def align_with(self, other: 'ReturnSeries',
                   method: ReturnSeriesMethod) -> Tuple['ReturnSeries', 'ReturnSeries']:
        """
        Align this series with another using specified method.

        Args:
            other: Series to align with
            method: Alignment method to use

        Returns:
            Tuple of aligned series (self, other)
        """
        if method == ReturnSeriesMethod.ALL_INCLUDING_GAPS:
            return self, other

        elif method == ReturnSeriesMethod.ALL_OVERLAPPING:
            df = pd.DataFrame({'self': self.data, 'other': other.data})
            aligned = df.dropna()
            return (
                ReturnSeries(aligned['self'], method),
                ReturnSeries(aligned['other'], method)
            )

        elif method == ReturnSeriesMethod.SYNCHRONIZED_AVERAGE:
            return self._align_synchronized_average(other)

        elif method == ReturnSeriesMethod.PAIRWISE_OVERLAPPING:
            return self._align_pairwise(other)

        elif method == ReturnSeriesMethod.PAIRWISE_AVERAGE:
            return self._align_pairwise_average(other)

        raise ValueError(f"Unsupported alignment method: {method}")

    def _align_synchronized_average(self,
                                  other: 'ReturnSeries'
                                  ) -> Tuple['ReturnSeries', 'ReturnSeries']:
        """
        Implement synchronized average method from background note.

        Distributes returns evenly across gaps while preserving total returns.
        Example:
            Time    X     Y        Action
            t1      1%    1%       Keep
            t2      2%    NaN      Average Y's t3 return over t2,t3
            t3      1%    2%       Average Y's t3 return over t2,t3
        """
        # Get combined index and identify all gaps
        combined_idx = self.data.index.union(other.data.index)
        df = pd.DataFrame({'self': self.data, 'other': other.data}, index=combined_idx)

        # Process each series
        aligned_data = {}
        for col in ['self', 'other']:
            series = df[col]
            gaps = self._identify_gaps(series)

            # Process each gap
            aligned_series = series.copy()
            for gap_start, gap_end in gaps:
                # Get returns before and after gap
                pre_gap_idx = series.index[gap_start-1]
                post_gap_idx = series.index[gap_end+1]
                pre_gap = series[pre_gap_idx]
                post_gap = series[post_gap_idx]

                # Calculate average return across gap
                gap_length = gap_end - gap_start + 1
                avg_return = (post_gap - pre_gap) / gap_length

                # Fill gap with average returns
                gap_indices = series.index[gap_start:gap_end+1]
                aligned_series[gap_indices] = avg_return

            aligned_data[col] = aligned_series

        return (
            ReturnSeries(aligned_data['self'], self.method),
            ReturnSeries(aligned_data['other'], self.method)
        )

    def _align_pairwise(self,
                        other: 'ReturnSeries'
                        ) -> Tuple['ReturnSeries', 'ReturnSeries']:
        """
        Implement pairwise overlapping method from background note.

        Uses only periods where both series have valid data.
        Example:
            Time    A     B
            t1      1%    1%    Include
            t2      1%    NaN   Exclude
            t3      1%    1%    Include
        """
        # Create DataFrame with both series
        df = pd.DataFrame({
            'self': self.data,
            'other': other.data
        })

        # Keep only overlapping periods
        aligned = df.dropna()

        return (
            ReturnSeries(aligned['self'], self.method),
            ReturnSeries(aligned['other'], self.method)
        )

    def _align_pairwise_average(self,
                               other: 'ReturnSeries'
                               ) -> Tuple['ReturnSeries', 'ReturnSeries']:
        """
        Implement pairwise average method from background note.

        For gaps, distribute returns evenly while preserving total returns.
        Example:
            Time    A     B           A distributed    B distributed
            t1      1%    NaN         1%              0.5%
            t2      1%    1%          1%              0.5%
            t3      NaN   -1%         0.5%            -1%
            t4      1%    1%          0.5%            1%
        """
        # Get combined index
        combined_idx = self.data.index.union(other.data.index)
        df = pd.DataFrame({
            'self': self.data,
            'other': other.data
        }, index=combined_idx)

        # Process each series independently
        aligned_data = {}
        for col in ['self', 'other']:
            series = df[col]
            gaps = self._identify_gaps(series)

            # Process each gap
            aligned_series = series.copy()
            for gap_start, gap_end in gaps:
                # Get cumulative return across gap
                pre_gap_idx = series.index[gap_start-1]
                post_gap_idx = series.index[gap_end+1]
                cum_return = series[post_gap_idx] - series[pre_gap_idx]

                # Distribute return evenly
                gap_length = gap_end - gap_start + 1
                avg_return = cum_return / gap_length

                # Fill gap
                gap_indices = series.index[gap_start:gap_end+1]
                aligned_series[gap_indices] = avg_return

            aligned_data[col] = aligned_series

        return (
            ReturnSeries(aligned_data['self'], self.method),
            ReturnSeries(aligned_data['other'], self.method)
        )

class TimeSeriesCollection:
    """Collection of time series with alignment options."""

    def __init__(self,
                 series: Dict[str, TimeSeries],
                 method: ReturnSeriesMethod = ReturnSeriesMethod.ALL_INCLUDING_GAPS):
        self.series = series
        self.method = method
        self._validate()

    def align(self, method: Optional[ReturnSeriesMethod] = None) -> 'TimeSeriesCollection':
        """Align series according to specified method."""
        method = method or self.method

        if method == ReturnSeriesMethod.ALL_INCLUDING_GAPS:
            return self
        elif method == ReturnSeriesMethod.ALL_OVERLAPPING:
            df = pd.DataFrame({k: s.data for k, s in self.series.items()})
            aligned = df.dropna()
            return TimeSeriesCollection(
                {k: TimeSeries(aligned[k], method=method)
                 for k in self.series.keys()}
            )

    def align_all(self, method: Optional[ReturnSeriesMethod] = None) -> 'TimeSeriesCollection':
        """
        Align all series according to specified method.

        Args:
            method: Alignment method to use, defaults to collection's method

        Returns:
            New TimeSeriesCollection with aligned series
        """
        method = method or self.method

        if method in [ReturnSeriesMethod.ALL_INCLUDING_GAPS,
                     ReturnSeriesMethod.ALL_OVERLAPPING]:
            return self._align_all_series(method)
        elif method in [ReturnSeriesMethod.PAIRWISE_OVERLAPPING,
                       ReturnSeriesMethod.PAIRWISE_AVERAGE]:
            return self._align_pairwise_series(method)
        elif method == ReturnSeriesMethod.SYNCHRONIZED_AVERAGE:
            return self._align_synchronized_average()

        raise ValueError(f"Unsupported alignment method: {method}")

    def _align_all_series(self, method: ReturnSeriesMethod) -> 'TimeSeriesCollection':
        """Align all series together."""
        df = pd.DataFrame({k: s.data for k, s in self.series.items()})

        if method == ReturnSeriesMethod.ALL_INCLUDING_GAPS:
            aligned = df
        else:  # ALL_OVERLAPPING
            aligned = df.dropna()

        return TimeSeriesCollection(
            {k: ReturnSeries(aligned[k], method=method)
             for k in aligned.columns},
            method=method
        )

    def _align_pairwise_series(self, method: ReturnSeriesMethod) -> 'TimeSeriesCollection':
        """
        Align series using pairwise methods.

        For N series, creates N(N-1)/2 pairs and aligns each pair independently.
        Returns collection with aligned series based on method:
        - PAIRWISE_OVERLAPPING: Uses only overlapping periods for each pair
        - PAIRWISE_AVERAGE: Distributes returns across gaps for each pair
        """
        series_names = list(self.series.keys())
        n_series = len(series_names)
        aligned_pairs = {}

        # Process all pairs
        for i in range(n_series):
            for j in range(i+1, n_series):
                name_i, name_j = series_names[i], series_names[j]
                series_i = self.series[name_i]
                series_j = self.series[name_j]

                # Align pair using specified method
                aligned_i, aligned_j = series_i.align_with(series_j, method)

                # Store aligned pair
                pair_key = f"{name_i}_{name_j}"
                aligned_pairs[pair_key] = (aligned_i, aligned_j)

        return TimeSeriesCollection(
            self._reconstruct_from_pairs(aligned_pairs, series_names),
            method=method
        )

    def _align_synchronized_average(self) -> 'TimeSeriesCollection':
        """
        Align all series using synchronized average method.

        Implements the synchronized average method from the background note:
        1. Identify all gaps in each series
        2. For each gap, distribute returns across gap period
        3. Ensure total return across gap is preserved
        """
        df = pd.DataFrame({k: s.data for k, s in self.series.items()})
        aligned_data = {}

        for col in df.columns:
            series = df[col]
            gaps = self._identify_gaps(series)

            # Process each gap
            aligned_series = series.copy()
            for gap_start, gap_end in gaps:
                # Get returns before and after gap
                pre_gap = series.iloc[gap_start-1]
                post_gap = series.iloc[gap_end+1]

                # Calculate average return across gap
                gap_length = gap_end - gap_start + 1
                avg_return = (post_gap - pre_gap) / gap_length

                # Fill gap with average returns
                aligned_series.iloc[gap_start:gap_end+1] = avg_return

            aligned_data[col] = aligned_series

        return TimeSeriesCollection(
            {k: ReturnSeries(v, method=ReturnSeriesMethod.SYNCHRONIZED_AVERAGE)
             for k, v in aligned_data.items()},
            method=ReturnSeriesMethod.SYNCHRONIZED_AVERAGE
        )

    def _identify_gaps(self, series: pd.Series) -> List[Tuple[int, int]]:
        """
        Identify gaps (NaN sequences) in a series.

        Returns:
            List of (start_idx, end_idx) tuples for each gap
        """
        is_nan = series.isna()
        gaps = []
        gap_start = None

        for i, nan_flag in enumerate(is_nan):
            if nan_flag and gap_start is None:
                gap_start = i
            elif not nan_flag and gap_start is not None:
                gaps.append((gap_start, i-1))
                gap_start = None

        # Handle gap at end of series
        if gap_start is not None:
            gaps.append((gap_start, len(series)-1))

        return gaps

    def _reconstruct_from_pairs(self,
                               aligned_pairs: Dict[str, Tuple[ReturnSeries, ReturnSeries]],
                               series_names: List[str]) -> Dict[str, ReturnSeries]:
        """
        Reconstruct individual series from aligned pairs.

        For pairwise methods, need to combine multiple aligned versions
        of each series into a single consistent series.
        """
        # Implementation depends on specific requirements for
        # combining multiple aligned versions of the same series
        pass

### 1.2 Validation Framework

```python
class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}

class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_return_series(series: pd.Series) -> ValidationResult:
        """Validate return series data."""
        result = ValidationResult()

        # Check data type
        if not isinstance(series, pd.Series):
            result.is_valid = False
            result.errors.append("Data must be pandas Series")
            return result

        # Check index
        if not isinstance(series.index, pd.DatetimeIndex):
            result.is_valid = False
            result.errors.append("Index must be DatetimeIndex")

        # Check for infinite values
        if np.any(np.isinf(series.dropna())):
            result.is_valid = False
            result.errors.append("Series contains infinite values")

        # Add metrics
        result.metrics.update({
            'length': len(series),
            'missing_pct': series.isna().mean() * 100,
            'start_date': series.index[0],
            'end_date': series.index[-1]
        })

        return result
```

## 2. Error Handling

```python
class FTSError(Exception):
    """Base exception for all FTS errors."""
    pass

class ValidationError(FTSError):
    """Data validation errors."""
    pass

class ProcessingError(FTSError):
    """Data processing errors."""
    pass

class ConfigurationError(FTSError):
    """Configuration related errors."""
    pass
```

## 3. Memory Management

```python
@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_pct: float = 0.75  # Maximum memory usage as percentage of system RAM
    chunk_size_mb: int = 100      # Default chunk size in MB
    cache_size_mb: int = 1000     # Maximum cache size in MB

class MemoryManager:
    """Memory management system."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._total_memory = psutil.virtual_memory().total
        self._max_memory = int(self._total_memory * self.config.max_memory_pct)
        self._cache = {}
```

## 4. Annualization

```python
"""Base module for core functionality."""

# Trading days per period for annualization
ANNUALIZATION_FACTORS = {
    'D': 252,    # Daily
    'W': 52,     # Weekly
    'M': 12,     # Monthly
    'Q': 4,      # Quarterly
    'A': 1       # Annual
}

def get_annualization_factor(frequency: str) -> float:
    """
    Get annualization factor for given frequency.

    Args:
        frequency: Return series frequency ('D', 'W', 'M', 'Q', 'A')

    Returns:
        Square root of annualization factor

    Raises:
        ValueError: If frequency not supported
    """
    if frequency not in ANNUALIZATION_FACTORS:
        raise ValueError(
            f"Unsupported frequency: {frequency}. "
            f"Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
        )
    return np.sqrt(ANNUALIZATION_FACTORS[frequency])
```
