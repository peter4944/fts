# Data Module Implementation

## 1. Data Import and Export

```python
class CSVImporter:
    """CSV data import with validation."""

    def __init__(self, validation_rules: Optional[Dict] = None):
        self.validation_rules = validation_rules or {}

    def import_price_series(self,
                           filepath: str,
                           date_column: str = 'Date') -> Dict[str, TimeSeries]:
        """
        Import price series from CSV.

        Expected CSV format:
        Date,Series1,Series2,...
        2024-01-01,100.0,200.0,...
        2024-01-02,101.0,201.0,...

        Args:
            filepath: Path to CSV file
            date_column: Name of date column (default: 'Date')

        Returns:
            Dictionary of TimeSeries objects

        Raises:
            ValidationError: If CSV format is invalid
        """
        # Read CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValidationError(f"Failed to read CSV file: {str(e)}")

        # Validate CSV structure
        if date_column not in df.columns:
            raise ValidationError(f"Date column '{date_column}' not found")

        # Get price columns (all columns except date)
        price_columns = [col for col in df.columns if col != date_column]
        if not price_columns:
            raise ValidationError("No price series columns found")

        # Convert date column
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValidationError(f"Invalid date format in column '{date_column}': {str(e)}")

        df.set_index(date_column, inplace=True)

        # Validate price data
        for col in price_columns:
            if not pd.to_numeric(df[col], errors='coerce').notna().any():
                raise ValidationError(f"No valid numeric data in column '{col}'")

        # Create TimeSeries objects
        return {
            col: TimeSeries(df[col])
            for col in price_columns
        }

    def _validate_price_data(self, df: pd.DataFrame, column: str) -> None:
        """
        Validate price data in a column.

        Checks:
        - Contains numeric data
        - No negative prices
        - No extreme values
        """
        series = pd.to_numeric(df[column], errors='coerce')

        if series.isna().all():
            raise ValidationError(f"Column '{column}' contains no valid numeric data")

        if (series < 0).any():
            raise ValidationError(f"Column '{column}' contains negative prices")

        # Check for extreme values (e.g., > 3 std from mean)
        mean = series.mean()
        std = series.std()
        if ((series - mean).abs() > 3 * std).any():
            warnings.warn(f"Column '{column}' contains potential outliers")

class DataExporter:
    """Export data in various formats."""

    @staticmethod
    def to_csv(series_collection: TimeSeriesCollection,
               filepath: str,
               date_format: str = '%Y-%m-%d'):
        """Export collection to CSV."""
        df = pd.DataFrame({
            name: series.data
            for name, series in series_collection.series.items()
        })
        df.index = df.index.strftime(date_format)
        df.to_csv(filepath)
```

## 2. Gap Detection and Handling

```python
@dataclass
class GapAnalysis:
    """Results of gap analysis."""
    gaps: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]
    statistics: Dict[str, Dict[str, float]]
    recommendations: Dict[str, str]

class GapAnalyzer:
    """Analyze gaps in time series data."""

    def analyze_gaps(self,
                    series_collection: TimeSeriesCollection) -> GapAnalysis:
        """
        Analyze gaps in multiple series.

        Provides:
        - Location and length of gaps
        - Gap statistics
        - Handling recommendations
        """
        gaps = {}
        stats = {}
        recommendations = {}

        for name, series in series_collection.series.items():
            # Find gaps
            series_gaps = self._find_gaps(series.data)
            gaps[name] = series_gaps

            # Calculate statistics
            stats[name] = self._calculate_gap_statistics(series_gaps)

            # Generate recommendations
            recommendations[name] = self._recommend_handling(stats[name])

        return GapAnalysis(gaps, stats, recommendations)

    def _find_gaps(self,
                   series: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find start and end dates of gaps."""
        gaps = []
        is_missing = series.isna()

        if not is_missing.any():
            return gaps

        gap_start = None
        for date, missing in is_missing.items():
            if missing and gap_start is None:
                gap_start = date
            elif not missing and gap_start is not None:
                gaps.append((gap_start, date))
                gap_start = None

        return gaps

    def _calculate_gap_statistics(self,
                                gaps: List[Tuple[pd.Timestamp, pd.Timestamp]]
                                ) -> Dict[str, float]:
        """Calculate gap statistics."""
        if not gaps:
            return {
                'total_gaps': 0,
                'avg_gap_length': 0,
                'max_gap_length': 0,
                'gap_density': 0
            }

        gap_lengths = [(end - start).days for start, end in gaps]
        return {
            'total_gaps': len(gaps),
            'avg_gap_length': np.mean(gap_lengths),
            'max_gap_length': max(gap_lengths),
            'gap_density': len(gaps) / sum(gap_lengths)
        }

    def _recommend_handling(self,
                          stats: Dict[str, float]) -> str:
        """Recommend gap handling method based on statistics."""
        if stats['total_gaps'] == 0:
            return "No gaps present"

        if stats['gap_density'] > 0.5:
            return "Consider using SYNCHRONIZED_AVERAGE method"
        elif stats['max_gap_length'] > 30:
            return "Consider using PAIRWISE_OVERLAPPING method"
        else:
            return "Standard ALL_OVERLAPPING method should suffice"
```

## 3. Data Preprocessing

```python
class DataPreprocessor:
    """Preprocess financial time series data."""

    @staticmethod
    def standardize_frequency(series: pd.Series,
                            target_freq: str,
                            method: str = 'last') -> pd.Series:
        """
        Standardize series to target frequency.

        Args:
            series: Input series
            target_freq: Target frequency ('D', 'W', 'M')
            method: Resampling method
        """
        return series.resample(target_freq).agg(method)

    @staticmethod
    def remove_outliers(series: pd.Series,
                       n_std: float = 3.0) -> pd.Series:
        """Remove outliers based on standard deviation."""
        mean = series.mean()
        std = series.std()
        return series[np.abs(series - mean) <= n_std * std]

    @staticmethod
    def fill_small_gaps(series: pd.Series,
                       max_gap: int = 5,
                       method: str = 'linear') -> pd.Series:
        """
        Fill small gaps in data.

        Args:
            series: Input series
            max_gap: Maximum gap size to fill
            method: Interpolation method
        """
        if method == 'linear':
            return series.interpolate(limit=max_gap)
        elif method == 'ffill':
            return series.fillna(method='ffill', limit=max_gap)
        elif method == 'bfill':
            return series.fillna(method='bfill', limit=max_gap)
        else:
            raise ValueError(f"Unsupported fill method: {method}")
```
