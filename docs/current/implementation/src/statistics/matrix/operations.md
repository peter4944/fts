# Matrix Operations Module

## Overview
Provides core matrix calculation and manipulation functions for correlation and covariance matrices using composition pattern.

## Imports
```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.api import EWM

from fts.core.base import ReturnSeries
from fts.core.constants import ANNUALIZATION_FACTORS
from fts.core.validation import ValidationRules
from fts.statistics.metrics import (
    semi_volatility,
    calculate_volatility
)
```

## Classes

### MatrixData
```python
class MatrixData:
    """
    Core matrix data container and validator.

    Attributes:
        data: The underlying matrix data
        labels: Optional labels for matrix rows/columns
    """
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[List[str], pd.Index]] = None
    ):
        """
        Initialize matrix data.

        Args:
            data: Matrix data
            labels: Optional labels for dimensions

        Raises:
            ValueError: If data is not 2D square matrix
        """
        self.data = np.asarray(data)
        if isinstance(data, pd.DataFrame):
            self.labels = data.columns
        else:
            self.labels = labels
        ValidationRules.validate_matrix_structure(self.data, self.labels)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with labels."""
        if self.labels is None:
            return pd.DataFrame(self.data)
        return pd.DataFrame(self.data, index=self.labels, columns=self.labels)

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions."""
        return self.data.shape
```

### CorrelationMatrix(MatrixData):
```python
class CorrelationMatrix(MatrixData):
    """
    Class representing correlation matrices.

    Attributes:
        data: The underlying correlation matrix data
        correlation_type: Type of correlation ('pearson', 'spearman', 'kendall', 'dtw')
        volatilities: Optional annualized volatilities for covariance conversion
        frequency: Data frequency for annualization
    """
    SUPPORTED_METHODS = {
        'pearson': 'Standard linear correlation',
        'spearman': 'Rank correlation',
        'kendall': 'Kendall tau rank correlation',
        'ewma': 'Exponentially weighted moving average',
        'semi': 'Semi-correlation (downside correlation)',
        'dtw': 'Dynamic time warping correlation'
    }

    @classmethod
    def from_returns(
        cls,
        returns: pd.DataFrame,
        method: str = 'pearson',
        frequency: str = 'daily',
        ewma_halflife: Optional[float] = None,
        min_periods: Optional[int] = None
    ) -> 'CorrelationMatrix':
        """
        Compute correlation matrix from return series.

        Args:
            returns: DataFrame of return series
            method: Correlation method ('pearson', 'spearman', 'kendall', 'ewma', 'semi')
            frequency: Return frequency for annualization
            ewma_halflife: Halflife for EWMA (required if method='ewma')
            min_periods: Minimum number of observations required

        Returns:
            CorrelationMatrix instance

        Notes:
            - DTW correlation should be computed via DTWCorrelation class
            - Volatilities are always annualized
            - For EWMA, longer halflife = more smoothing

        See Also:
            - pearson: Standard linear correlation
            - spearman: Rank-based correlation, robust to outliers
            - kendall: Another rank-based measure, more robust than spearman
            - ewma: Time-weighted correlation using exponential weights
            - semi: Downside correlation considering only negative returns
        """
        if method not in cls.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported correlation method: {method}. "
                f"Supported methods: {list(cls.SUPPORTED_METHODS.keys())}"
            )

        if method == 'dtw':
            raise ValueError("DTW correlation should be computed via DTWCorrelation")

        # Calculate correlation based on method
        if method == 'ewma':
            if ewma_halflife is None:
                raise ValueError("ewma_halflife required for EWMA correlation")

            # Use statsmodels for EWMA calculation
            ewm = returns.ewm(
                halflife=ewma_halflife,
                min_periods=min_periods or 1
            )
            correlation = ewm.corr()

            # Take the final correlation matrix
            correlation = correlation.xs(
                correlation.index[-1],
                level=0,
                drop_level=True
            )

        else:  # pearson, spearman, kendall
            correlation = returns.corr(
                method=method,
                min_periods=min_periods
            )

        # Calculate annualized volatilities
        volatilities = returns.std(min_periods=min_periods) * np.sqrt(ANNUALIZATION_FACTORS[frequency])

        return cls(
            data=correlation,
            correlation_type=method,
            volatilities=volatilities,
            frequency=frequency,
            labels=correlation.columns
        )

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        method: str = 'pearson',
        frequency: str = 'daily',
        **kwargs
    ) -> 'CorrelationMatrix':
        """Convert price series to returns and calculate correlation."""
        # Use ReturnSeries for conversion
        returns = ReturnSeries.from_prices(prices)
        return cls.from_returns(returns, method, frequency, **kwargs)

    def to_covariance(self, volatilities: np.ndarray) -> 'CovarianceMatrix':
        """Convert correlation to covariance using volatilities."""
        vol_matrix = np.diag(volatilities)
        covariance = vol_matrix @ self.data @ vol_matrix
        return CovarianceMatrix(covariance, self.frequency, self.labels)

    @classmethod
    def semi_correlation(
        cls,
        returns: pd.DataFrame,
        threshold: float = 0.0,
        frequency: str = 'daily'
    ) -> 'CorrelationMatrix':
        """
        Calculate semi-correlation matrix (downside correlation).

        Args:
            returns: Return series DataFrame
            threshold: Return threshold (default 0 for downside)
            frequency: Return frequency for annualization

        Returns:
            CorrelationMatrix with semi-correlations

        Notes:
            - Only considers periods where returns < threshold
            - Useful for downside risk analysis
            - Can be converted to semi-covariance
        """
        # Get downside returns
        downside_mask = returns < threshold
        downside_returns = returns.copy()
        downside_returns[~downside_mask] = np.nan

        # Calculate semi-correlation
        correlation = downside_returns.corr(method='pearson')

        # Calculate semi-volatilities using metrics function
        semi_vols = pd.Series(
            {col: semi_volatility(returns[col], threshold, True, frequency)
             for col in returns.columns}
        )

        return cls(
            data=correlation,
            correlation_type='semi',
            volatilities=semi_vols,
            frequency=frequency,
            labels=correlation.columns
        )
```

### CovarianceMatrix(MatrixData):
```python
class CovarianceMatrix(MatrixData):
    """
    Class representing covariance matrices.

    Attributes:
        data: The underlying covariance matrix (annualized)
        frequency: Data frequency used for annualization
    """
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        frequency: str = 'daily',
        labels: Optional[Union[List[str], pd.Index]] = None
    ):
        super().__init__(data, labels)
        self.frequency = frequency
        self._validate_matrix()

    @classmethod
    def from_returns(
        cls,
        returns: pd.DataFrame,
        method: str = 'standard',
        frequency: str = 'daily',
        **kwargs
    ) -> 'CovarianceMatrix':
        """Calculate covariance matrix from returns."""
        # Use standard library methods but with centralized annualization
        covariance = returns.cov()
        ann_factor = ANNUALIZATION_FACTORS[frequency]
        covariance *= ann_factor

        return cls(
            data=covariance,
            frequency=frequency,
            labels=covariance.columns
        )

    @classmethod
    def from_correlation_and_volatilities(
        cls,
        correlation: Union[CorrelationMatrix, np.ndarray, pd.DataFrame],
        volatilities: Union[np.ndarray, pd.Series],
        frequency: str = 'daily'
    ) -> 'CovarianceMatrix':
        """
        Construct covariance matrix from correlation and volatilities.

        Args:
            correlation: Correlation matrix
            volatilities: Annualized volatilities
            frequency: Data frequency

        Returns:
            CovarianceMatrix instance

        Notes:
            - Volatilities must be annualized
            - Correlation can be matrix or CorrelationMatrix instance
        """
        if isinstance(correlation, CorrelationMatrix):
            corr_data = correlation.data
            labels = correlation.labels
        else:
            corr_data = np.asarray(correlation)
            labels = getattr(correlation, 'columns', None)

        if isinstance(volatilities, pd.Series):
            volatilities = volatilities.values

        # Convert using direct matrix multiplication
        vol_matrix = np.diag(volatilities)
        covariance = vol_matrix @ corr_data @ vol_matrix

        return cls(
            data=covariance,
            frequency=frequency,
            labels=labels
        )

    def to_correlation(self) -> Tuple[CorrelationMatrix, np.ndarray]:
        """
        Convert covariance matrix to correlation matrix.

        Returns:
            Tuple of:
            - CorrelationMatrix instance
            - Array of volatilities extracted from covariance

        Notes:
            - Volatilities are already annualized (from covariance)
            - Returns both for convenience in further calculations
        """
        # Extract volatilities from diagonal
        volatilities = np.sqrt(np.diag(self.data))

        # Create inverse volatility matrix
        inv_vol_matrix = np.diag(1 / volatilities)

        # Calculate correlation
        correlation = inv_vol_matrix @ self.data @ inv_vol_matrix

        corr_matrix = CorrelationMatrix(
            data=correlation,
            correlation_type='derived',
            volatilities=volatilities,
            frequency=self.frequency,
            labels=self.labels
        )

        return corr_matrix, volatilities

    @property
    def volatilities(self) -> np.ndarray:
        """Extract volatilities from covariance matrix."""
        return np.sqrt(np.diag(self.data))

    def _validate_matrix(self) -> None:
        """Validate covariance matrix properties."""
        ValidationRules.validate_covariance_matrix(self.data)
```

### SimilarityMatrix
```python
class SimilarityMatrix(MatrixData):
    """
    Class representing similarity matrices with conversion to correlation.

    Attributes:
        data: The underlying similarity matrix (values in [0,1])
        similarity_type: Type of similarity measure ('dtw', 'pearson', etc.)
        inverse_similarities: Optional matrix of similarities to inverse series
    """
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        similarity_type: str,
        inverse_similarities: Optional[np.ndarray] = None,
        labels: Optional[Union[List[str], pd.Index]] = None
    ):
        """
        Initialize similarity matrix.

        Args:
            data: Similarity matrix data
            similarity_type: Type of similarity measure
            inverse_similarities: Optional matrix of inverse similarities
            labels: Optional labels for dimensions

        Raises:
            ValueError: If matrix properties are invalid
        """
        super().__init__(data, labels)
        self.similarity_type = similarity_type
        if inverse_similarities is not None:
            self.inverse_similarities = np.asarray(inverse_similarities)
            self._validate_inverse_similarities()
        else:
            self.inverse_similarities = None
        self._validate_similarities()

    def _validate_similarities(self) -> None:
        """Validate similarity matrix properties."""
        if not ValidationRules.check_matrix_symmetry(self.data):
            raise ValueError("Similarity matrix must be symmetric")

        if not np.all((self.data >= 0) & (self.data <= 1)):
            raise ValueError("Similarity values must be in [0,1] range")

        if not np.allclose(np.diag(self.data), 1.0):
            raise ValueError("Similarity matrix diagonal must be 1.0")

    def _validate_inverse_similarities(self) -> None:
        """Validate inverse similarity matrix if provided."""
        if self.inverse_similarities.shape != self.data.shape:
            raise ValueError("Inverse similarities must match matrix shape")

        if not ValidationRules.check_matrix_symmetry(self.inverse_similarities):
            raise ValueError("Inverse similarity matrix must be symmetric")

        if not np.all((self.inverse_similarities >= 0) & (self.inverse_similarities <= 1)):
            raise ValueError("Inverse similarity values must be in [0,1] range")

    def to_correlation(self) -> CorrelationMatrix:
        """
        Convert similarity matrix to correlation matrix.

        Returns:
            CorrelationMatrix instance

        Notes:
            - If inverse_similarities provided, uses comparison to determine sign
            - Otherwise converts [0,1] similarities to [-1,1] correlations
        """
        if self.inverse_similarities is not None:
            correlation = np.where(
                self.inverse_similarities > self.data,
                -(2 * self.inverse_similarities - 1),  # Negative correlation
                2 * self.data - 1                      # Positive correlation
            )
        else:
            warnings.warn(
                f"Converting {self.similarity_type} similarities without "
                "inverse relationship testing"
            )
            correlation = 2 * self.data - 1

        return CorrelationMatrix(
            data=correlation,
            correlation_type=self.similarity_type,
            labels=self.labels
        )
```

## Usage Examples

### Basic Correlation Matrix
```python
# From returns
returns_df = pd.DataFrame(...)  # Your return series
corr_matrix = CorrelationMatrix.from_returns(
    returns=returns_df,
    method='pearson',
    frequency='daily'
)

# From prices
prices_df = pd.DataFrame(...)  # Your price series
corr_matrix = CorrelationMatrix.from_prices(
    prices=prices_df,
    method='spearman'
)

# Semi-correlation
semi_corr = CorrelationMatrix.semi_correlation(
    returns=returns_df,
    threshold=0.0  # Downside only
)
```

### Covariance Matrix
```python
# From returns
cov_matrix = CovarianceMatrix.from_returns(
    returns=returns_df,
    frequency='daily'
)

# From correlation and volatilities
corr_matrix = CorrelationMatrix.from_returns(returns_df)
volatilities = calculate_volatility(returns_df)
cov_matrix = CovarianceMatrix.from_correlation_and_volatilities(
    correlation=corr_matrix,
    volatilities=volatilities
)

# Convert back to correlation
corr_matrix, vols = cov_matrix.to_correlation()
print(f"Extracted volatilities: {vols}")
```

### DTW Similarity
```python
# Create similarity matrix from DTW distances
sim_matrix = SimilarityMatrix(
    data=similarities,
    similarity_type='dtw',
    inverse_similarities=inverse_sims
)

# Convert to correlation
corr_matrix = sim_matrix.to_correlation()
```

### Matrix Properties
```python
# Access underlying data
print(f"Matrix shape: {corr_matrix.shape}")
print(f"As DataFrame:\n{corr_matrix.to_dataframe()}")

# Get volatilities from covariance
vols = cov_matrix.volatilities
print(f"Annualized volatilities: {vols}")
```

### Error Handling Examples
```python
# Invalid matrix validation
try:
    # Non-symmetric matrix
    invalid_data = np.array([[1, 0.5], [0.3, 1]])
    corr_matrix = CorrelationMatrix(
        data=invalid_data,
        correlation_type='pearson'
    )
except ValueError as e:
    print(f"Validation error: {e}")  # "Matrix must be symmetric"

# Missing volatilities for covariance conversion
try:
    corr_matrix = CorrelationMatrix.from_returns(returns_df)
    cov_matrix = corr_matrix.to_covariance()  # Missing volatilities
except ValueError as e:
    print(f"Conversion error: {e}")  # "Must provide volatilities or returns"

# Invalid similarity values
try:
    invalid_sims = np.array([[1, 1.5], [1.5, 1]])  # Values > 1
    sim_matrix = SimilarityMatrix(
        data=invalid_sims,
        similarity_type='dtw'
    )
except ValueError as e:
    print(f"Similarity error: {e}")  # "Values must be in [0,1] range"
```

## Implementation Notes

### Matrix Properties
- All matrices are guaranteed symmetric
- Correlation matrices have ones on diagonal
- Values are properly bounded (-1 to 1 for correlation)
- Non-PSD matrices trigger warnings but are allowed
- Volatilities are always annualized

### Conversion Process
- Correlation ↔ Covariance uses volatility scaling
- Similarity → Correlation uses linear scaling
- DTW similarity requires inverse testing
- All conversions preserve matrix symmetry

### Performance Considerations
- Uses numpy for efficient matrix operations
- Validates only required properties
- Caches volatilities when possible
- Handles large matrices efficiently
