# Matrix Adjustment Methods

## Overview
Provides methods for adjusting correlation and covariance matrices to ensure desired properties.

## Classes

### MatrixAdjustment
```python
class MatrixAdjustment:
    """Base class for matrix adjustment methods."""

    @staticmethod
    def make_positive_definite(
        matrix: np.ndarray,
        min_eigenvalue: float = 1e-8,
        method: str = 'nearest'
    ) -> np.ndarray:
        """
        Adjust matrix to ensure positive definiteness.

        Args:
            matrix: Input matrix to adjust
            min_eigenvalue: Minimum eigenvalue threshold
            method: Adjustment method ('nearest', 'eigenvalue', 'diag_boost')

        Returns:
            Adjusted positive definite matrix

        Notes:
            - 'nearest': Finds nearest PSD matrix (Higham's method)
            - 'eigenvalue': Floor eigenvalues at min threshold
            - 'diag_boost': Add small constant to diagonal
        """
        pass  # Implementation to follow

class LedoitWolfShrinkage(MatrixAdjustment):
    """
    Ledoit-Wolf optimal shrinkage estimation for covariance matrices.

    Implements the Ledoit-Wolf shrinkage method (constant correlation version)
    which combines sample covariance with a structured estimator using an
    optimal shrinkage intensity.

    Uses sklearn.covariance.LedoitWolf implementation.

    Reference:
        Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
        large-dimensional covariance matrices.

    Notes:
        This implements the constant correlation target version.
        Other shrinkage methods that could be implemented:
        - Identity matrix target
        - Market factor target
        - Diagonal target
        - Industry-blocked target
        See: Ledoit & Wolf (2017) "Nonlinear Shrinkage of the Covariance Matrix
        for Portfolio Selection: Markowitz Meets Goldilocks"
    """
    def __init__(self):
        """Initialize Ledoit-Wolf shrinkage estimator."""
        from sklearn.covariance import LedoitWolf
        self._estimator = LedoitWolf()

    def fit(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        frequency: str = 'daily'
    ) -> None:
        """
        Compute optimal shrinkage parameters.

        Args:
            returns: Asset returns (n_observations, n_assets)
            frequency: Return frequency for annualization
        """
        self._estimator.fit(np.asarray(returns))

    def transform(self, sample_cov: np.ndarray) -> np.ndarray:
        """
        Apply shrinkage to sample covariance matrix.

        Args:
            sample_cov: Sample covariance matrix

        Returns:
            Shrunk covariance matrix
        """
        if not hasattr(self._estimator, 'shrinkage_'):
            raise ValueError("Must call fit() before transform()")

        return self._estimator.covariance_

    def fit_transform(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        frequency: str = 'daily'
    ) -> np.ndarray:
        """
        Compute and apply optimal shrinkage in one step.

        Args:
            returns: Asset returns
            frequency: Return frequency

        Returns:
            Shrunk covariance matrix
        """
        return self._estimator.fit_transform(np.asarray(returns))

    @property
    def shrinkage_constant(self) -> float:
        """Get the estimated optimal shrinkage constant."""
        if not hasattr(self._estimator, 'shrinkage_'):
            raise ValueError("Must call fit() first")
        return self._estimator.shrinkage_

class NearestPSD(MatrixAdjustment):
    """
    Find nearest positive semi-definite matrix.
    Based on Higham's algorithm.
    """
    pass  # Implementation to follow

class EigenvalueAdjustment(MatrixAdjustment):
    """
    Adjust matrices by modifying eigenvalues.
    """
    pass  # Implementation to follow
```

## Usage Example
```python
# Example of matrix adjustment workflow
from fts.statistics.matrix.adjustments import MatrixAdjustment, LedoitWolfShrinkage

# Basic PSD enforcement
adjusted = MatrixAdjustment.make_positive_definite(matrix, method='nearest')

# Ledoit-Wolf shrinkage
shrinkage = LedoitWolfShrinkage()
shrunk_matrix = shrinkage.fit_transform(sample_cov)

# With correlation matrix
corr_matrix = CorrelationMatrix(data)
if not corr_matrix.is_positive_definite():
    adjusted = MatrixAdjustment.make_positive_definite(
        corr_matrix.data,
        method='eigenvalue'
    )
    corr_matrix = CorrelationMatrix(adjusted)
```

## Implementation Notes
1. All methods preserve symmetry
2. Methods available for both correlation and covariance matrices
3. Different approaches for different use cases:
   - Nearest PSD: Best mathematical properties
   - Eigenvalue adjustment: Simple and fast
   - Diagonal boost: Minimal impact on structure
4. Integration with matrix classes via methods or properties
