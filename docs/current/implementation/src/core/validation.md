# Core Validation Module

## Overview
Provides centralized validation rules and configuration for the entire library.

```python
from enum import Enum
from typing import Dict, NamedTuple, Optional, Union, List
import warnings
import numpy as np
import pandas as pd

class DataRequirements(NamedTuple):
    """Data requirements with recommended and minimum values."""
    recommended: int
    minimum: int
    description: str

class ValidationRules:
    """
    Central configuration for data validation and recommendations.

    Philosophy:
    - Allow calculations when computationally possible
    - Warn when below recommended thresholds
    - Provide clear guidance on reliability
    - Use standard errors to quantify uncertainty
    """

    # Requirements format: (recommended, minimum, description)
    OBSERVATIONS = {
        'correlation': {
            'pearson': DataRequirements(30, 3, "Pearson correlation estimation"),
            'spearman': DataRequirements(30, 3, "Spearman rank correlation"),
            'kendall': DataRequirements(30, 3, "Kendall's tau correlation"),
            'dtw': DataRequirements(60, 5, "DTW-based correlation")
        },
        'volatility': {
            'standard': DataRequirements(30, 2, "Standard deviation estimation"),
            'ewma': DataRequirements(60, 10, "EWMA volatility"),
            'garch': DataRequirements(100, 20, "GARCH model fitting")
        }
    }

    @classmethod
    def check_observations(
        cls,
        n_obs: int,
        calculation_type: str,
        method: str,
        suppress_warnings: bool = False
    ) -> bool:
        """
        Check observation count against requirements.

        Args:
            n_obs: Number of observations
            calculation_type: Type of calculation
            method: Specific method
            suppress_warnings: Whether to suppress warning messages

        Returns:
            bool: True if meets recommended threshold, False otherwise

        Notes:
            - Never prevents calculation
            - Issues warning if below recommended threshold
            - Includes reliability guidance in warning
        """
        requirements = cls.OBSERVATIONS[calculation_type][method]

        if n_obs < requirements.minimum:
            if not suppress_warnings:
                warnings.warn(
                    f"Extremely low sample size ({n_obs}) for {requirements.description}. "
                    f"Results likely unreliable. Minimum suggested: {requirements.minimum}"
                )
            return False

        if n_obs < requirements.recommended:
            if not suppress_warnings:
                warnings.warn(
                    f"Sample size ({n_obs}) below recommended ({requirements.recommended}) "
                    f"for {requirements.description}. Results may be less reliable."
                )
            return False

        return True

    @staticmethod
    def validate_matrix_structure(
        data: np.ndarray,
        labels: Optional[Union[List[str], pd.Index]] = None
    ) -> None:
        """
        Validate matrix structural properties.

        Args:
            data: Matrix data to validate
            labels: Optional dimension labels

        Raises:
            ValueError: If matrix structure is invalid
        """
        if len(data.shape) != 2:
            raise ValueError("Data must be 2-dimensional")

        if data.shape[0] != data.shape[1]:
            raise ValueError("Matrix must be square")

        if labels is not None and len(labels) != data.shape[0]:
            raise ValueError("Labels length must match matrix dimensions")

    @staticmethod
    def check_matrix_symmetry(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Check if matrix is symmetric within tolerance."""
        return np.allclose(matrix, matrix.T, atol=tolerance)

    @staticmethod
    def validate_correlation_matrix(matrix: np.ndarray, tolerance: float = 1e-8) -> None:
        """
        Validate correlation matrix properties.

        Properties:
        1. Symmetric
        2. Ones on diagonal
        3. Values between -1 and 1
        4. Positive semi-definite
        """
        if not ValidationRules.check_matrix_symmetry(matrix, tolerance):
            raise ValueError("Correlation matrix must be symmetric")

        if not np.allclose(np.diag(matrix), 1.0, atol=tolerance):
            raise ValueError("Correlation matrix diagonal must be 1.0")

        if np.any(np.abs(matrix) > 1 + tolerance):
            raise ValueError("Correlation values must be in [-1,1] range")

        eigenvals = np.linalg.eigvalsh(matrix)
        if not np.all(eigenvals >= -tolerance):
            raise ValueError("Correlation matrix must be positive semi-definite")

    @staticmethod
    def validate_covariance_matrix(matrix: np.ndarray, tolerance: float = 1e-8) -> None:
        """
        Validate covariance matrix properties.

        Properties:
        1. Symmetric (required)
        2. Non-negative diagonal (required)
        3. Positive semi-definite (warning if not)

        Notes:
            - Non-PSD matrices will trigger warning but not fail
            - Use matrix adjustment methods (shrinkage, nearest PSD) if PSD required
            - See matrix/operations.md for PSD enforcement methods
        """
        if not ValidationRules.check_matrix_symmetry(matrix, tolerance):
            raise ValueError("Covariance matrix must be symmetric")

        if np.any(np.diag(matrix) < -tolerance):
            raise ValueError("Covariance matrix diagonal must be non-negative")

        eigenvals = np.linalg.eigvalsh(matrix)
        if not np.all(eigenvals >= -tolerance):
            warnings.warn(
                "Matrix is not positive semi-definite. This may cause issues in "
                "some applications (e.g., portfolio optimization). Consider using "
                "matrix adjustment methods if PSD is required."
            )
```

## TODO: Module Validation Consistency
The following modules need to be updated to use these centralized validation rules:

1. statistics/
   - timeseries.md (correlation calculations)
   - metrics.md (statistical measures)
   - matrix/operations.md (matrix validations)

2. dtw/
   - correlation.md (sample size checks)
   - similarity.md (window size validation)

3. data/
   - alignment.md (overlap and gap handling)

Implementation Notes:
- Remove module-specific validation rules
- Use ValidationRules.check_observations consistently
- Maintain warning-only approach (no hard failures)
- Include standard errors where applicable
- Document reliability indicators in results
