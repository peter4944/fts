from typing import Dict
import numpy as np
import pandas as pd
from ..core import ANNUALIZATION_FACTORS, get_annualization_factor

class PerformanceMetrics:
    """Calculate performance metrics."""

    @staticmethod
    def annualized_return(returns: pd.Series,
                           frequency: str = 'D') -> float:
        """Calculate annualized return."""
        ann_factor = ANNUALIZATION_FACTORS[frequency]  # No sqrt needed for returns
        return (1 + returns.mean()) ** ann_factor - 1
