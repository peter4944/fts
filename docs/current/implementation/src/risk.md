import numpy as np
import pandas as pd
from ..core import ANNUALIZATION_FACTORS, get_annualization_factor

class RiskMetrics:
    """Calculate risk metrics."""

    @staticmethod
    def value_at_risk(returns: pd.Series,
                      confidence: float = 0.95,
                      frequency: str = 'D') -> float:
        """Calculate annualized Value at Risk."""
        ann_factor = get_annualization_factor(frequency)
        return np.percentile(returns, (1 - confidence) * 100) * ann_factor
