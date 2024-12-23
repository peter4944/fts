from typing import Dict, Optional
import numpy as np
import pandas as pd
from ..core import ANNUALIZATION_FACTORS, get_annualization_factor

class BackfillGenerator:
    """Generate synthetic historical data."""

    ANNUALIZATION_FACTORS = {
        'D': 252,    # Daily
        'W': 52,     # Weekly
        'M': 12,     # Monthly
        'Q': 4,      # Quarterly
        'A': 1       # Annual
    }
