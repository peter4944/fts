# Backfilling Shorter Time-Series Data: A Practical Approach

## Introduction

A common challenge in portfolio analysis is dealing with assets that have different lengths of historical data. For example, an emerging market ETF might have returns dating back only to 2015, while developed market indices have histories extending to the 1970s. The traditional approach of truncating all series to match the shortest history discards valuable information.

This note outlines a straightforward approach to backfilling shorter time series using information from longer-history assets. The methodology described here builds on academic work by Stambaugh (1997), Page (2013), and Jiang & Martin (2016), but focuses on practical implementation.

The key insights are:

1. For basic backfilling, simple linear regression provides optimal estimates
2. The uncertainty around these estimates can be directly characterized using regression residuals
3. Complex simulation approaches, while theoretically interesting, are typically unnecessary for practical applications

## Mathematical Framework

### Basic Backfilling Process

Consider two assets:

- A long-history asset (e.g., S&P 500) with returns spanning 1970-2023
- A short-history asset (e.g., Emerging Market ETF) with returns from 2015-2023

The backfilling process involves three steps:

1. **Establish Relationship**

   - Use the overlapping period (2015-2023) to estimate how the short-history asset relates to the long-history asset
   - This relationship is captured through ordinary least squares (OLS) regression
2. **Generate Estimates**

   - Apply the estimated relationship to the long-history asset's earlier data
   - This provides our best estimate of what the short-history asset's returns would have been
3. **Characterize Uncertainty**

   - Use regression residuals to understand the uncertainty around our estimates
   - These residuals tell us about volatility, skewness, and other distribution properties

### Mathematical Details

The regression equation is:

```
Short_Return = α + β × Long_Return + ε
```

Where:

- α (alpha) represents the average excess return of the short-history asset
- β (beta) captures the relationship between the two assets
- ε (epsilon) represents the residual or unexplained portion of returns

For practitioners, key points to understand:

- The regression automatically finds the best linear relationship between assets
- Residuals capture how much the actual returns deviate from this relationship
- The distribution of residuals tells us about the uncertainty in our estimates

## Core Algorithm and Implementation

### The Two-Step Approach

#### Step 1: Generating Backfilled Values

The backfilling process is straightforward:

1. **Perform Regression**

   - Using the overlapping period where both return series exist
   - For multiple explanatory assets, use multiple regression
   - Standard regression diagnostics (R², p-values) help assess relationship quality
2. **Calculate Predicted Values**

   - Apply regression coefficients to long-history returns
   - These predictions become our backfilled values
   - No need for complex adjustments or simulations

Example:

```
Regression Period (2015-2023):
EM_Return = 0.02 + 1.1 × SP500_Return

Backfill (2010-2014):
If SP500_Return in 2010 was 15%
Then Backfilled_EM_2010 = 0.02 + 1.1 × 15% = 18.5%
```

#### Step 2: Characterizing Uncertainty

The regression residuals provide complete information about the uncertainty in our backfilled values:

1. **Distribution Properties**

   - Standard deviation of residuals → volatility of estimates
   - Skewness of residuals → tendency for extreme values
   - Kurtosis of residuals → frequency of outliers
2. **Practical Interpretation**

   - Mean of residuals is zero (by construction)
   - Standard deviation shows typical prediction error
   - Higher moments capture non-normal behavior

Example:

```
If regression residuals have:
- Standard deviation = 3%
- Skewness = -0.5
- Kurtosis = 4.0

Then our backfilled value of 18.5% for 2010:
- Has ±3% typical variation
- Tends toward negative surprises
- Has more extreme outcomes than normal
```

### Implementation Considerations

#### Data Requirements

- Minimum overlap period: prefer at least 24 months
- Relationship stability: check for regime changes
- Data quality: adjust for corporate actions, splits

#### Model Selection

1. **Simple Cases**

   - Single explanatory asset → standard OLS
   - Stable relationships → no need for complexity
2. **Complex Cases**

   - Multiple explanatory assets → multiple regression
   - Unstable relationships → consider rolling windows
   - Non-linear relationships → consider transformations

#### Quality Control

1. **Regression Diagnostics**

   - R² > 0.3 suggests reasonable relationship
   - Significant t-statistics for coefficients
   - Well-behaved residuals (no patterns)
2. **Economic Sense**

   - Coefficients should be economically reasonable
   - Direction and magnitude of relationship
   - Consistency with market understanding

I'll outline the implementation structure for our backfill module, focusing on core functionality while keeping it at an appropriate level for a background note.

## Implementation Structure

### Module Overview

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_white
from scipy import stats

@dataclass
class RegressionResults:
    """Store regression and distribution statistics"""
    coefficients: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    residual_stats: Dict[str, float]  # std, skew, kurtosis
    analysis_timestamp: pd.Timestamp

@dataclass
class BackfillResults:
    """Store backfilled series and associated metadata"""
    synthetic_returns: pd.Series
    regression_results: RegressionResults
    target_ticker: str
    explanatory_tickers: List[str]
```

### Core Functions

```python
def backfill_series(
    target_series: pd.Series,
    explanatory_series: pd.DataFrame,
    min_overlap_periods: int = 24
) -> BackfillResults:
    """
    Main function to backfill shorter time series.
  
    Parameters:
    -----------
    target_series : pd.Series
        Series to be backfilled
    explanatory_series : pd.DataFrame
        Longer series used for prediction
    min_overlap_periods : int
        Minimum required overlap periods
      
    Returns:
    --------
    BackfillResults object containing synthetic series and metadata
    """
    # Validate inputs
    _validate_inputs(target_series, explanatory_series, min_overlap_periods)
  
    # Perform regression on overlap period
    reg_results = perform_regression(target_series, explanatory_series)
  
    # Generate synthetic returns
    synthetic_returns = generate_synthetic_returns(
        reg_results, explanatory_series)
  
    return BackfillResults(
        synthetic_returns=synthetic_returns,
        regression_results=reg_results,
        target_ticker=target_series.name,
        explanatory_tickers=list(explanatory_series.columns)
    )
```

### Supporting Functions

```python
def perform_regression(
    target: pd.Series,
    explanatory: pd.DataFrame
) -> RegressionResults:
    """
    Perform regression and calculate residual statistics.
    """
    # Align series and get overlap period
    aligned_data = pd.concat([target, explanatory], axis=1).dropna()
  
    # Run regression
    X = sm.add_constant(aligned_data[explanatory.columns])
    y = aligned_data[target.name]
    model = OLS(y, X).fit()
  
    # Calculate residual statistics
    residual_stats = {
        'std': model.resid.std(),
        'skew': stats.skew(model.resid),
        'kurt': stats.kurtosis(model.resid)
    }
  
    return RegressionResults(
        coefficients=dict(zip(X.columns, model.params)),
        t_stats=dict(zip(X.columns, model.tvalues)),
        p_values=dict(zip(X.columns, model.pvalues)),
        r_squared=model.rsquared,
        residual_stats=residual_stats,
        analysis_timestamp=pd.Timestamp.now()
    )

def generate_synthetic_returns(
    reg_results: RegressionResults,
    explanatory_data: pd.DataFrame
) -> pd.Series:
    """
    Generate synthetic returns using regression results
    """
    X = sm.add_constant(explanatory_data)
    coeffs = pd.Series(reg_results.coefficients)
  
    return pd.Series(
        np.dot(X, coeffs),
        index=explanatory_data.index
    )
```

### Usage Example

```python
# Load data
target_returns = pd.read_csv('returns.csv')['EMG.L']
explanatory_returns = pd.read_csv('returns.csv')[['SPY', 'EFA']]

# Perform backfill
results = backfill_series(
    target_returns,
    explanatory_returns,
    min_overlap_periods=24
)

# Access results
synthetic_series = results.synthetic_returns
regression_stats = results.regression_results
```

### Storage Interface

```python
def save_to_bigquery(
    backfill_results: BackfillResults,
    table_name: str,
    project_id: str
) -> None:
    """
    Save backfilled results to BigQuery using schema from appendix
    """
    # Implementation details covered in appendix
    pass
```


## Evolution of Backfilling Methodologies

The challenge of working with financial assets having different length histories has long troubled portfolio managers and analysts. Should we simply discard the longer histories to create an equal-length sample? Or is there a way to use all available information? Over the past few decades, three significant approaches have emerged to address this problem, each building upon its predecessors while seeking greater practical utility.

Robert Stambaugh first tackled this problem in his 1997 paper, demonstrating that discarding longer histories wastes valuable information. His insight was that longer histories of established markets, like the S&P 500, contain useful information about newer markets, such as emerging economy indices. Using maximum likelihood estimation, Stambaugh developed a rigorous mathematical framework for estimating the relationships between assets. While groundbreaking, his approach assumed returns followed a normal distribution – an assumption we know often fails in financial markets.

Recognizing this limitation, Sébastien Page introduced a more flexible approach in 2013. Page's method preserved Stambaugh's insights but added a simulation component to capture non-normal distributions. By generating thousands of possible scenarios, his approach could better reflect the fat tails and skewness common in financial returns. However, this came at the cost of computational complexity and introduced new questions: How many simulations are enough? How should we average results across simulations?

In 2016, Jiang and Martin made a remarkable discovery. They showed that the complexity of Page's simulation approach, while theoretically sound, was unnecessary in practice. Their insight was simple but powerful: a single set of regression residuals contains all the information we need about the uncertainty in our estimates. Rather than generating multiple scenarios, we can characterize the distribution of potential outcomes directly from these residuals.

This progression brings us to a practical conclusion for today's practitioners. For basic backfilling – creating a synthetic history for a shorter-history asset – simple ordinary least squares regression provides optimal estimates. The regression gives us our best estimate of what returns would have been, while the residuals tell us about the uncertainty in these estimates. We can directly compute volatility, skewness, and other distributional properties from these residuals without need for complex simulations.

Consider a portfolio manager needing to backfill emerging market returns before 2015. Using the Jiang and Martin insight, they would:
1. Perform a regression using the post-2015 overlap period
2. Use the regression to generate pre-2015 estimates
3. Use the regression residuals to characterize the uncertainty in these estimates

This approach provides the same information as more complex methods but with greater clarity and simplicity. The residuals from a single regression tell us everything we need to know about how much our synthetic returns might deviate from their predicted values, whether returns are likely to be skewed, and how often we might see extreme values.

The elegance of this solution lies in its simplicity. While more complex approaches might seem more sophisticated, Jiang and Martin proved that simpler methods provide equivalent results. For practitioners, this means we can focus on the economic logic of our backfilling choices – which relationships we believe are stable and meaningful – rather than getting lost in computational complexity.



## References

### Core Methodological Papers

1. Stambaugh, R. F. (1997). "Analyzing Investments Whose Histories Differ in Length." *Journal of Financial Economics*, 45(3), 285-331.
   - First rigorous treatment of the unequal histories problem
   - Establishes MLE framework for parameter estimation

2. Page, S. (2013). "How to Combine Long and Short Return Histories Efficiently." *Financial Analysts Journal*, 69(1), 45-52.
   - Extends framework to non-normal distributions
   - Introduces simulation-based approach

3. Jiang, Y., & Martin, D. (2016). "Turning Long and Short Return Histories into Equal Histories: A Better Way to Backfill Returns." 
   - Simplifies previous approaches
   - Proves sufficiency of simpler methods
   - Provides computational efficiency gains

### Additional Related Literature

4. Anderson, T. W. (1957). "Maximum Likelihood Estimates for a Multivariate Normal Distribution When Some Observations Are Missing." *Journal of the American Statistical Association*, 52(278), 200-203.
   - Foundational work on MLE with missing observations

5. Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data*. Wiley Series in Probability and Statistics.
   - Comprehensive treatment of missing data problems
   - Provides theoretical foundation for various approaches

6. Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
   - Broader context of financial time series analysis
   - Discussion of return predictability and estimation issues


## Appendix A: Data Storage Implementation

### BigQuery Schema Design for Synthetic Returns

The storage of synthetic return data presents unique challenges. We need to maintain clear separation between actual and synthetic returns while preserving all information necessary for reproducibility and analysis. The schema design below leverages BigQuery's nested structure capabilities to create an efficient and comprehensive storage solution.

#### Schema Structure

```sql
CREATE TABLE synthetic_returns (
    ticker STRING,
    return_date DATE,
    return_value FLOAT64,
    model STRUCT<
        model_id STRING,
        analysis_timestamp TIMESTAMP,
        regression_stats STRUCT<
            r2 FLOAT64,
            residual_std FLOAT64,
            residual_skew FLOAT64,
            residual_kurt FLOAT64
        >,
        explanatory_series ARRAY<
            STRUCT<
                ticker STRING,
                coefficient FLOAT64,
                t_stat FLOAT64,
                p_value FLOAT64
            >
        >
    >
)
PARTITION BY return_date
CLUSTER BY ticker
```

This schema design offers several key advantages for practical applications:

1. **Data Segregation**
By storing synthetic returns in a separate table from actual returns, we maintain data integrity while allowing easy identification of synthetic data. The model metadata stored alongside each return provides complete transparency about how the synthetic values were generated.

2. **Reproducibility**
The schema captures all necessary information to reproduce the synthetic returns:
- Coefficients and statistics from the regression model
- Identity of explanatory series used
- Time stamp of when the analysis was performed
- Complete statistical properties of the residuals

3. **Query Efficiency**
The schema leverages BigQuery's strengths through:
- Partitioning by date for efficient time-series queries
- Clustering by ticker for fast lookup of specific assets
- Nested structures to keep related data together

#### Example Queries

Inserting synthetic returns with full model information:
```sql
INSERT INTO synthetic_returns
SELECT
    'EMG.L' as ticker,
    return_date,
    return_value,
    STRUCT(
        'MODEL_20231204_001' as model_id,
        CURRENT_TIMESTAMP() as analysis_timestamp,
        STRUCT(
            0.85 as r2,
            0.02 as residual_std,
            -0.3 as residual_skew,
            3.5 as residual_kurt
        ) as regression_stats,
        [
            STRUCT(
                'SPY' as ticker,
                1.1 as coefficient,
                15.5 as t_stat,
                0.001 as p_value
            ),
            STRUCT(
                'EFA' as ticker,
                0.4 as coefficient,
                5.2 as t_stat,
                0.01 as p_value
            )
        ] as explanatory_series
    ) as model
```

Retrieving synthetic returns with their model details:
```sql
SELECT
    r.ticker,
    r.return_date,
    r.return_value,
    r.model.regression_stats.r2,
    exp.ticker as explanatory_ticker,
    exp.coefficient
FROM synthetic_returns r,
UNNEST(r.model.explanatory_series) as exp
WHERE r.ticker = 'EMG.L'
    AND r.return_date BETWEEN '2010-01-01' AND '2015-12-31'
```

### Implementation Considerations

When implementing this storage solution, consider:

1. **Data Versioning**
The model_id field can incorporate version information, allowing multiple versions of synthetic data to coexist for comparison and analysis.

2. **Audit Trail**
The analysis_timestamp field provides a crucial audit trail, helping track when synthetic data was generated and enabling investigation of any anomalies.

3. **Storage Efficiency**
The nested structure reduces redundancy while maintaining query efficiency, particularly important when storing synthetic data for many assets.

4. **Data Governance**
Clear separation of synthetic and actual returns supports data governance requirements and prevents accidental mixing of actual and synthetic data in analysis.

This storage design provides a robust foundation for managing synthetic return data while maintaining transparency and reproducibility. The schema can be extended as needed to accommodate additional metadata or analysis requirements.

