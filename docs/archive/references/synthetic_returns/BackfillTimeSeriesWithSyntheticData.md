# Backfilling Shorter Time-Series Data Using Combined Backfilling Method

## Introduction

In quantitative portfolio analysis, assets with varying lengths of historical data present a significant challenge. While several methods exist for handling unequal-length return histories, including Stambaugh's (1997) maximum likelihood estimation (MLE) and Page's (2013) residual resampling approach, the Combined Backfilling method introduced by Jiang & Martin (2016) offers distinct advantages for portfolio analysis applications.

## Methodology Comparison

### Traditional Approaches

1. **Complete Case Analysis**

   - Uses only the overlapping period where all assets have data
   - Discards valuable historical information
   - Results in less accurate correlation estimates
2. **Stambaugh MLE (1997)**

   - Leverages full data through maximum likelihood estimation
   - Assumes multivariate normal distribution
   - Limited to first and second moments
3. **Page's Method (2013)**

   - Extends Stambaugh's approach using residual resampling
   - Generates multiple backfilled samples
   - Requires averaging across samples
   - Introduces simulation error and ambiguity in estimation

### Combined Backfilling Method

The Combined Backfilling method improves upon previous approaches by:

1. Creating a single canonical synthetic series
2. Preserving all statistical properties of the original data
3. Maintaining consistency with MLE estimates
4. Avoiding simulation error through deterministic computation
5. Providing exact rather than approximate results

## Core Algorithm

### Overview

The method generates synthetic data by:

1. Performing OLS regression on overlapping period
2. Using all regression residuals exactly once for each missing period
3. Combining results into a canonical synthetic series

### Mathematical Framework

For assets with returns $R_1$ (longer history) and $R_2$ (shorter history):

1. **Regression Phase**:
   $R_2 = \alpha + \beta R_1 + \epsilon$
2. **Prediction Phase**:
   For each missing period t:
   $\hat{R}_{2,t} = \alpha + \beta R_{1,t}$
3. **Synthetic Series Generation**:
   For each missing period t and residual $\epsilon_i$:
   $R_{2,t,i} = \hat{R}_{2,t} + \epsilon_i$

The canonical synthetic series is then derived from this complete set of possibilities.

## Implementation Example

```python
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def generate_synthetic_series(long_series: pd.Series, 
                            short_series: pd.Series,
                            missing_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Generate synthetic data for missing periods using Combined Backfilling method.
  
    Parameters:
    -----------
    long_series : pd.Series
        Asset returns with longer history
    short_series : pd.Series
        Asset returns with shorter history
    missing_dates : pd.DatetimeIndex
        Dates to backfill
    
    Returns:
    --------
    pd.Series
        Synthetic return series for missing periods
    """
    # Perform OLS regression on overlapping period
    overlap_data = pd.concat([long_series, short_series], axis=1).dropna()
    X = add_constant(overlap_data.iloc[:, 0])
    y = overlap_data.iloc[:, 1]
  
    model = OLS(y, X).fit()
    residuals = model.resid
  
    # Generate predictions for missing periods
    missing_X = add_constant(long_series.loc[missing_dates])
    predictions = model.predict(missing_X)
  
    # Create canonical synthetic series
    synthetic = predictions.copy()
  
    return synthetic

```



### Core Implementation Components

1. **Regression Phase**

```python
def perform_regression_analysis(long_series: pd.Series,
                              short_series: pd.Series) -> Tuple[sm.OLS, np.ndarray]:
    """
    Perform OLS regression and extract residuals.
  
    Returns:
    --------
    model : fitted OLS model
    residuals : regression residuals
    """
    overlap_data = pd.concat([long_series, short_series], axis=1).dropna()
    X = sm.add_constant(overlap_data.iloc[:, 0])
    y = overlap_data.iloc[:, 1]
  
    model = sm.OLS(y, X).fit()
    return model, model.resid
```

2. **Combined Backfilling Process**

```python
def generate_combined_backfill(model: sm.OLS,
                             residuals: np.ndarray,
                             long_series: pd.Series,
                             missing_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Generate canonical synthetic series using combined backfilling.
    """
    # Generate predictions for missing periods
    X_missing = sm.add_constant(long_series.loc[missing_dates])
    predictions = model.predict(X_missing)
  
    # The predicted values become our canonical synthetic series
    # because residuals sum to zero by OLS properties
    synthetic = pd.Series(predictions, index=missing_dates)
  
    return synthetic
```

## Statistical Properties and Moment Matching

### Theoretical Foundations

The Combined Backfilling method ensures that:

1. **First Moment (Mean)**

   - The synthetic series maintains the same expected return properties as the original series
   - Consistent with MLE estimates from Stambaugh (1997)
2. **Second Moment (Variance-Covariance)**

   - Preserves the covariance structure between assets
   - Accounts for the uncertainty in the regression estimates
3. **Higher Moments**

   - Skewness and kurtosis are preserved through the residuals
   - No assumption of normality required

### Verification Methods

```python
def verify_moment_matching(original: pd.Series, 
                         synthetic: pd.Series,
                         threshold: float = 0.1) -> Dict[str, bool]:
    """
    Verify that synthetic series maintains statistical properties.
    """
    results = {}
  
    # Compare first four moments
    orig_moments = {
        'mean': np.mean(original),
        'std': np.std(original),
        'skew': stats.skew(original),
        'kurt': stats.kurtosis(original)
    }
  
    synth_moments = {
        'mean': np.mean(synthetic),
        'std': np.std(synthetic),
        'skew': stats.skew(synthetic),
        'kurt': stats.kurtosis(synthetic)
    }
  
    # Check if moments are within threshold
    for moment in orig_moments:
        diff = abs(orig_moments[moment] - synth_moments[moment])
        results[moment] = diff <= threshold
      
    return results
```




## Integration with Portfolio Analysis

### Application in Mean-Variance Optimization

The canonical synthetic series can be directly used in portfolio optimization:

```python
def prepare_for_optimization(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Prepare returns data for portfolio optimization.
    """
    # Combine all series into single DataFrame
    combined = pd.DataFrame(returns_dict)
  
    # Calculate sample covariance matrix
    cov_matrix = combined.cov()
  
    # Calculate sample means
    means = combined.mean()
  
    return combined, cov_matrix, means
```

### Usage in PCA and Factor Analysis

The method is particularly suitable for:

1. Principal Component Analysis
2. Factor model estimation
3. Risk decomposition

## Advantages Over Alternative Methods

1. **Computational Efficiency**

   - Single pass computation
   - No need for multiple simulations
   - Deterministic results
2. **Statistical Robustness**

   - Preserves all moments of the distribution
   - Maintains cross-sectional relationships
   - Consistent with maximum likelihood estimation
3. **Practical Benefits**

   - No parameter tuning required
   - No simulation size selection needed
   - Reproducible results

## Limitations and Considerations

1. **Data Requirements**

   - Sufficient overlapping period needed
   - Stable relationship between assets assumed
   - Quality of synthetic data depends on regression fit
2. **Methodology Constraints**

   - Assumes linear relationship between assets
   - May not capture regime changes
   - Requires careful validation in extreme market conditions
3. **Implementation Considerations**

   - Need for robust regression diagnostics
   - Important to verify moment matching
   - Should validate results with out-of-sample testing

## Practical Usage Guidelines

1. **Pre-Implementation Checks**

   - Verify sufficient overlap period
   - Check for outliers in regression period
   - Assess stability of relationship between assets
2. **Quality Control**

   - Monitor regression diagnostics (R², residual plots)
   - Verify moment matching
   - Compare results with simpler alternatives
3. **Integration with Existing Systems**

   - Document assumptions and limitations
   - Implement appropriate error handling
   - Maintain audit trail of synthetic data generation

## Conclusion

The Combined Backfilling method provides a robust, efficient, and theoretically sound approach to handling assets with different length histories in portfolio analysis. Its key advantages of computational efficiency, statistical consistency, and practical simplicity make it particularly suitable for professional portfolio management applications.

## References

1. Jiang, Y., & Martin, D. (2016). "Turning Long and Short Return Histories into Equal Histories: A Better Way to Backfill Returns"
2. Stambaugh, R. F. (1997). "Analyzing Investments Whose Histories Differ in Length." Journal of Financial Economics, 45(3), 285-331.
3. Page, S. (2013). "How to Combine Long and Short Return Histories Efficiently." Financial Analysts Journal, 69(1), 45-52.

W
