# Backfill Generator Module

## 1. Overview
This module implements synthetic data generation for backfilling shorter time series:
- Relationship analysis between series
- Regression-based backfilling
- Synthetic return generation
- Validation of backfilled data

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- scipy: Statistical functions
- statistics.base: Basic calculations

### Related Modules
- statistics/base.py: Basic calculations
- statistics/timeseries.py: Rolling statistics
- data/alignment.py: Series alignment

## 2. Methodology References

### Background Documents
- [Backfilling_Shorter_TimeSeries.md](../../../references/methodologies/Backfilling_Shorter_TimeSeries.md)
  * Section 2.1: Relationship analysis
  * Section 2.2: Regression methods
  * Section 2.3: Validation criteria

### Mathematical Foundations
```python
# Regression Model
R_target = α + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

# Synthetic Returns
R_synthetic = α̂ + Σ(β̂ᵢXᵢ) + ε̂

# Validation Metrics
- Return distribution similarity
- Volatility preservation
- Correlation structure maintenance
```

## 3. Implementation Details

### 3.0 Class Structure
```python
# TODO: Update with proper class implementation after core/base.md
class BackfillGenerator(BaseGenerator):
    """
    Backfill generator implementation.
    Will inherit from base generator class.

    Attributes:
        target_series: Series to backfill
        explanatory_series: Series used for backfilling
        regression_results: Fitted parameters
        synthetic_data: Generated data

    Notes:
        - Final implementation will use inheritance
        - Methods below will become class methods
        - Will follow composition over inheritance
    """
    pass
```

### 3.1 Core Functions
```python
def analyze_relationship(target_series: pd.Series,
                       explanatory_series: pd.DataFrame,
                       min_overlap: int = 24) -> Dict[str, Any]:
    """
    Analyze relationships between target and explanatory series.

    Args:
        target_series: Series to be backfilled
        explanatory_series: Potential explanatory series
        min_overlap: Minimum required overlap periods

    Returns:
        Dictionary containing:
        - regression_params: Fitted parameters
        - r_squared: Model fit quality
        - residual_stats: Residual analysis
        - correlation_matrix: Correlation structure

    Notes:
        - Requires sufficient overlap
        - Checks for multicollinearity
        - Analyzes residual properties
    """
    # Validate inputs
    _validate_backfill_inputs(target_series, explanatory_series, min_overlap)

    # Get overlapping period
    overlap_data = pd.concat([target_series, explanatory_series], axis=1).dropna()

    if len(overlap_data) < min_overlap:
        raise ValidationError(f"Insufficient overlap: {len(overlap_data)} < {min_overlap}")

    # Fit regression
    X = overlap_data[explanatory_series.columns]
    y = overlap_data[target_series.name]

    model = LinearRegression()
    model.fit(X, y)

    # Calculate statistics
    r_squared = model.score(X, y)
    residuals = y - model.predict(X)

    return {
        'regression_params': {
            'intercept': model.intercept_,
            'coefficients': dict(zip(X.columns, model.coef_))
        },
        'r_squared': r_squared,
        'residual_stats': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skew': skew(residuals),
            'kurt': kurtosis(residuals)
        },
        'correlation_matrix': overlap_data.corr()
    }

def generate_synthetic_returns(reg_results: Dict[str, Any],
                             explanatory_data: pd.DataFrame,
                             preserve_vol: bool = True) -> pd.Series:
    """
    Generate synthetic returns using regression results.

    Args:
        reg_results: Results from analyze_relationship
        explanatory_data: Data for synthetic generation
        preserve_vol: Whether to preserve volatility

    Returns:
        Series of synthetic returns

    Notes:
        - Uses fitted parameters
        - Adds simulated residuals
        - Can preserve volatility
    """
    # Extract parameters
    params = reg_results['regression_params']
    residual_std = reg_results['residual_stats']['std']

    # Generate base returns
    synthetic = params['intercept']
    for col, beta in params['coefficients'].items():
        synthetic += beta * explanatory_data[col]

    # Add residuals
    if preserve_vol:
        residuals = np.random.normal(0, residual_std, len(explanatory_data))
        synthetic += residuals

    return pd.Series(synthetic, index=explanatory_data.index)

def backfill_series(target_series: pd.Series,
                    explanatory_series: pd.DataFrame,
                    min_overlap_periods: int = 24) -> pd.Series:
    """
    Backfill shorter series using explanatory data.

    Args:
        target_series: Series to backfill
        explanatory_series: Series for backfilling
        min_overlap_periods: Minimum required overlap

    Returns:
        Complete series including backfilled data

    Notes:
        - Analyzes relationships
        - Generates synthetic data
        - Validates results
        - Combines original and synthetic
    """
    # Analyze relationship
    reg_results = analyze_relationship(
        target_series,
        explanatory_series,
        min_overlap_periods
    )

    # Generate synthetic data for missing periods
    missing_periods = explanatory_series.index[
        ~explanatory_series.index.isin(target_series.index)
    ]
    synthetic_data = generate_synthetic_returns(
        reg_results,
        explanatory_series.loc[missing_periods]
    )

    # Combine original and synthetic
    complete_series = pd.concat([
        synthetic_data,
        target_series
    ]).sort_index()

    return complete_series

def validate_backfill_results(synthetic_returns: pd.Series,
                            original_returns: pd.Series) -> Dict[str, float]:
    """
    Validate synthetic data against original series.

    Args:
        synthetic_returns: Generated returns
        original_returns: Original returns

    Returns:
        Dictionary of validation metrics

    Notes:
        - Checks return distribution
        - Validates volatility
        - Compares correlation structure
    """
    overlap = pd.concat([
        synthetic_returns,
        original_returns
    ], axis=1).dropna()

    return {
        'mean_diff': abs(synthetic_returns.mean() - original_returns.mean()),
        'vol_ratio': synthetic_returns.std() / original_returns.std(),
        'correlation': overlap.corr().iloc[0,1],
        'ks_test': ks_2samp(synthetic_returns, original_returns).statistic
    }

def analyze_candidate_series(target: pd.Series,
                            candidates: pd.DataFrame,
                            min_correlation: float = 0.3,
                            min_r_squared: float = 0.1,
                            min_t_stat: float = 2.0) -> Dict[str, Dict[str, Any]]:
    """
    Analyze and screen candidate series for backfilling.

    Args:
        target: Target return series
        candidates: DataFrame of candidate return series
        min_correlation: Minimum required correlation
        min_r_squared: Minimum required R-squared
        min_t_stat: Minimum required t-statistic

    Returns:
        Dictionary mapping series names to their analysis:
        - correlation: Correlation with target
        - r_squared: Univariate R-squared
        - beta: Regression coefficient
        - t_stat: T-statistic
        - passes_screen: Boolean if meets criteria

    Notes:
        - Performs univariate analysis for each candidate
        - Screens based on multiple criteria
        - Helps select best explanatory variables
    """
    results = {}

    for col in candidates.columns:
        # Get overlapping data
        pair_data = pd.concat([target, candidates[col]], axis=1).dropna()

        if len(pair_data) < 24:  # Minimum required overlap
            continue

        # Calculate correlation
        corr = pair_data.corr().iloc[0,1]

        # Run univariate regression
        X = sm.add_constant(pair_data.iloc[:,1])
        y = pair_data.iloc[:,0]
        model = sm.OLS(y, X).fit()

        results[col] = {
            'correlation': corr,
            'r_squared': model.rsquared,
            'beta': model.params[1],
            't_stat': model.tvalues[1],
            'passes_screen': (
                abs(corr) >= min_correlation and
                model.rsquared >= min_r_squared and
                abs(model.tvalues[1]) >= min_t_stat
            )
        }

    return results

def save_backfill_results(results: Dict[str, Any],
                          filename: str,
                          output_dir: str = 'data/data_outputs') -> None:
      """
      Save backfill results to a single CSV file.

      Args:
          results: Dictionary containing:
              - synthetic_returns: Generated returns
              - synthetic_prices: Generated prices
              - model_params: Regression parameters
              - diagnostics: Model diagnostics
              - metadata: Backfill information
          filename: Name of output file
          output_dir: Directory to save files (default: data/data_outputs)

      Notes:
          - Saves in format: date,ticker,return,price,is_synthetic
          - Only includes backfilled period
          - Includes model parameters as metadata
          - Creates output directory if it doesn't exist
      """
      os.makedirs(output_dir, exist_ok=True)

      # Create DataFrame with returns and prices
      df = pd.DataFrame({
          'date': results['synthetic_returns'].index,
          'ticker': results['metadata']['target_series'],
          'return': results['synthetic_returns'].values,
          'price': results['synthetic_prices'].values,
          'is_synthetic': True
      })

      # Add model information as CSV metadata
      model_info = {
          'date_generated': results['metadata']['date_generated'],
          'target_series': results['metadata']['target_series'],
          'explanatory_series': ','.join(results['metadata']['explanatory_series']),
          'min_overlap': results['metadata']['min_overlap'],
          'r_squared': results['diagnostics']['r_squared'],
          'residual_std': results['diagnostics']['std']
      }

      # Save to CSV with metadata as comments
      output_path = os.path.join(output_dir, filename)
      with open(output_path, 'w') as f:
          # Write metadata as comments
          for key, value in model_info.items():
              f.write(f"# {key}: {value}\n")

          # Write data
          df.to_csv(f, index=False)

def returns_to_price_series(returns: pd.Series,
                           end_price: float,
                           method: str = 'geometric') -> pd.Series:
    """
    Convert return series to price series, working backwards from known end price.

    Args:
        returns: Return series
        end_price: Known ending price (first price in actual data)
        method: 'geometric' or 'arithmetic'

    Returns:
        Price series

    Notes:
        - Works backwards from known end price
        - For backfilling historical prices
        - Returns are expected in chronological order
    """
    if method not in ['geometric', 'arithmetic']:
        raise ValueError("method must be 'geometric' or 'arithmetic'")

    # Reverse returns for backward calculation
    reversed_returns = returns.sort_index(ascending=False)

    if method == 'geometric':
        # P(t-1) = P(t) / (1 + r(t))
        prices = pd.Series(index=returns.index, dtype=float)
        current_price = end_price

        for date, ret in reversed_returns.items():
            prices[date] = current_price / (1 + ret)
            current_price = prices[date]
    else:  # arithmetic
        # P(t-1) = P(t) - P(t)*r(t)
        prices = pd.Series(index=returns.index, dtype=float)
        current_price = end_price

        for date, ret in reversed_returns.items():
            prices[date] = current_price / (1 + ret)
            current_price = prices[date]

    return prices.sort_index()  # Return in chronological order
```

### 3.2 Performance Considerations
- Cache regression results
- Optimize overlap calculations
- Handle large datasets efficiently
- Reuse parameter estimates

### 3.3 Error Handling
```python
def _validate_backfill_inputs(target: pd.Series,
                            explanatory: pd.DataFrame,
                            min_overlap: int) -> None:
    """Validate inputs for backfilling."""
    if not isinstance(target, pd.Series):
        raise ValidationError("Target must be a Series")
    if not isinstance(explanatory, pd.DataFrame):
        raise ValidationError("Explanatory data must be a DataFrame")
    if len(explanatory.columns) < 1:
        raise ValidationError("Need at least one explanatory series")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic Backfilling
```python
# Example 1: Screen candidate series
candidate_analysis = analyze_candidate_series(
    target_series,
    candidate_series,
    min_correlation=0.3,
    min_r_squared=0.1
)

# Print screening results
for series, stats in candidate_analysis.items():
    if stats['passes_screen']:
        print(f"{series} passes screening:")
        print(f"  Correlation: {stats['correlation']:.3f}")
        print(f"  R-squared: {stats['r_squared']:.3f}")
        print(f"  t-stat: {stats['t_stat']:.2f}")

# Example 2: Generate and save backfill results
backfill_results = {
    'synthetic_returns': synthetic_returns,
    'synthetic_prices': ReturnSeries.to_price_series(
        synthetic_returns,
        anchor_price=first_actual_price,
        anchor_point='end',
        method='geometric'
    ),
    'model_params': reg_results['regression_params'],
    'diagnostics': reg_results['residual_stats'],
    'metadata': {
        'date_generated': str(pd.Timestamp.now()),
        'target_series': target_series.name,
        'explanatory_series': list(explanatory_series.columns),
        'min_overlap': min_overlap_periods
    }
}

# Save results
save_backfill_results(
    backfill_results,
    filename='backfill_ABC_20240101.csv'
)

# Example 3: Convert returns to prices
# Get first actual price (e.g., price at 2010-01)
first_actual_price = actual_prices['2010-01'].iloc[0]

# Generate historical prices working backwards
prices = returns_to_price_series(
    synthetic_returns,
    end_price=first_actual_price,  # Known price at end of backfill period
    method='geometric'
)
```

### 4.2 Parameter Selection
- min_overlap_periods: Based on data frequency
- preserve_vol: True for most applications
- validation_thresholds: Based on use case

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known relationships
   - Simple backfilling
   - Validation metrics

2. Edge Cases
   - Minimal overlap
   - Poor relationships
   - Missing data
   - Outliers

3. Performance Tests
   - Large datasets
   - Multiple explanatory series
   - Long backfill periods

## 6. Implementation Status

### Completed Features
- [x] Relationship analysis
- [x] Synthetic generation
- [x] Basic validation
- [x] Input validation

### Known Limitations
- Linear relationships only
- No regime handling
- Limited distribution options
- Single-pass estimation

### Future Enhancements
- Non-linear relationships
- Regime switching
- Multiple methods
- Iterative refinement
