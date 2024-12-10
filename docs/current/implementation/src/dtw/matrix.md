# DTW Matrix Module

## 1. Overview
This module implements DTW-based correlation matrix construction:
- Pairwise DTW distance calculations
- Distance to correlation conversion
- Full correlation matrix assembly
- Matrix validation and properties

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- dtw.similarity: Core DTW calculations
- dtw.correlation: Distance to correlation conversion
- data.alignment: Series alignment preprocessing

### Related Modules
- dtw/similarity.py: Core DTW calculations
- dtw/correlation.py: Correlation conversion
- statistics/base.py: Matrix utilities
- data/alignment.py: Handles non-overlapping data alignment

### Prerequisites
Input data must be pre-aligned using one of the following methods from data/alignment.py:
- Method 1: All-overlapping (synchronized)
- Method 2: Synchronized average
- Method 3: Pairwise overlapping
- Method 4: Pairwise average

## 2. Methodology References

### Background Documents
- [DTW_to_CorrelationMatrix.md](../../../references/methodologies/DTW_to_CorrelationMatrix.md)
  * Section 4.1: Matrix construction
  * Section 4.2: Matrix properties
  * Section 4.3: Validation requirements
- [NonOverlappingData.md](../../../references/methodologies/NonOverlappingData.md)
  * Section 2: Alignment methods
  * Section 3: Implementation considerations

### Mathematical Foundations
```python
# Matrix Construction Process
# Prerequisite: Data alignment (handled by data/alignment.py)
# - Method 1: All-overlapping periods
# - Method 2: Synchronized average
# - Method 3: Pairwise overlapping
# - Method 4: Pairwise average

# Matrix Construction
1. Calculate pairwise DTW distances
2. Convert each distance to correlation
3. Ensure matrix properties:
   - Symmetry: C[i,j] = C[j,i]
   - Unit diagonal: C[i,i] = 1
   - Positive semi-definite
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def build_correlation_matrix_synchronized(returns: pd.DataFrame,
                                        window_size: Optional[int] = None) -> pd.DataFrame:
    """
    Build DTW correlation matrix from synchronized return series (Methods 1 or 2).

    Args:
        returns: DataFrame of pre-aligned return series
        window_size: Optional Sakoe-Chiba band width

    Returns:
        Correlation matrix

    Notes:
        - For use with Method 1 (all-overlapping) or Method 2 (synchronized average)
        - All series must have same length and alignment
    """
    n_series = len(returns.columns)
    corr_matrix = np.zeros((n_series, n_series))

    # Get optimal window size if not provided
    if window_size is None:
        window_size = get_window_size(_infer_frequency(returns))

    # Calculate pairwise correlations
    for i in range(n_series):
        for j in range(i+1):  # Lower triangle only
            if i == j:
                corr_matrix[i,j] = 1.0
            else:
                # Calculate DTW distance
                dist = calculate_dtw_distance(
                    returns.iloc[:,i],
                    returns.iloc[:,j],
                    window_size
                )

                # Calculate inverse distance
                dist_inv = calculate_dtw_distance(
                    returns.iloc[:,i],
                    -returns.iloc[:,j],
                    window_size
                )

                # Convert to correlation
                corr = dtw_to_correlation(dist, dist_inv)

                # Fill both triangles
                corr_matrix[i,j] = corr
                corr_matrix[j,i] = corr

    return pd.DataFrame(
        corr_matrix,
        index=returns.columns,
        columns=returns.columns
    )

def build_correlation_matrix_pairwise(pairwise_returns: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]],
                                      assets: List[str],
                                      window_size: Optional[int] = None) -> pd.DataFrame:
    """
    Build DTW correlation matrix from pairwise-aligned returns (Methods 3 or 4).

    Args:
        pairwise_returns: Dictionary mapping asset pairs to their aligned return series
            Format: {('A','B'): (series_A_aligned_to_B, series_B_aligned_to_A), ...}
        assets: List of all asset names to determine matrix order
        window_size: Optional Sakoe-Chiba band width

    Returns:
        Correlation matrix

    Notes:
        - For use with Method 3 (pairwise overlapping) or Method 4 (pairwise average)
        - Each pair can have different length
        - Same asset can have different vectors in different pairs
    """
    n_assets = len(assets)
    corr_matrix = np.zeros((n_assets, n_assets))

    # Get optimal window size if not provided
    if window_size is None:
        # Use first pair to infer frequency
        first_pair = next(iter(pairwise_returns.values()))[0]
        window_size = get_window_size(_infer_frequency(first_pair))

    # Calculate pairwise correlations
    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets[:i+1]):
            if i == j:
                corr_matrix[i,j] = 1.0
            else:
                # Get pre-aligned pair
                pair_key = (asset_i, asset_j)
                rev_pair_key = (asset_j, asset_i)

                if pair_key in pairwise_returns:
                    series_i, series_j = pairwise_returns[pair_key]
                else:
                    series_j, series_i = pairwise_returns[rev_pair_key]

                # Calculate DTW distance
                dist = calculate_dtw_distance(series_i, series_j, window_size)
                dist_inv = calculate_dtw_distance(series_i, -series_j, window_size)

                # Convert to correlation
                corr = dtw_to_correlation(dist, dist_inv)

                # Fill both triangles
                corr_matrix[i,j] = corr
                corr_matrix[j,i] = corr

    return pd.DataFrame(corr_matrix, index=assets, columns=assets)

def validate_matrix_properties(matrix: pd.DataFrame) -> None:
    """
    Validate correlation matrix properties.

    Args:
        matrix: Correlation matrix

    Raises:
        ValidationError: If matrix properties are violated

    Notes:
        Checks:
        - Symmetry
        - Unit diagonal
        - Values in [-1,1]
        - Positive semi-definite
    """
    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        raise ValidationError("Matrix is not symmetric")

    # Check unit diagonal
    if not np.allclose(np.diag(matrix), 1.0):
        raise ValidationError("Diagonal elements are not 1.0")

    # Check correlation bounds
    if (matrix < -1.0).any() or (matrix > 1.0).any():
        raise ValidationError("Correlations must be in [-1,1]")

    # Check positive semi-definite
    eigvals = np.linalg.eigvals(matrix)
    if not np.all(eigvals >= -1e-10):  # Allow for numerical error
        raise ValidationError("Matrix is not positive semi-definite")

def _validate_matrix_inputs(returns: pd.DataFrame) -> None:
    """Validate inputs for matrix construction."""
    if not isinstance(returns, pd.DataFrame):
        raise ValidationError("Returns must be a DataFrame")
    if len(returns.columns) < 2:
        raise ValidationError("Need at least 2 series for correlation matrix")

def validate_dtw_inputs(data: Union[pd.DataFrame, Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]],
                        method: str) -> None:
    """
    Validate data meets DTW requirements.

    Args:
        data: Aligned data (DataFrame or Dict)
        method: Alignment method used

    Raises:
        ValidationError: If requirements not met

    Notes:
        - Methods 1/2: All series must have same length
        - Methods 3/4: Each pair must have sufficient overlap
        - All methods: Data must be normalized
    """
    if method in ['method1', 'method2']:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError(f"Method {method} requires DataFrame")
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")
    else:  # method3 or method4
        if not isinstance(data, dict):
            raise ValidationError(f"Method {method} requires Dict")
        for pair, (series1, series2) in data.items():
            if len(series1) < 10 or len(series2) < 10:  # Example minimum length
                raise ValidationError(f"Insufficient data for pair {pair}")

    # Check normalization (example check)
    def is_normalized(series):
        return abs(series.mean()) < 0.1 and abs(series.std() - 1) < 0.1

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if not is_normalized(data[col]):
                raise ValidationError(f"Series {col} not normalized")
    else:
        for pair, (series1, series2) in data.items():
            if not is_normalized(series1) or not is_normalized(series2):
                raise ValidationError(f"Series in pair {pair} not normalized")
```

### 3.2 Performance Considerations
- Exploit matrix symmetry
- Cache normalized series
- Reuse window sizes
- Handle large matrices efficiently

### 3.3 Error Handling
```python
def _validate_matrix_inputs(returns: pd.DataFrame) -> None:
    """Validate inputs for matrix construction."""
    if not isinstance(returns, pd.DataFrame):
        raise ValidationError("Returns must be a DataFrame")
    if len(returns.columns) < 2:
        raise ValidationError("Need at least 2 series for correlation matrix")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Complete Workflow Examples
```python
# Load data
data_dict = load_csv_data(
    'data/data_inputs/bquxjob_54a84be3_193b0ccad19.csv',
    date_column='date',
    value_column='price_adjusted_usd',
    asset_column='ticker_factset'
)

# Convert to DataFrame
returns_df = pd.DataFrame(data_dict)

# Method 1: All-overlapping periods
aligned_m1 = align_series_method1(returns_df)
corr_matrix_m1 = build_correlation_matrix_synchronized(aligned_m1)

# Method 2: Synchronized average
aligned_m2 = align_series_method2(returns_df)
corr_matrix_m2 = build_correlation_matrix_synchronized(aligned_m2)

# Method 3: Pairwise overlapping
pairwise_m3 = align_series_pairwise(returns_df, method='overlap')
corr_matrix_m3 = build_correlation_matrix_pairwise(
    pairwise_m3,
    assets=returns_df.columns
)

# Method 4: Pairwise average
pairwise_m4 = align_series_pairwise(returns_df, method='average')
corr_matrix_m4 = build_correlation_matrix_pairwise(
    pairwise_m4,
    assets=returns_df.columns
)
```

#### Method Selection Guidelines
- Method 1: Maximum data quality, minimum sample size
- Method 2: Balance between quality and sample size
- Method 3: Maximum pairwise overlap
- Method 4: Maximum data utilization

#### Basic Matrix Construction
```python
# Method 1 or 2 (synchronized)
aligned_returns = align_series_synchronized(returns_df)
corr_matrix = build_correlation_matrix_synchronized(aligned_returns)

# Method 3 or 4 (pairwise)
pairwise_aligned = align_series_pairwise(returns_df)
corr_matrix = build_correlation_matrix_pairwise(
    pairwise_aligned,
    assets=returns_df.columns
)
```

### 4.2 Parameter Selection
- window_size: Based on data frequency
- validation_tolerance: For numerical stability
- memory_efficient: For large matrices

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known correlation patterns
   - Matrix properties
   - Window size effects

2. Edge Cases
   - Perfect correlation/anti-correlation
   - No correlation
   - Small number of series
   - Large number of series

3. Performance Tests
   - Large matrices
   - Memory usage
   - Computation time

## 6. Implementation Status

### Completed Features
- [x] Matrix construction
- [x] Property validation
- [x] Performance optimization
- [x] Error handling

### Known Limitations
- Memory intensive for large matrices
- No sparse matrix support
- No incremental updates

### Future Enhancements
- Sparse matrix support
- Parallel computation
- Incremental updates
- GPU acceleration
</rewritten_file>
