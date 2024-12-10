# Data Loading and Export Module

## 1. Overview
This module implements data loading and export functionality:
- CSV data import with validation
- Multiple file format support
- Data export capabilities
- Basic data structure validation

### Core Dependencies
- numpy: Numerical computations
- pandas: Data handling and I/O
- data.gaps: Gap detection
- statistics.base: Basic validation

### Related Modules
- data/gaps.py: Gap detection
- data/alignment.py: Series alignment
- statistics/base.py: Basic validation

## 2. Methodology References

### Background Documents
- [NonOverlappingData.md](../../../references/methodologies/NonOverlappingData.md)
  * Section 1.1: Data requirements
  * Section 1.5: Data validation

### Data Structure Requirements
```python
# Required CSV Structure
required_columns = {
    'date': pd.DatetimeIndex,  # Index column
    'value': float,            # Price/return data
    'asset': str,              # Asset identifier
}

# Optional Metadata
metadata = {
    'frequency': str,          # Data frequency
    'asset_class': str,        # Asset classification
    'currency': str,           # Price currency
    'adjustment': str          # Price adjustment type
}
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def load_csv_data(filepath: str,
                 date_column: str = 'date',
                 value_column: str = 'value',
                 asset_column: str = 'asset',
                 **kwargs) -> Dict[str, pd.Series]:
    """
    Load and validate CSV data.

    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        value_column: Name of value column
        asset_column: Name of asset identifier column
        **kwargs: Additional pandas read_csv arguments

    Returns:
        Dictionary mapping asset IDs to their time series

    Notes:
        - Validates data structure
        - Handles multiple assets
        - Checks data quality
    """
    # Read CSV
    df = pd.read_csv(filepath, **kwargs)

    # Validate structure
    _validate_csv_structure(df, date_column, value_column, asset_column)

    # Convert to time series
    df[date_column] = pd.to_datetime(df[date_column])
    series_dict = {}

    for asset in df[asset_column].unique():
        asset_data = df[df[asset_column] == asset]
        series = pd.Series(
            data=asset_data[value_column].values,
            index=asset_data[date_column],
            name=asset
        )
        series_dict[asset] = series

    return series_dict

def export_results(data: Dict[str, pd.Series],
                  filepath: str,
                  include_metadata: bool = True) -> None:
    """
    Export time series data to CSV.

    Args:
        data: Dictionary of time series
        filepath: Output file path
        include_metadata: Whether to include metadata

    Notes:
        - Preserves data structure
        - Includes optional metadata
        - Maintains data types
    """
    # Convert to DataFrame
    series_list = []
    for asset, series in data.items():
        df = series.to_frame(name='value')
        df['asset'] = asset
        series_list.append(df)

    result_df = pd.concat(series_list).reset_index()
    result_df.rename(columns={'index': 'date'}, inplace=True)

    # Add metadata if requested
    if include_metadata:
        result_df['frequency'] = _infer_frequency(result_df)
        result_df['data_type'] = _infer_data_type(result_df)

    # Export
    result_df.to_csv(filepath, index=False)

def validate_csv_structure(df: pd.DataFrame,
                         date_column: str,
                         value_column: str,
                         asset_column: str) -> None:
    """
    Validate CSV data structure.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        value_column: Name of value column
        asset_column: Name of asset column

    Raises:
        ValidationError: If structure is invalid

    Notes:
        - Checks column presence
        - Validates data types
        - Checks for duplicates
    """
    # Check required columns
    required = {date_column, value_column, asset_column}
    if not required.issubset(df.columns):
        raise ValidationError(f"Missing required columns: {required - set(df.columns)}")

    # Check data types
    try:
        pd.to_datetime(df[date_column])
    except:
        raise ValidationError(f"Invalid date format in {date_column}")

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValidationError(f"Non-numeric data in {value_column}")

    # Check for duplicates
    duplicates = df.groupby([date_column, asset_column]).size()
    if (duplicates > 1).any():
        raise ValidationError("Duplicate entries found")

def _infer_frequency(df: pd.DataFrame) -> str:
    """Infer data frequency from timestamps."""
    dates = pd.to_datetime(df['date'])
    return pd.infer_freq(dates) or 'Unknown'

def _infer_data_type(df: pd.DataFrame) -> str:
    """Infer whether data represents prices or returns."""
    values = df['value']
    if (values > 100).mean() > 0.5:  # Heuristic for prices
        return 'price'
    return 'return'
```

### 3.2 Performance Considerations
- Efficient CSV reading
- Memory management for large files
- Optimize data type conversions
- Handle multiple assets efficiently

### 3.3 Error Handling
```python
def _validate_filepath(filepath: str) -> None:
    """Validate file path and format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not filepath.endswith('.csv'):
        raise ValidationError("Only CSV files are supported")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Data Import
```python
# Load data with default column names
data_dict = load_csv_data('data.csv')

# Load FactSet price data
data_dict = load_csv_data(
    'data/data_inputs/bquxjob_54a84be3_193b0ccad19.csv',
    date_column='date',
    value_column='price_adjusted_usd',
    asset_column='ticker_factset'
)
```

### 4.2 Data Requirements
- CSV format
- Required columns present
- Proper data types
- No duplicate entries

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Valid CSV loading
   - Multiple assets
   - Data export

2. Edge Cases
   - Missing columns
   - Invalid data types
   - Duplicate entries
   - Empty files

3. Performance Tests
   - Large files
   - Many assets
   - Memory usage

## 6. Implementation Status

### Completed Features
- [x] CSV data loading
- [x] Data validation
- [x] Data export
- [x] Structure checking

### Known Limitations
- CSV format only
- Limited metadata support
- No streaming support

### Future Enhancements
- Additional file formats
- Streaming data support
- Enhanced metadata
- Custom validation rules
