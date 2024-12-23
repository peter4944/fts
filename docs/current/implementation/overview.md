# Implementation Overview

## 1. Documentation Organization

### 1.1 Implementation Documents
Detailed implementations are found in:

- Core implementation: [core.md](../src/core.md)
- Data handling: [data.md](../src/data.md)
- Statistical analysis: [statistics.md](../src/statistics.md)
- DTW implementation: [dtw.md](../src/dtw.md)
- Volatility models: [volatility.md](../src/volatility.md)
- Backfill functionality: [backfill.md](../src/backfill.md)
- Utilities: [utils.md](../src/utils.md)
- Validation framework: [validation.md](../src/validation.md)
- Error handling: [errors.md](../src/errors.md)

### 1.2 Project Documentation
- Requirements: [user_requirements.md](../../requirements/user_requirements.md)
- Gap Analysis: [gap_analysis.md](../../requirements/gap_analysis.md)
- System Design: [system_design.md](../../design/system_design.md)
- Traceability Matrix: [requirements_traceability.md](../../requirements/requirements_traceability.md)
- Background Methodologies: [methodologies/](../../references/methodologies/)

### 1.3 Background Methodologies
#### Core Methodologies
- DTW to Correlation Matrix: [DTW_to_CorrelationMatrix.md](../../references/methodologies/DTW_to_CorrelationMatrix.md)
- Non-overlapping Data Handling: [NonOverlappingData.md](../../references/methodologies/NonOverlappingData.md)
- Volatility Forecasting (HAR/GARCH): [VolatilityForecasting_HAR_GARCH.md](../../references/methodologies/VolatilityForecasting_HAR_GARCH.md)
- PCA Factor Loading Time Series: [PCA_FactorLoadingTimeSeries.md](../../references/methodologies/PCA_FactorLoadingTimeSeries.md)

#### Statistical Methods
- Arithmetic to Geometric Returns: [Arithmetic_to_Geometric_returns.md](../../references/methodologies/Arithmetic_to_Geometric_returns.md)
- Shrinking Covariance Matrix: [Shrinking_Covariance_Matrix.md](../../references/methodologies/Shrinking_Covariance_Matrix.md)
- Backfilling Shorter Time Series: [Backfilling_Shorter_TimeSeries.md](../../references/methodologies/Backfilling_Shorter_TimeSeries.md)

#### Implementation Approaches
- Memory Management for Large Datasets: [Memory_Management_LargeDatasets.md](../../references/methodologies/Memory_Management_LargeDatasets.md)
- Parallel Processing Strategies: [Parallel_Processing_Strategies.md](../../references/methodologies/Parallel_Processing_Strategies.md)
- Validation Framework Design: [Validation_Framework_Design.md](../../references/methodologies/Validation_Framework_Design.md)

## 2. Project Structure

### 2.1 Directory Structure
```
project_root/
├── pyproject.toml           # Project configuration
├── README.md               # Project documentation
├── CHANGELOG.md           # Version history
├── LICENSE               # License information
├── .gitignore           # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks
├── src/
│   └── fts/              # Main package
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py        # Base classes, constants
│       │   ├── validation.py  # Input validation
│       │   ├── errors.py      # Error handling
│       │   └── transformations/
│       │       ├── __init__.py
│       │       └── returns.py  # Return series transformations
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py      # Data import/export
│       │   ├── alignment.py   # Non-overlapping data handling
│       │   └── gaps.py        # Gap detection and handling
│       ├── statistics/
│       │   ├── __init__.py
│       │   ├── metrics.py     # Statistical metrics
│       │   ├── timeseries.py  # Time series operations
│       │   └── matrix/        # Matrix operations
│       │       ├── __init__.py
│       │       ├── operations.md  # Basic matrix calculations and classes
│       │       └── adjustments.md # Matrix adjustments (including shrinkage)
│       ├── distribution/
│       │   ├── __init__.py
│       │   └── skew_student_t.py  # Skewed-t distribution fitting and calculations
│       ├── dtw/
│       │   ├── __init__.py
│       │   ├── similarity.py   # DTW calculations
│       │   └── correlation.py  # DTW-specific correlation
│       ├── volatility/
│       │   ├── __init__.py
│       │   ├── garch.py      # GARCH modeling
│       │   └── har.py        # HAR modeling
│       ├── backfill/
│       │   ├── __init__.py
│       │   └── generator.py   # Synthetic data
├── tests/                    # Test directory (mirrors src structure)
│   └─ fts/                  # Test package
│       ├── core/            # Core tests
│       │   ├── __init__.py
│       │   ├── test_base.py
│       │   ├── test_validation.py
│       │   └── test_errors.py
│       │   └── transformations/
│       │       ├── __init__.py
│       │       └── test_returns.py
│       ├── data/           # Data tests
│       │   ├── test_loader.py
│       │   ├── test_alignment.py
│       │   └── test_gaps.py
│       ├── statistics/     # Statistics tests
│       │   ├── test_metrics.py
│       │   ├── test_timeseries.py
│       │   └── matrix/
│       │       ├── test_operations.py
│       │       └── test_adjustments.py
│       ├── distribution/   # Distribution tests
│       │   └── test_skew_student_t.py
│       ├── dtw/           # DTW tests
│       │   ├── test_similarity.py
│       │   ├── test_correlation.py
│       ├── volatility/    # Volatility tests
│       │   ├── test_garch.py
│       │   └── test_har.py
│       ├── backfill/      # Backfill tests
│       │   └── test_generator.py
├── docs/                     # Documentation
│   ├── current/              # Current active documentation
│   │   ├── requirements/    # Requirements documentation
│   │   │   ├── user_requirements.md
│   │   │   ├── gap_analysis.md
│   │   │   └── requirements_traceability.md
│   │   ├── design/         # System design documentation
│   │   │   ├── system_design.md
│   │   │   └── architecture/
│   │   │       └── diagrams/
│   │   ├── implementation/ # Implementation documentation
│   │   │   ├── overview.md # This document
│   │   │   ├── src/       # Module implementations
│   │   │   │   ├── core/
│   │   │   │   │   ├── base.md
│   │   │   │   │   ├── validation.md
│   │   │   │   │   └── errors.md
│   │   │   │   ├── transformations/
│   │   │   │   │   └── returns.md
│   │   │   │   ├── data/
│   │   │   │   │   ├── loader.md
│   │   │   │   │   ├── alignment.md
│   │   │   │   │   └── gaps.md
│   │   │   │   ├── statistics/
│   │   │   │   │   ├── metrics.md
│   │   │   │   │   ├── timeseries.md
│   │   │   │   │   └── matrix/
│   │   │   │   │       ├── operations.md
│   │   │   │   │       └── adjustments.md
│   │   │   │   ├── distribution/
│   │   │   │   │   └── skew_student_t.md
│   │   │   │   ├── dtw/
│   │   │   │   │   ├── similarity.md
│   │   │   │   ��  ├── correlation.md
│   │   │   │   ├── volatility/
│   │   │   │   │   ├── garch.md
│   │   │   │   │   └── har.md
│   │   │   │   ├── backfill/
│   │   │   │   │   └── generator.md
│   │   └── references/    # Reference materials
│   │       ├── academic_papers/
│   │       ├── methodologies/
│   │       └── external/
│   ├── archive/           # Previous versions
│   │   ├── requirements/
│   │   │   ├── v1/
│   │   │   └── v2/
│   │   ├── design/
│   │   └── implementation/
│   └── templates/         # Documentation templates
└── config/                  # Configuration files
```

### 2.2 Module Function Mapping

#todo: add function for standardise returns as excess returns, relative to adj or standard volatilitu
# todo: ad function to standardise return for factors

#### Core Module (core/)
##### base.py
TimeSeries class
  * __init__(data: pd.Series, metadata: Optional[Dict] = None)
  * validate()
  * align_with(other: TimeSeries) -> Tuple[TimeSeries, TimeSeries]
ReturnSeries class (inherits from TimeSeries)
  * from_price_series(prices: pd.Series, geometric: bool = True)
  * standardize() -> ReturnSeries
TimeSeriesCollection class
  * __init__(series: Dict[str, TimeSeries])
  * align(method: str) -> TimeSeriesCollection

##### validation.py
validate_returns(returns: pd.Series) -> None
validate_parameters(params: Dict[str, Any]) -> None
validate_frequency(frequency: str) -> None
validate_alignment(series1: pd.Series, series2: pd.Series) -> None

##### errors.py
FTSError (base exception)
ValidationError
ProcessingError
ConfigurationError

#### Data Module (data/)
##### loader.py
load_csv_data(filepath: str, date_column: str = 'Date') -> Dict[str, TimeSeries]
export_results(data: TimeSeriesCollection, filepath: str) -> None
validate_csv_structure(df: pd.DataFrame) -> None

##### alignment.py
- align_series(series1: pd.Series, series2: pd.Series, method: str) -> Tuple[pd.Series, pd.Series]
- handle_non_overlapping_data(series: List[pd.Series]) -> pd.DataFrame
- pairwise_alignment(series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]
- synchronized_average_alignment(series: List[pd.Series]) -> pd.DataFrame

##### gaps.py
- detect_gaps(series: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]
- analyze_gap_patterns(series: pd.Series) -> Dict[str, Any]
- handle_missing_data(series: pd.Series, method: str) -> pd.Series
- fill_small_gaps(series: pd.Series, max_gap: int = 5) -> pd.Series

#### Statistics Module (statistics/)
##### metrics.py
- mean_return(returns: pd.Series, geometric: bool = False) -> float
- stdev(returns: pd.Series, annualized: bool = True) -> float
- skewness(returns: pd.Series) -> float
- kurtosis(returns: pd.Series) -> float
- correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame
- covariance_matrix(returns: pd.DataFrame, annualized: bool = True) -> pd.DataFrame
- normalize_returns(returns: pd.Series) -> pd.Series
- semi_volatility(returns: pd.Series, threshold: float = 0.0) -> float

##### matrix/operations.py
MatrixData class
  * __init__(data: Union[np.ndarray, pd.DataFrame], labels: Optional[List[str]])
  * to_dataframe() -> pd.DataFrame
  * shape property -> Tuple[int, int]

CorrelationMatrix class
  * from_returns(returns: pd.DataFrame, method: str) -> CorrelationMatrix
  * from_prices(prices: pd.DataFrame, method: str) -> CorrelationMatrix
  * to_covariance(volatilities: np.ndarray) -> CovarianceMatrix
  * semi_correlation(returns: pd.DataFrame) -> CorrelationMatrix

CovarianceMatrix class
  * from_returns(returns: pd.DataFrame) -> CovarianceMatrix
  * from_correlation_and_volatilities(...) -> CovarianceMatrix
  * to_correlation() -> Tuple[CorrelationMatrix, np.ndarray]
  * volatilities property -> np.ndarray

SimilarityMatrix class
  * __init__(data: np.ndarray, similarity_type: str)
  * to_correlation() -> CorrelationMatrix

##### matrix/adjustments.py
LedoitWolfShrinkage class
  * fit(returns: np.ndarray) -> None
  * transform(sample_cov: np.ndarray) -> np.ndarray
  * fit_transform(returns: np.ndarray) -> np.ndarray
  * shrinkage_constant property -> float

NearestPSD class
  * make_positive_definite(matrix: np.ndarray) -> np.ndarray

EigenvalueAdjustment class
  * adjust_eigenvalues(matrix: np.ndarray) -> np.ndarray

##### timeseries.py
- rolling_statistics(returns: pd.Series, window: int) -> pd.DataFrame
- rolling_correlation(returns1: pd.Series, returns2: pd.Series, window: int) -> pd.Series
- rolling_beta(returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series
- drawdown_series(returns: pd.Series) -> pd.Series
- pca_factor_returns(returns: pd.DataFrame, n_factors: int) -> pd.DataFrame

##### adjusted.py
- variance_drag(volatility: float) -> float
- kurtosis_drag(kurtosis: float, volatility: float) -> float
- skew_drag(skewness: float, volatility: float) -> float
- geometric_return(arithmetic_return: float, volatility: float) -> float
- kelly_fraction(returns: pd.Series, distribution: str = 'normal') -> float
- probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float) -> float
- deflated_sharpe_ratio(returns: pd.Series, trials: int) -> float
- max_theoretical_drawdown(sharpe_ratio: float, distribution: str = 'normal') -> float
- adj_geometric_return(returns: pd.Series, include_higher_moments: bool = True) -> float
- adj_volatility(returns: pd.Series, include_higher_moments: bool = True) -> float
- adj_geometric_sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0, include_higher_moments: bool = True) -> float

#### Distribution Module (distribution/)
##### skew_student_t.py
- fit_skewed_t(returns: pd.Series) -> Dict[str, float]
- calculate_moments(params: Dict[str, float]) -> Dict[str, float]
- distribution_test(returns: pd.Series) -> Dict[str, float]
- calculate_student_t_geometric_return(arithmetic_return: float, vol: float, params: Dict[str, float]) -> float
- calculate_student_t_drag(params: Dict[str, float], vol: float) -> float
- calculate_student_t_heavy_tail_drag(params: Dict[str, float], vol: float) -> float
- calculate_student_t_kurtosis_drag(params: Dict[str, float], vol: float) -> float
- calculate_student_t_skew_drag(params: Dict[str, float], vol: float) -> float
- calculate_student_t_volatility(params: Dict[str, float]) -> float
- calculate_student_t_sharpe_ratio(returns: pd.Series, params: Dict[str, float], rf_rate: float = 0.0) -> float
- calculate_student_t_mtd(vol_target: float, sr_adj: float, params: Dict[str, float], lambda_param: float = 0.2) -> float

#### DTW Module (dtw/)
##### similarity.py
- calculate_dtw_distance(series1: pd.Series, series2: pd.Series, window_size: int) -> float
- normalize_series(series: pd.Series) -> pd.Series
- get_window_size(frequency: str) -> int

##### correlation.py
- dtw_to_correlation(similarity: float, inverse_similarity: float) -> float
- handle_negative_correlation(series1: pd.Series, series2: pd.Series) -> float

##### matrix.py
- build_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame
- validate_matrix_properties(matrix: pd.DataFrame) -> None

#### Volatility Module (volatility/)
##### garch.py
- fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> Dict[str, Any]
- forecast_garch_path(returns: pd.Series, horizon: int) -> pd.Series
- forecast_garch_window(returns: pd.Series, window: int, horizon: int) -> pd.Series
- calculate_persistence() -> float
- calculate_long_term_volatility() -> float

##### har.py
- calculate_har_components(returns: pd.Series) -> Dict[str, pd.Series]
- fit_har_model(rv_components: Dict[str, pd.Series]) -> Dict[str, Any]
- forecast_har_window(returns: pd.Series, window: int, horizon: int) -> pd.Series
- calculate_har_residuals(fitted_model: Dict[str, Any]) -> pd.Series

#### Backfill Module (backfill/)
##### generator.py
- analyze_relationship(target_series: pd.Series,
                      explanatory_series: pd.DataFrame) -> Dict[str, Any]
- analyze_candidate_series(target: pd.Series,
                           candidates: pd.DataFrame) -> Dict[str, Dict]
- generate_synthetic_returns(reg_results: Dict[str, Any],
                               explanatory_data: pd.DataFrame) -> pd.Series
- backfill_series(target_series: pd.Series,
                 explanatory_series: pd.DataFrame) -> pd.Series
- validate_backfill_results(synthetic_returns: pd.Series,
                          original_returns: pd.Series) -> Dict[str, float]
- save_backfill_results(results: Dict[str, Any],
                       output_dir: str) -> None
- returns_to_price_series(returns: pd.Series,
                         start_price: float) -> pd.Series

#### Covariance Module (covariance/)
##### shrinkage.py
- ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray
- calculate_shrinkage_constant(sample_cov: np.ndarray,
                             returns: pd.DataFrame) -> float
- calculate_target_matrix(sample_cov: np.ndarray) -> np.ndarray
- validate_covariance_matrix(matrix: np.ndarray) -> bool

### 2.3 Project Configuration

#### Core Configuration
```toml
[project]
name = "fts-library"
version = "1.0.0"
description = "Financial Time Series Analysis Library"
requires-python = ">=3.8"
license = "MIT"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "Topic :: Office/Business :: Financial",
    "Programming Language :: Python :: 3.8",
]

[project.dependencies]
numpy = ">=1.21.0"
pandas = ">=1.3.0"
scipy = ">=1.7.0"
scikit-learn = ">=1.0.0"
statsmodels = ">=0.13.0"
arch = ">=4.19.0"
fastdtw = ">=0.3.0"

[project.urls]
Documentation = "https://github.com/username/fts-library/docs"
Source = "https://github.com/username/fts-library"

[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"
```

#### Development Configuration
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "mypy>=0.950",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "pre-commit>=2.17.0"
]

[tool.pdm]
python_requires = ">=3.8"

[tool.pdm.scripts]
test = "pytest tests/"
lint = "pre-commit run --all-files"
format = "black ."
typecheck = "mypy src/"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
multi-line-output = 3

[tool.mypy]
python_version = "3.8"
strict = true
ignore_missing_imports = true
```

#### Tool Configurations
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
```

## 3. Module Architecture and Relationships

### 3.1 Class Hierarchy
```mermaid
classDiagram
    class TimeSeries {
        +data: pd.Series
        +metadata: Dict
        +validate()
    }
    class ReturnSeries {
        +method: ReturnSeriesMethod
        +align_with(other)
        +standardize()
    }
    class TimeSeriesCollection {
        +series: Dict[str, TimeSeries]
        +align_all()
    }
    class ValidationResult {
        +is_valid: bool
        +errors: List[str]
        +warnings: List[str]
        +metrics: Dict
    }
    class DTWParameters {
        +window_size: int
        +frequency: str
        +distance_metric: str
    }
    class VolatilityModel {
        +fit(returns)
        +predict()
    }

    TimeSeries <|-- ReturnSeries : inherits
    TimeSeriesCollection o-- TimeSeries : contains
    ReturnSeries --> ValidationResult : uses
    DTWParameters --> ValidationResult : uses
    VolatilityModel --> ReturnSeries : uses
```

### 3.2 Module Dependencies
```mermaid
graph TD
    subgraph Statistics ["Statistics Layer"]
        Metrics[metrics.py] --> MatrixOps[matrix/operations.py]
        MatrixOps --> MatrixAdj[matrix/adjustments.py]
    end

    subgraph DTW ["DTW Layer"]
        DTWSim[similarity.py] --> DTWCorr[correlation.py]
        DTWCorr --> MatrixOps
    end

    subgraph Risk ["Risk Layer"]
        RiskCalc[calculations.py] --> MatrixOps
        RiskCalc --> MatrixAdj
    end

    DTW --> Statistics
    Risk --> Statistics
```

### 3.3 Key Interactions

1. **Data Flow**
   - Import → Validation → Processing → Analysis → Export

2. **Matrix Operations Flow**
   - Basic calculations (statistics/matrix/operations.py)
     * Correlation/covariance matrix computation
     * Matrix conversions and transformations
   - Adjustments (statistics/matrix/adjustments.py)
     * PSD enforcement
     * Shrinkage estimation
     * Matrix corrections

3. **Validation Chain**
   - Data validation
   - Matrix validation  # Moved to matrix/validation.py
   - Parameter validation
   - Result validation

## 4. Development Standards and Core Components

### 4.1 Development Standards
- Python 3.8+
- Type hints required
- Documentation in Google style
- Unit test coverage > 80%
- Pre-commit hooks for quality

### 4.2 Core Components

#### Base Classes
- **TimeSeries**: Base container for time series data
  - Data storage and validation
  - Metadata management
  - Basic operations

- **ReturnSeries**: Specialized container for return calculations
  - Return series alignment
  - Statistical calculations
  - Standardization methods

- **TimeSeriesCollection**: Container for multiple series
  - Series management
  - Bulk operations
  - Alignment strategies

#### Analysis Components
- **Statistical Analysis**
  - Moment calculations
  - Distribution fitting
  - Correlation analysis

- **DTW Implementation**
  - Window size optimization
  - Distance calculations
  - Correlation matrix building

- **Volatility Models**
  - HAR model implementation
  - GARCH model fitting
  - Realized volatility calculations

- **Backfill Module**
  - OLS regression
  - Synthetic data generation
  - Validation framework

#### Infrastructure
- **Memory Management**
  - Chunked processing
  - Cache management
  - Memory monitoring

- **Validation Framework**
  - Data validation
  - Parameter validation
  - Result validation

## 5. Testing Strategy

### 5.1 Testing Approach
- Unit tests for all components
- Integration tests for module interactions
- Performance tests for critical paths
- Validation tests for data quality

### 5.2 Test Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for core data structures
- 100% coverage for validation framework
- Critical path testing for all modules

### 5.3 Test Organization
Tests mirror the source structure:
- Unit tests per module
- Integration tests for module interactions
- Performance benchmarks
- Data quality validation suites

## 6. Data Handling Requirements

### 6.1 Return Series Generation
The library supports multiple methods for handling non-overlapping data:

#### Basic Return Series Methods
- ALL_INCLUDING_GAPS: Include all periods, allowing NaN values per series
- ALL_OVERLAPPING: Only periods where all series have valid data

#### Advanced Alignment Methods
- SYNCHRONIZED_AVERAGE: Distribute returns across gaps
- PAIRWISE_OVERLAPPING: Use overlapping periods per pair
- PAIRWISE_AVERAGE: Distribute returns across gaps per pair

### 6.2 Data Quality Standards

#### Minimum Sample Requirements
- Basic Statistics: 20 observations
- Correlation Analysis: 30 overlapping observations
- Simple Regression: 30 observations
- Multiple Regression: 30 + 10 × (n_variables) observations
- PCA Analysis: 50 observations minimum
- Volatility Forecasting: 60 observations minimum

#### Missing Value Treatment
1. Time Series Alignment
   - Include all dates where at least one series has observation
   - NaN for non-trading days/missing observations
   - No forward/backward filling by default

2. Analysis-Specific Handling
   - Return Calculations: Use available points only
   - Correlation: Pairwise complete observations only
   - Regression: Complete cases only
   - PCA: Complete cases only

#### Warning Thresholds
- Data gaps > 5 consecutive days
- Missing data pattern analysis if >5% missing
- Series length discrepancy warnings if >20%
- Return size warnings for values >|20%|

### 6.3 Input Data Requirements
- Prices must be adjusted close prices
- Adjustments for dividends and corporate actions must be pre-applied
- Zero or negative prices will trigger validation errors
- Missing data points are allowed for non-trading periods
- Calculations use all available actual data points
- No automatic forward/backward filling of missing data

## 7. Return Calculations and Conventions

### 7.1 Return Types
- Simple Returns: r_t = (P_t - P_{t-1})/P_{t-1}
- Log Returns: r_t = ln(P_t/P_{t-1})
- Default: Log returns unless explicitly specified

### 7.2 Conversion Rules
- Simple to Log: r_log = ln(1 + r_simple)
- Log to Simple: r_simple = exp(r_log) - 1
- Conversion warnings for large returns (>|20%|)

### 7.3 Calculation Conventions
- All volatilities are annualized
- All Sharpe ratios and performance metrics are annualized
- Default frequency is daily
- Distribution fitting is for parameter estimation only
- Series statistics use full available data
- Correlation analysis uses overlapping periods only
- Standard frequency conversions use last-day-of-period

### 7.4 Risk-Free Rate Specifications
- Format: Annualized 1-month rate as decimal (e.g., 0.05 for 5%)
- Input Options:
  * As separate series aligned with return dates
  * As constant value for entire period
- Default: 0 if not provided
- Required for:
  * Sharpe ratio calculations
  * Excess return calculations
  * Risk-adjusted performance metrics

## 8. Performance Requirements

### 8.1 Processing Time Limits
| Operation | Data Size | Max Time |
|-----------|-----------|----------|
| CSV Import | 1GB | 30s |
| Statistical Analysis | 1M rows | 60s |
| Backfill Generation | 10 years daily | 120s |

### 8.2 Memory Usage Limits
| Operation | Max Memory |
|-----------|------------|
| Base Usage | System Memory Dependent |
| Peak Usage | Up to 75% of System Memory |
| Large Dataset | Up to 90% of System Memory |

### 8.3 Computation Optimization
Priority order for optimization:
1. Vectorized operations using numpy/pandas
2. Numba-accelerated functions for heavy computations
3. GPU acceleration for large matrix operations where available

### 8.4 Memory Management Strategy
- Chunked processing for large datasets
- Cache management with configurable limits
- Memory monitoring and cleanup
- Efficient data structure usage

## 9. Error Handling Strategy

### 9.1 Error Categories
```python
class FTSError(Exception):
    """Base exception for all FTS errors."""
    pass

class ValidationError(FTSError):
    """Data validation errors."""
    pass

class ProcessingError(FTSError):
    """Data processing errors."""
    pass

class ConfigurationError(FTSError):
    """Configuration related errors."""
    pass
```

### 9.2 Error Handling Principles
- Early validation to catch errors at input
- Explicit error messages with context
- Recovery procedures where possible
- Graceful degradation for non-critical errors

### 9.3 Validation Framework
- Input data validation
- Parameter validation
- Result validation
- Configuration validation

### 9.4 Logging Strategy
- ERROR: All exceptions and errors
- WARNING: Data quality issues, performance warnings
- INFO: Major processing steps
- DEBUG: Detailed processing information

## Function Registry and Implementation Map

### Legend
[AG]: ../../references/methodologies/Arithmetic_to_Geometric_returns.md
[VF]: ../../references/methodologies/VolatilityForecasting_HAR_GARCH.md
[BF]: ../../references/methodologies/Backfilling_Shorter_TimeSeries.md
[DTW]: ../../references/methodologies/DTW_to_CorrelationMatrix.md
[PCA]: ../../references/methodologies/PCA_FactorLoadingTimeSeries.md
[SC]: ../../references/methodologies/Shrinking_Covariance_Matrix.md

**Source:**
- UR: Original User Requirements
- DS: Added in Design Stage
- IP: Added in Implementation Planning
- BN: Added from Background Notes
- ID: Added in Implementation Discussion

### Core Functions
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| get_annualization_factor | DS | core/constants.py | ✓ | Central annualization handling | AG |
| validate_returns | UR | core/validation.py | ✓ | Basic data validation | - |

### Statistical Measures
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| ret_mean | UR | statistics/metrics.py | ✓ | Basic return measure | - |
| [variance_drag][AG#2.1] | BN | statistics/returns.py | ✓ | From methodology | AG#2.1 |
| [kurtosis_drag][AG#2.1] | BN | statistics/returns.py | ✓ | From methodology | AG#2.1 |
| [skew_drag][AG#2.1] | BN | statistics/returns.py | ✓ | From methodology | AG#2.1 |
| [ledoit_wolf_shrinkage][SC#2] | BN | statistics/covariance.py | ✓ | Covariance denoising | SC#2 |

### Performance Metrics
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| standard_sharpe_ratio | UR | performance/sharpe.py | ✓ | Basic Sharpe ratio | AG#3.1 |
| [geometric_sharpe_ratio][AG#3.1] | BN | statistics/returns.py | ✓ | With variance drag | AG#3.1 |
| [probabilistic_sharpe_ratio][AG#3.2] | BN | performance/sharpe.py | ✓ | Statistical significance | AG#3.2 |
| [deflated_sharpe_ratio][AG#3.3] | BN | performance/sharpe.py | ✓ | Multiple testing adjustment | AG#3.3 |
| [kelly_fraction_normal][AG#4.1] | BN | performance/kelly.py | ✓ | Basic Kelly criterion | AG#4.1 |
| [kelly_fraction_student_t][AG#4.2] | BN | performance/kelly.py | ✓ | Heavy-tail adjustment | AG#4.2 |

### Volatility Models
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| [realized_volatility][VF#2] | UR | volatility/realized.py | ✓ | Basic volatility measure | VF#2 |
| [garch_fit][VF#1] | UR | volatility/garch.py | ✓ | GARCH(1,1) fitting | VF#1 |
| [forecast_garch_path][VF#1.3] | BN | volatility/garch.py | ✓ | Day-by-day volatility path | VF#1.3 |
| [forecast_garch_window][VF#1.4] | ID | volatility/garch.py | ✓ | Rolling window from path | VF#1.4 |
| [har_components][VF#2] | UR | volatility/har.py | ✓ | RV components | VF#2 |
| [har_fit][VF#2] | UR | volatility/har.py | ✓ | HAR model fitting | VF#2 |
| [forecast_har_window][VF#2] | UR | volatility/har.py | ✓ | Window-aligned HAR forecast | VF#2 |
| [calculate_har_components][VF#2] | UR | volatility/har.py | ✓ | Cross-ref: [VF#2] |

### DTW Analysis
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| [calculate_dtw_similarity][DTW#3] | BN | dtw/similarity.py | ✓ | DTW-based correlation | DTW#3 |
| [build_dtw_correlation_matrix][DTW#3] | BN | dtw/correlation.py | ✓ | Full correlation matrix | DTW#3 |

### PCA Analysis
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| calculate_factor_loadings | BN | pca/loadings.py | ✓ | PCA factor loadings | PCA#3 |
| generate_factor_scores | BN | pca/scores.py | ✓ | Time series of loadings | PCA#6 |

### Backfilling
| Function | Source | Module | Status | Notes | Reference |
|----------|---------|---------|---------|-------|-----------|
| backfill_series | BN | backfill/generator.py | ✓ | Basic backfilling | BF#2 |
| regression_backfill | BN | backfill/regression.py | ✓ | Regression-based | BF#3 |

### Status Legend
- ✅: Implemented
- 🚧: In progress
- ❌: Not started
- ⚠️: Needs review

### Original User Requirements Status
#### 3.0 Data Handling Requirements
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| validate_returns | validate_returns | core/validation.py | ✓ | Basic validation |
| validate_parameters | validate_parameters | core/validation.py | ✓ | Parameter checks |
| validate_frequency | validate_frequency | core/validation.py | ✓ | Time series frequency |
| validate_alignment | validate_alignment | core/validation.py | ✓ | Series alignment |
| handle_missing_data | handle_missing_data | data/gaps.py | ✓ | Missing data handling |
| align_series | align_series | data/alignment.py | ✓ | Time series alignment |
| standardize_frequency | standardize_frequency | data/alignment.py | ✓ | Frequency conversion |
| load_csv_data | load_csv_data | data/loader.py | ✓ | CSV import |
| export_results | export_results | data/loader.py | ✓ | Data export |
| cache_management | manage_cache | core/cache.py | ✓ | Memory optimization |

#### 3.1 Series Conversion Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| price_to_ret | price_to_ret | statistics/base.py | ✓ | - |
| ret_to_price | ret_to_price | statistics/base.py | ✓ | - |
| excess_ret | excess_returns | statistics/base.py | ✓ | - |
| alpha_ret | - | - | 🔴 | Deprecated in v2.0 |
| ret_to_drawdown | drawdown_series | statistics/timeseries.py | ✓ | Renamed |
| standardize_ret | standardize_returns | statistics/base.py | ✓ | - |

#### 3.2.1 Statistical Analysis Functions - Basic
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| ret_mean | mean_return | statistics/metrics.py | ✓ | Renamed |
| ret_volatility | stdev | statistics/metrics.py | ✓ | Renamed |
| ret_skew | skewness | statistics/metrics.py | ✓ | Renamed |
| ret_kurtosis | kurtosis | statistics/metrics.py | ✓ | Renamed |
| ret_stats | ret_stats | statistics/metrics.py | ✓ | - |

#### 3.2.2 Statistical Analysis Functions - Annualized Returns
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| annualize_ret | annualized_return | statistics/adjusted.py | ✓ | - |
| arithmetic_to_geometric_ret | geometric_return | statistics/adjusted.py | ✓ | Cross-ref: [AG#2.1] |
| geometric_to_arithmetic_ret | - | - | 🔴 | Not implemented |
| calculate_variance_drag | variance_drag | statistics/adjusted.py | ✓ | Cross-ref: [AG#2.1] |
| calculate_kurtosis_drag | kurtosis_drag | statistics/adjusted.py | ✓ | Cross-ref: [AG#2.1] |
| calculate_skew_drag | skew_drag | statistics/adjusted.py | ✓ | Cross-ref: [AG#2.1] |
| calculate_total_drag | - | - | 🔴 | Not implemented |

#### 3.2.3 Statistical Analysis Functions - Volatility Adjustments
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| adjust_volatility_kurtosis | adjusted_volatility_normal | statistics/adjusted.py | ✓ | Cross-ref: [AG#2.2] |
| annualize_volatility | volatility | statistics/base.py | ✓ | - |
| calculate_downside_volatility | downside_volatility | statistics/base.py | ✓ | - |
| volatility_of_volatility | vol_of_vol | statistics/base.py | ✓ | - |

#### 3.2.4 Statistical Analysis Functions - Drawdowns
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| calculate_drawdown_series | drawdown_series | statistics/timeseries.py | ✓ | - |
| maximum_drawdown | max_drawdown | statistics/timeseries.py | ✓ | - |
| average_drawdown | avg_drawdown | statistics/timeseries.py | ✓ | - |
| drawdown_duration | drawdown_duration | statistics/timeseries.py | ✓ | - |
| theoretical_max_drawdown | max_theoretical_drawdown | statistics/adjusted.py | ✓ | Cross-ref: [AG#4.1] |

#### 3.3 Risk and Performance Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| standard_sharpe_ratio | standard_sharpe_ratio | statistics/adjusted.py | ✓ | Cross-ref: [AG#3.1] |
| geometric_sharpe_ratio | geometric_sharpe_ratio | statistics/adjusted.py | ✓ | Cross-ref: [AG#3.1] |
| adjusted_sharpe_ratio | adjusted_geometric_sharpe_ratio | statistics/adjusted.py | ✓ | Cross-ref: [AG#3.1] |
| sortino_ratio | sortino_ratio | statistics/adjusted.py | ✓ | - |
| calmar_ratio | calmar_ratio | statistics/adjusted.py | ✓ | - |
| probabilistic_sharpe_ratio | probabilistic_sharpe_ratio | statistics/adjusted.py | ✓ | Cross-ref: [AG#3.2] |
| information_ratio | information_ratio | statistics/adjusted.py | ✓ | - |
| treynor_ratio | treynor_ratio | statistics/adjusted.py | ✓ | - |

#### 3.4 Time-Varying Window Statistics Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| rolling_volatility | rolling_volatility | statistics/timeseries.py | ✓ | Cross-ref: [VF#2] |
| rolling_volatility_downside | rolling_volatility_downside | statistics/timeseries.py | ✓ | - |
| rolling_correlation | rolling_correlation | statistics/timeseries.py | ✓ | - |
| rolling_beta | rolling_beta | statistics/timeseries.py | ✓ | - |
| rolling_sharpe | rolling_sharpe | statistics/timeseries.py | ✓ | - |
| rolling_ret | rolling_returns | statistics/timeseries.py | ✓ | - |
| hurst_exponent | hurst_exponent | statistics/timeseries.py | ✓ | - |

#### 3.5 Correlation and Dependency Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| correlation | calculate_correlation_matrix | statistics/base.py | ✓ | - |
| rank_correlation | spearman_correlation | statistics/base.py | ✓ | - |
| correlation_to_covariance | correlation_to_covariance | statistics/base.py | ✓ | - |
| covariance_to_correlation | covariance_to_correlation | statistics/base.py | ✓ | - |
| semi_covariance | semi_covariance | statistics/base.py | ✓ | - |

#### 3.6 Matrix Transformation Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| correlation_cluster | correlation_cluster | statistics/base.py | ✓ | - |
| shrink_covariance | ledoit_wolf_shrinkage | covariance/shrinkage.py | ✓ | Cross-ref: [SC#2] |

#### 3.7 Distribution Fitting Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| fit_gaussian | fit_normal_dist | distribution/skew_student_t.py | ✓ | Renamed |
| fit_student_t | fit_t_dist | distribution/skew_student_t.py | ✓ | Renamed |
| fit_skewed_t | fit_skewed_t_dist | distribution/skew_student_t.py | ✓ | - |
| fit_nig | - | - | 🔴 | Deprecated in v2.0 |
| distribution_test | test_normality | distribution/skew_student_t.py | ✓ | - |
| implied_drag_student_t | - | - | ❌ | Not implemented |
| implied_heavy_tail_drag_student_t | - | - | ❌ | Not implemented |
| implied_excess_kurtosis_drag_student_t | - | - | ❌ | Not implemented |
| implied_skew_drag_student_t | - | - | ❌ | Not implemented |
| implied_drag_variance | variance_drag | statistics/adjusted.py | ✓ | Renamed |

#### 3.8 Copula Functions
All functions in this section are deprecated in v2.0 and moved to separate packages.

#### 3.9 Portfolio Optimization Functions
All functions in this section are deprecated in v2.0 and moved to separate packages.

#### 3.10 Utility Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| volatility_target | kelly_fraction_normal | statistics/adjusted.py | ✓ | Cross-ref: [AG#4.1] |
| max_theoretical_drawdown | max_theoretical_drawdown | statistics/adjusted.py | ✓ | Cross-ref: [AG#4.1] |

#### 3.11-3.12 Interest Rate and DCF Functions
All functions in these sections are marked as deprecated in v2.0 and moved to separate packages.

#### 3.13 Synthetic Series Generation Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| regress_ret | analyze_relationship | backfill/generator.py | ✓ | Cross-ref: [BF#2] |
| analyze_residuals | validate_backfill_results | backfill/generator.py | ✓ | Cross-ref: [BF#2] |
| backfill_ret | backfill_series | backfill/generator.py | ✓ | Cross-ref: [BF#2] |

#### 3.14 PCA Factor Analysis Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| pca_decomposition | pca_factor_returns | statistics/timeseries.py | ✓ | Cross-ref: [PCA#3] |
| select_pca_factors | pca_factor_returns | statistics/timeseries.py | ✓ | Cross-ref: [PCA#3] |
| pca_factor_ret | calculate_factor_loadings | pca/loadings.py | ✓ | Cross-ref: [PCA#3] |
| pca_idiosyncratic_ret | generate_factor_scores | pca/scores.py | ✓ | Cross-ref: [PCA#6] |

#### 3.15 Volatility Forecasting Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| forecast_garch | forecast_garch | volatility/garch.py | ✓ | Cross-ref: [VF#1] |
| forecast_garch_path | forecast_garch_path | volatility/garch.py | ✓ | Day-by-day path |
| forecast_garch_window | forecast_garch_window | volatility/garch.py | ✓ | Rolling window from path |
| forecast_har | forecast_har | volatility/har.py | ✓ | Cross-ref: [VF#2] |
| forecast_har_window | forecast_har_window | volatility/har.py | ✓ | Window-aligned forecasts |
| calculate_har_components | calculate_har_components | volatility/har.py | ✓ | Cross-ref: [VF#2] |
| calculate_realized_vol | calculate_realized_volatility | volatility/har.py | ❌ | Window RV calculation |
| fit_garch_model | fit_garch | volatility/garch.py | ✓ | Cross-ref: [VF#1] |
| fit_har_model | fit_har_model | volatility/har.py | ✓ | Cross-ref: [VF#2] |

#### 3.16 Dynamic Time Warping Functions
| Required Function | Current Implementation | Module | Status | Notes |
|------------------|----------------------|---------|---------|-------|
| dtw_distance | calculate_dtw_distance | dtw/similarity.py | ✓ | Cross-ref: [DTW#3] |
| dtw_similarity | calculate_dtw_distance | dtw/similarity.py | ✓ | Cross-ref: [DTW#3] |
| dtw_correlation | dtw_to_correlation | dtw/correlation.py | ✓ | Cross-ref: [DTW#3] |
