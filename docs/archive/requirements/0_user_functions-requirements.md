# Financial Time Series Functions

## 1. Required additional Research References to consider
- Stambaugh, Correlation and volatility where price histories differ in length
- Jorion Bayes-Stein shrinkage model (returns and covariance)
- Principal Component Regression (PCR)
- Adjusted Sharpe Ratio
- M-Squared (a measure of risk-adjusted return)
- Copulas (student-t -symmetric tails, gaussian, clayton asymmetric positive lower tail dependence)

## 1. Overview & Objectives

### 1.1 Library Purpose
A Python library for financial time series analysis focusing on:
- Statistical analysis of return series
- Risk and performance metrics
- Portfolio optimization
- Multi-asset correlation and dependency analysis

### 1.2 Core Capabilities
- Handle both price and return series data
- Support multiple return calculation methods (log, simple)
- Provide comprehensive statistical analysis
- Enable portfolio optimization with various approaches
- Support different time frequencies (daily, weekly, monthly)



### 1.2 Core Capabilities
- Handle both price and return series data
- back-fill missing data data in price/return series
- Support multiple return calculation methods (log, simple)
- Provide comprehensive statistical analysis
- Enable portfolio optimization with various closed form approaches
- Support different time frequencies (daily, weekly, monthly)

### Simulations - integrate with a library or subpackage to perform simulations from distribution output - see reference document 
- simulate returns based on specified Distribution and Moments thereof
- simulate returns based on specified Copula and Marginals, factor loadings, and correlation matrix
- note: the simulations are outside the scope of this library, but the library should support estimation of distribution moments, pca components etc to enable the simulation library to run simulations


### 1.3 Scope

The financial time series library performs all necessary calculations of price series to return series and provides a comprehensive set of statistical, risk, and performance metrics. It also includes functions for dependency analysis and portfolio optimization. The library aims to support large datasets and provide efficient computation for various financial analysis tasks. The outputs and estimatation of the library in the form of matricies, PCA factors, fundamental factors, Distribution fits are used as inputs to the simulation library.

We limit the scope of the library to the functions that can directly be obtained from the estimation of returns series, correlation matrices, series statistics, therefore  mean variance optimised portfolio weight are within the scope, whereas more advanced simulations based on distribution fits etc. are outside the scope and will be handled by the simulation library. Estimation of risk factors from PCA, and fundamental factors from factor 


## 2. Data Structures

### 2.1 Price and Return Series
- Decide on the data structure for price and return series:
  - a. List of price and date tuples.
  - b. Pandas DataFrame with date as index and price as a column.
  - c. Numpy array with date as the first column and price as the second column.
  - d. Dataclass with date and price attributes.

### 2.2 Frequency and Return Type
- Frequency: daily, weekly, monthly, quarterly, or yearly.
- Return type: simple or log (default: log).
- Return_periodicity: period, annualized
- Sharpe, sortino,information etc performance ratios are always annualised
- volatilities are always annualised

## 3. Return Series Conversions
- `price_to_return_series(price_series, return_type) -> return_series`
- `return_to_price_series(return_series, initial_price, return_type) -> price_series`

- `excess_returns(return_series, risk_free_rate) -> excess_return_series`
- `alpha_returns(return_series, benchmark_return_series) -> alpha_return_series`
- 
- generate synthetic benchmark return series based factor_loadings
- generate synthetic return series based on factor loadings 'can be betas to style and market factors' or 'principal components'

- backfill - implement Stambaugh and other methodologies to 'backfill' missing data in return series - this needs to be further scoped out in terms of what functions are needed to perform this operation

## 4. Scalar Distribution Statistics
- `return_series_volatility(return_series) -> volatility`
- `return_series_vol_vol(return_series, window) -> volatility_of_volatility`
- `return_series_skew(return_series) -> skew`
- `return_series_kurtosis(return_series) -> kurtosis`
- `return_series_mean(return_series) -> mean`
- `log_return_series_median(return_series) -> median`
- `log_return_to_cumulative(return_series, cumulative_return_type: cumulative/annualised) -> cumulative_return_series`
- `histogram(return_series, buckets) -> histogram (list of tuples)`

## 5. Risk and Performance Measures
- `sharpe_ratio(return_series, risk_free_rate) -> sharpe_ratio`
- `sharp_ratio_adjusted(return_series, risk_free_rate, skew, kurtosis) -> sharpe_ratio_adj`
- `sortino_ratio(return_series) -> sortino_ratio`
- `maximum_drawdown(return_series) -> maximum_drawdown`
- `maximum_drawdown_duration(return_series) -> maximum_drawdown_duration`
- `maximum_drawdown_start(return_series) -> maximum_drawdown_start`
- `maximum_drawdown_end(return_series) -> maximum_drawdown_end`
- `information_ratio(return_series, benchmark_return_series) -> information_ratio`
- `max_drawdown_theoretical(sharpe_ratio, portfolio_volatility) -> max_drawdown`
- `return_series_stats(return_series) -> stats (list of tuples)`

## 6. Time-Varying Statistics
- `rolling_volatility(return_series, window) -> rolling_volatility (dataframe)`

## 7. Return Series Correlation and Dependencies
- `correlation(return_series) -> correlation_matrix`
- `rank_correlation(return_series, Method) -> rank_correlation_matrix`
- `correlation_to_covariance_matrix(correlation_matrix, volatilities) -> covariance_matrix`
- `covariance_to_correlation_matrix(covariance_matrix) -> correlation_matrix`
- `semi_covariance(return_series, threshold) -> semi_covariance_matrix`

## 8. Correlation Transformations and Adjustments
- `correlation_cluster(correlation_matrix, threshold) -> clusters`
- `shrink_covariance_matrix(covariance_matrix, shrinkage_target) -> shrunk_covariance_matrix`
- `pca_decomposition(covariance_matrix) -> pca_components, pca_explained_variance`

## 9. Distribution Fitting
- Gaussian
- Student-t
- Skew-student-t
- NIG (Normal Inverse Gaussian)



## 10. Utility Functions
- `volatility_target(sharpe_ratio) -> volatility_target`
- `sharpe_ratio_adjusted(sharpe_ratio, volatility, skew, kurtosis) -> sharpe_ratio_adj`
- `e_drawdown(sharpe_ratio, target_volatility) -> e_drawdown`

### 10.1 Interest Rate Utility Functions
- `compounding_conversion()`
- `forward_rate() -> forward_rate`
- `zero_coupon_rate(from BEY)`
- `bey_coupon_rate(from zero coupon rate)`
- `bond_price()`
- `time_value_of_money(principal, rate, time) -> future_value`
- `discount_rate(principal, future_value, time) -> rate`
- `present_value(future_value, rate, time) -> principal`

### 10.2 Discount Cash-Flow Models
- `npv(cash_flows, discount_rate, type: FCF, FCFE) -> npv`
- `WACC(equity, debt, equity_cost, debt_cost, tax_rate) -> WACC`
- `XIRR(cash_flows, guess) -> XIRR`

## 11. Copulas
- Student-t (symmetric tails)
- Gaussian
- Clayton (asymmetric positive lower tail dependence)
- `copula_fit(return_series, copula_type) -> copula`

## 12. Dynamic Time Warping
- Convert DTW (-1,1) to correlation equivalent

## 13. Portfolio Functions

### 13.1 Mean-Variance Optimization (MVO)
- `mean_variance_optimization(return_series, constraints) -> optimal_weights`
- `marginal_risk_contribution() -> marginal_risk_contribution`
- `component_risk_contribution() -> component_risk_contribution`
- `portfolio_volatility() -> portfolio_volatility`
- `e_portfolio_return() -> e_portfolio_return`
- `black_litterman() -> optimal_weights`

### 13.2 Risk Parity
- `risk_parity_optimization() -> optimal_weights`
- `pca_risk_parity_optimization() -> optimal_weights`

### 13.3 Hierarchical Risk Parity
- `hierarchical_risk_parity() -> optimal_weights`# Financial Time Series Functions



what would be a good way of defining a python dataclass for financial  timeseries data. The timeseries can be either of price or returns, and has a datetime associated with each element. the timeseries object has properties such as volatility, mean and other statistical properties. if the object can hold multiple asset timeseries, an operation can also be to compute the correlation matrix, principle components and other statistical methods of dependency between series


Optoions:
- Define a dataclass for the timeseries object relating to price and returns and statistical properties
- Define a separate datacalass for a correlation matrix object, with matrix operations and transformations, e.g PCA

Or just have a FinancialTimeSeries class like below, with a method to compute the correlation matrix and principal components, IF the object contains multiple series:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

@dataclass
class FinancialTimeSeries:
    # Core data storage
    data: pd.DataFrame
    series_type: str = field(default="price")  # "price" or "returns"
    
    def __post_init__(self):
        # Validate and process input data
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Convert prices to returns if needed
        if self.series_type == "returns" and not self._is_returns():
            self.data = np.log(self.data / self.data.shift(1)).dropna()
    
    def _is_returns(self) -> bool:
        """Check if data is already in returns format"""
        return np.all(np.abs(self.data) < 1.0)  # Simple heuristic
    
    @property
    def volatility(self) -> Union[float, pd.Series]:
        """Calculate annualized volatility for each series"""
        return self.data.std() * np.sqrt(252)  # Assuming daily data
    
    @property
    def mean(self) -> Union[float, pd.Series]:
        """Calculate mean for each series"""
        return self.data.mean()
    
    @property
    def skewness(self) -> Union[float, pd.Series]:
        """Calculate skewness for each series"""
        return self.data.skew()
    
    @property
    def kurtosis(self) -> Union[float, pd.Series]:
        """Calculate excess kurtosis for each series"""
        return self.data.kurtosis()
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix between series"""
        return self.data.corr()
    
    def rolling_statistics(self, window: int = 30) -> Dict[str, pd.DataFrame]:
        """Calculate rolling statistics"""
        return {
            'mean': self.data.rolling(window=window).mean(),
            'volatility': self.data.rolling(window=window).std() * np.sqrt(252),
            'skewness': self.data.rolling(window=window).skew(),
        }
    
    def principal_components(self, n_components: Optional[int] = None) -> tuple:
        """Compute principal components analysis"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.data)
        
        return (
            pd.DataFrame(components, index=self.data.index),
            pd.DataFrame(pca.components_),
            pca.explained_variance_ratio_
        )
    
    def autocorrelation(self, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Compute autocorrelation for specified lags"""
        return pd.DataFrame({
            f'lag_{lag}': [self.data[col].autocorr(lag=lag) 
                          for col in self.data.columns]
            for lag in lags
        }, index=self.data.columns)
    
    def decompose(self, period: int = 252):
        """Perform time series decomposition"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        results = {}
        for column in self.data.columns:
            decomposition = seasonal_decompose(
                self.data[column], 
                period=period, 
                model='additive'
            )
            results[column] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        return results
```