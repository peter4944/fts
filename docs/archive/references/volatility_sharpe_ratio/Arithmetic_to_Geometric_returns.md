# Arithmetic to Geometric Returns: Theory and Applications

## 1. Introduction and Basic Concepts

### 1.1 Overview
Converting arithmetic returns to geometric returns is essential for understanding actual investment performance over time. This conversion is particularly important because:
- Real-world investment returns compound over time
- Return distributions are rarely perfectly normal
- Risk factors can have compounding negative effects
- Portfolio rebalancing and position sizing decisions depend on accurate return estimates

### 1.2 Key Definitions

#### Basic Return Measures
- **Arithmetic Return ($R_a$)**: Simple average return over a period, calculated as the sum of returns divided by the number of periods
- **Geometric Return ($R_g$)**: Return that accounts for compounding effects, representing the actual realized return over time
- **Volatility ($\sigma$)**: Standard deviation of returns, measuring the dispersion of returns around their mean

#### Higher-Order Moments
- **Skewness ($\gamma_3$)**: Measures asymmetry in the return distribution
  - Positive skewness: More extreme positive returns than negative
  - Negative skewness: More extreme negative returns than positive
  - Zero skewness: Symmetric distribution

- **Kurtosis ($\gamma_4$)**: Measures the thickness of distribution tails
  - For a normal distribution, $\gamma_4 = 3$
  - **Excess Kurtosis** = $\gamma_4 - 3$
  - Higher kurtosis indicates more frequent extreme events
  - Particularly important for risk management and option pricing

### 1.3 Why Adjustments Matter

#### Volatility Drag
- Returns don't compound linearly
- Example: A +50% return followed by a -50% return results in a -25% total return
- This effect, known as volatility drag, increases with higher volatility
- Formula: $-\frac{\sigma^2}{2}$ represents the expected drag from volatility

#### Impact of Skewness
- Negative skewness is particularly dangerous for leveraged portfolios
- Common in financial markets (crashes occur more suddenly than rallies)
- Requires higher risk premiums as compensation
- Affects optimal position sizing and leverage decisions

#### Kurtosis Considerations
- Fat tails mean extreme events happen more frequently than predicted by normal distribution
- Critical for risk management and stress testing
- Affects value-at-risk (VaR) and expected shortfall calculations
- More prominent in:
  - High-frequency trading
  - Options and derivatives
  - Crisis periods
  - Illiquid assets

## 2. Return Measures and Adjustments

### 2.1 Return Components and Drag Effects

1. **Arithmetic Return ($R_a$)**
   $$ R_a = \text{Simple average of returns} $$

2. **Variance Drag**
   $$ \text{Variance Drag} = \frac{\sigma^2}{2} $$

3. **Kurtosis Drag**
   $$ \text{Kurtosis Drag} = \frac{(\gamma_4 - 3)\sigma^4}{24} $$

4. **Skewness Drag**
   $$ \text{Skew Drag} = \frac{\gamma_3\sigma^3}{6} $$

5. **Return Progression**
   - **Geometric Return**:
     $$ R_g = R_a - \text{Variance Drag} $$
   - **Adjusted Geometric Return**:
     $$ R_{g,adj} = R_g - \text{Kurtosis Drag} - \text{Skew Drag} $$

### 2.2 Volatility Adjustments

#### Under Normal Distribution
1. **Standard Volatility ($\sigma$)**:
   - Annualized log-return volatility

2. **Adjusted Volatility ($\sigma_{adj}$)**:
   $$ \sigma_{adj} = \sigma\sqrt{1 + \frac{(\gamma_4 - 3)\sigma^2}{4} + \frac{\gamma_3^2\sigma^2}{6}} $$

#### Under Skewed Student-t Distribution
1. **Student-t Volatility ($\sigma_t$)**:
   $$ \sigma_t = \sigma\sqrt{\frac{\nu}{\nu-2}} $$
   Where $\nu$ is degrees of freedom

2. **Skew-Adjusted Student-t Volatility ($\sigma_{t,adj}$)**:
   $$ \sigma_{t,adj} = \sigma_t(1 + \gamma_3\lambda) $$
   Where $\lambda$ is the market impact factor

## 3. Performance Metrics

### 3.1 Sharpe Ratio Under Normal Distribution
1. **Standard Sharpe Ratio**:
   $$ SR = \frac{R_a - R_f}{\sigma} $$

2. **Geometric Sharpe Ratio**:
   $$ SR_g = \frac{R_g - R_f}{\sigma} $$

3. **Adjusted Geometric Sharpe Ratio**:
   $$ SR_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}} $$

### 3.2 Sharpe Ratio Under Student-t Distribution
1. **Student-t Sharpe Ratio**:
   $$ SR_t = \frac{R_{g,adj} - R_f}{\sigma_t} $$

2. **Skew-Adjusted Student-t Sharpe Ratio**:
   $$ SR_{t,adj} = \frac{R_{g,adj} - R_f}{\sigma_{t,adj}} $$

### 3.3 Probabilistic Sharpe Ratio
$$ PSR(SR^*) = \Phi\left(\sqrt{n-1} \times \frac{SR - SR^*}{\sqrt{1 - \gamma_3 SR + \frac{(\gamma_4-1)}{4}SR^2}}\right) $$

Where:
- $n$ is sample size
- $SR^*$ is the threshold Sharpe ratio
- $\Phi$ is the cumulative normal distribution

### 3.4 Deflated Sharpe Ratio
The Deflated Sharpe Ratio (DSR) accounts for multiple testing bias when selecting strategies:

$$ DSR = PSR(SR_{\text{expected}}) $$

The expected maximum Sharpe ratio $(SR_{\text{expected}})$ after $T$ independent trials is:

$$ SR_{\text{expected}} = \sqrt{2\log(T)} - \frac{\log(\log(T)) + \log(4\pi)}{\sqrt{2\log(T)}} $$

Where $T$ represents:
- Number of strategy variations tested
- Number of parameter combinations tried
- Number of assets/portfolios evaluated
- Product of backtest length and rebalancing frequency

For example:
- Testing 100 parameter combinations: $T = 100$
- Testing 10 strategies on 10 assets: $T = 100$
- Testing monthly rebalancing over 5 years: $T = 60$

## 4. Kelly Criterion and Position Sizing

### 4.1 Under Normal Distribution
1. **Basic Kelly Fraction**:
   $$ f^*_{\text{normal}} = SR = \frac{R_a - R_f}{\sigma} $$

2. **Geometric Kelly Fraction**:
   $$ f^*_g = \frac{R_g - R_f}{\sigma} $$

3. **Adjusted Geometric Kelly Fraction**:
   $$ f^*_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}} $$

### 4.2 Under Student-t Distribution
1. **Student-t Kelly Fraction**:
   $$ f^*_t = f^*_{g,adj} \times \frac{\nu-2}{\nu+1} $$

2. **Skew-Adjusted Student-t Kelly Fraction**:
   $$ f^*_{t,adj} = f^*_t \times (1 - \gamma_3\lambda) $$

## 5. Maximum Theoretical Drawdown (MTD)
Assuming portfolio volatility is scaled to Kelly weight

### 5.1 Under Normal Distribution
1. **Basic MTD** (at full Kelly):
   $$ MTD_{\text{normal}} = \frac{\sigma}{2 \times SR} = 0.5 $$

2. **Adjusted MTD**:
   $$ MTD_{adj} = \frac{\sigma_{adj}}{2 \times SR_{g,adj}} $$

### 5.2 Under Student-t Distribution
1. **Student-t MTD**:
   $$ MTD_t = MTD_{\text{normal}} \times \sqrt{\frac{\nu}{\nu-2}} $$

2. **Skew-Adjusted Student-t MTD**:
   $$ MTD_{t,adj} = MTD_t \times (1 + \gamma_3\lambda) $$

### 5.3 Partial Kelly Adjustments
For a fraction $k$ of full Kelly:
$$ MTD_{\text{partial}} = k \times MTD_{\text{full}} $$

## 6. Implementation Considerations

### 6.1 Sample Size Requirements
- Minimum samples needed for stable estimates
- Impact of estimation error on each metric
- Rolling window vs. expanding window estimates

### 6.2 Distribution Testing
- Tests for normality
- Determination of degrees of freedom
- Stability of higher moments

### 6.3 Practical Constraints
- Transaction costs
- Market impact
- Margin requirements
- Leverage limits

## 7. Implementation Example

```python
import numpy as np
from scipy.stats import norm

def variance_drag(volatility: float) -> float:
    """Calculate variance drag from volatility."""
    return (volatility ** 2) / 2

def kurtosis_drag(excess_kurtosis: float, volatility: float) -> float:
    """Calculate kurtosis drag from excess kurtosis and volatility."""
    return (excess_kurtosis * volatility ** 4) / 24

def skew_drag(skewness: float, volatility: float) -> float:
    """Calculate skewness drag from skewness and volatility."""
    return (skewness * volatility ** 3) / 6

def geometric_return(arithmetic_return: float, volatility: float) -> float:
    """Calculate geometric return from arithmetic return and volatility."""
    return arithmetic_return - variance_drag(volatility)

def adjusted_geometric_return(arithmetic_return: float, volatility: float, 
                            excess_kurtosis: float, skewness: float) -> float:
    """Calculate adjusted geometric return accounting for all drag effects."""
    geo_return = geometric_return(arithmetic_return, volatility)
    kurt_drag = kurtosis_drag(excess_kurtosis, volatility)
    skew_drag_val = skew_drag(skewness, volatility)
    return geo_return - kurt_drag - skew_drag_val

def adjusted_volatility_normal(volatility: float, excess_kurtosis: float, 
                             skewness: float) -> float:
    """Calculate adjusted volatility under normal distribution assumption."""
    kurtosis_term = (excess_kurtosis * volatility ** 2) / 4
    skewness_term = (skewness ** 2 * volatility ** 2) / 6
    return volatility * np.sqrt(1 + kurtosis_term + skewness_term)

def student_t_volatility(volatility: float, df: float) -> float:
    """Calculate volatility under Student-t distribution."""
    return volatility * np.sqrt(df / (df - 2))

def skew_adjusted_student_t_volatility(volatility: float, df: float, 
                                     skewness: float, market_impact: float) -> float:
    """Calculate skew-adjusted Student-t volatility."""
    student_t_vol = student_t_volatility(volatility, df)
    return student_t_vol * (1 + skewness * market_impact)

def sharpe_ratio(return_value: float, volatility: float, risk_free_rate: float = 0) -> float:
    """Calculate basic Sharpe ratio."""
    return (return_value - risk_free_rate) / volatility

def geometric_sharpe_ratio(arithmetic_return: float, volatility: float, 
                          risk_free_rate: float = 0) -> float:
    """Calculate geometric Sharpe ratio."""
    geo_return = geometric_return(arithmetic_return, volatility)
    return sharpe_ratio(geo_return, volatility, risk_free_rate)

def adjusted_geometric_sharpe_ratio(arithmetic_return: float, volatility: float,
                                  excess_kurtosis: float, skewness: float,
                                  risk_free_rate: float = 0) -> float:
    """Calculate adjusted geometric Sharpe ratio."""
    adj_geo_return = adjusted_geometric_return(arithmetic_return, volatility,
                                             excess_kurtosis, skewness)
    adj_vol = adjusted_volatility_normal(volatility, excess_kurtosis, skewness)
    return sharpe_ratio(adj_geo_return, adj_vol, risk_free_rate)

def student_t_sharpe_ratio(arithmetic_return: float, volatility: float,
                          excess_kurtosis: float, skewness: float,
                          df: float, risk_free_rate: float = 0) -> float:
    """Calculate Student-t Sharpe ratio."""
    adj_geo_return = adjusted_geometric_return(arithmetic_return, volatility,
                                             excess_kurtosis, skewness)
    student_t_vol = student_t_volatility(volatility, df)
    return sharpe_ratio(adj_geo_return, student_t_vol, risk_free_rate)

def expected_max_sharpe_ratio(trials: int) -> float:
    """Calculate expected maximum Sharpe ratio after multiple trials."""
    log_t = np.log(trials)
    term1 = np.sqrt(2 * log_t)
    term2 = (np.log(log_t) + np.log(4 * np.pi)) / term1
    return term1 - term2

def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float,
                             n_samples: int, skewness: float,
                             kurtosis: float) -> float:
    """Calculate Probabilistic Sharpe Ratio."""
    sr_std = np.sqrt((1 - skewness * observed_sr + 
                      ((kurtosis - 1) / 4) * observed_sr ** 2) / (n_samples - 1))
    return norm.cdf((observed_sr - benchmark_sr) / sr_std)

def deflated_sharpe_ratio(observed_sr: float, n_samples: int,
                         trials: int, skewness: float,
                         kurtosis: float) -> float:
    """Calculate Deflated Sharpe Ratio."""
    expected_sr = expected_max_sharpe_ratio(trials)
    return probabilistic_sharpe_ratio(observed_sr, expected_sr, n_samples,
                                    skewness, kurtosis)

def kelly_fraction_normal(return_value: float, volatility: float,
                         risk_free_rate: float = 0) -> float:
    """Calculate Kelly fraction under normal distribution."""
    return sharpe_ratio(return_value, volatility, risk_free_rate)

def kelly_fraction_student_t(arithmetic_return: float, volatility: float,
                           excess_kurtosis: float, skewness: float,
                           df: float, market_impact: float,
                           risk_free_rate: float = 0) -> float:
    """Calculate Kelly fraction under Student-t distribution."""
    adj_geo_sr = adjusted_geometric_sharpe_ratio(arithmetic_return, volatility,
                                               excess_kurtosis, skewness,
                                               risk_free_rate)
    student_t_adj = (df - 2) / (df + 1)
    skew_adj = 1 - skewness * market_impact
    return adj_geo_sr * student_t_adj * skew_adj

def mtd_normal(sharpe_ratio: float) -> float:
    """Calculate Maximum Theoretical Drawdown under normal distribution."""
    return 0.5  # At full Kelly

def mtd_student_t(sharpe_ratio: float, df: float,
                  skewness: float, market_impact: float) -> float:
    """Calculate Maximum Theoretical Drawdown under Student-t distribution."""
    mtd_norm = mtd_normal(sharpe_ratio)
    df_adj = np.sqrt(df / (df - 2))
    skew_adj = 1 + skewness * market_impact
    return mtd_norm * df_adj * skew_adj

def mtd_partial_kelly(mtd_full: float, kelly_fraction: float) -> float:
    """Calculate Maximum Theoretical Drawdown for partial Kelly fraction."""
    return kelly_fraction * mtd_full
```
