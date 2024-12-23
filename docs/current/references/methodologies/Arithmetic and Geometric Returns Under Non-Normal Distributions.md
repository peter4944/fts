## 1. Introduction and Basic Concepts

### 1.1 Overview

Converting arithmetic returns to geometric returns is essential for understanding the actual realized return of an individual asset over time. This conversion is particularly important because:
- Real-world investment returns compound over time
- Return distributions are rarely perfectly normal
- Risk factors can have compounding negative effects
- Accurate estimates of returns are needed for volatility targeting and position sizing decisions

This note provides a comprehensive guide to the conversion of arithmetic to geometric returns and related adjustments, with formulas and Python code for practical applications. This note provides the foundational calculations and adjustments used in portfolio construction, while portfolio level adjustments are handled in a separate note on portfolio construction.

### 1.2 Key Definitions

#### 1.2.1 Basic Return Measures

- **Arithmetic Return ($R_a$)**: Simple average return over a period, calculated as the sum of returns divided by the number of periods. For asset *i*, it is denoted as $R_{a,i}$.
- **Geometric Return ($R_g$)**: Return that accounts for compounding effects, representing the actual realized return over time. For asset *i*, it is denoted as $R_{g,i}$.
- **Volatility ($\sigma$)**: Standard deviation of returns, measuring the dispersion of returns around their mean. For asset *i*, it is denoted as $\sigma_i$.

#### 1.2.2 Higher-Order Moments
- **Skewness ($\gamma_3$)**: Measures asymmetry in the return distribution. For asset *i*, it is denoted as $\gamma_{3,i}$.
  - Positive skewness: More extreme positive returns than negative
  - Negative skewness: More extreme negative returns than positive
  - Zero skewness: Symmetric distribution

- **Kurtosis ($\gamma_4$)**: Measures the thickness of distribution tails. For asset *i*, it is denoted as $\gamma_{4,i}$.
  - For a normal distribution, $\gamma_4 = 3$
  - **Excess Kurtosis** = $\gamma_4 - 3$
  - Higher kurtosis indicates more frequent extreme events

#### 1.2.3 Adjusted Sharpe Ratio
- **Adjusted Arithmetic Sharpe Ratio:** The Sharpe ratio adjusted for higher moments of the return distribution.  This can be calculated under the assumption of either normal or skewed student-t distributions, denoted as $SR_{a,adj}$ or $SR_{t,adj,arith}$, respectively.

### 1.3 Why Adjustments Matter

#### 1.3.1 Volatility Drag
- Returns don't compound linearly
- Example: A +50% return followed by a -50% return results in a -25% total return
- This effect, known as volatility drag, increases with higher volatility
- Formula: $-\frac{\sigma^2}{2}$ represents the expected drag from volatility

#### 1.3.2 Impact of Skewness
- Negative skewness is particularly dangerous for leveraged assets
- Common in financial markets (crashes occur more suddenly than rallies)
- Requires higher risk premiums as compensation
- Affects optimal position sizing and leverage decisions

#### 1.3.3 Kurtosis Considerations
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

#### 2.1.1 Under Normal Distribution

For an individual asset *i*:

1. **Arithmetic Return ($R_{a,i}$)**
   $$ R_{a,i} = \text{Simple average of returns} $$

2. **Variance Drag**
   $$ \text{Variance Drag}_i = \frac{\sigma_i^2}{2} $$

3. **Kurtosis Drag**
   $$ \text{Kurtosis Drag}_i = \frac{(\gamma_{4,i} - 3)\sigma_i^4}{24} $$

4. **Skew Drag**
   $$ \text{Skew Drag}_i = \frac{-\gamma_{3,i}\sigma_i^3}{6} $$

5. **Return Progression**
   - **Geometric Return**:
     $$ R_{g,i} = R_{a,i} - \text{Variance Drag}_i $$
   - **Adjusted Geometric Return**:
     $$ R_{g,adj,i} = R_{g,i} - \text{Kurtosis Drag}_i - \text{Skew Drag}_i $$

#### 2.1.2 Under Skewed Student-t Distribution

For an individual asset *i*, under the assumption of a skewed Student-t distribution with degrees of freedom $\nu$ and skewness $\gamma_{3,i}$:

1.  **Arithmetic Return ($R_{a,i}$)**
    $$ R_{a,i} = \text{Simple average of returns} $$

2.  **Variance Drag**
    $$ \text{Variance Drag}_i = \frac{\sigma_i^2}{2} $$

3.  **Heavy Tail Drag**
    $$ \text{Heavy Tail Drag}_i = \frac{\sigma_i^2}{2} \times \frac{2}{\nu - 2} $$

4.  **Skew Drag**
    $$ \text{Skew Drag}_i = \frac{-\gamma_{3,i}\sigma_i^3}{6} $$

5. **Return Progression**
    - **Geometric Return**
      $$ R_{g,i} = R_{a,i} - \text{Variance Drag}_i $$
    - **Adjusted Geometric Return (Skewed Student-t)**
      $$ R_{g,adj,t,i} = R_{a,i} - \text{Variance Drag}_i - \text{Heavy Tail Drag}_i + \text{Skew Drag}_i  $$
    
   Where $\lambda$ is market impact factor


### 2.2 Volatility Adjustments

#### 2.2.1 Under Normal Distribution
1. **Standard Volatility ($\sigma_i$)**:
   - Annualized log-return volatility

2. **Adjusted Volatility ($\sigma_{adj,i}$)**:
   $$ \sigma_{adj,i} = \sigma_i\sqrt{1 + \frac{(\gamma_{4,i} - 3)\sigma_i^2}{4} + \frac{\gamma_{3,i}^2\sigma_i^2}{6}} $$

#### 2.2.2 Under Skewed Student-t Distribution
1. **Student-t Volatility ($\sigma_{t,i}$)**:
   $$ \sigma_{t,i} = \sigma_i\sqrt{\frac{\nu}{\nu-2}} $$
   Where $\nu$ is degrees of freedom

2. **Skew-Adjusted Student-t Volatility ($\sigma_{t,adj,i}$)**:
   $$ \sigma_{t,adj,i} = \sigma_{t,i}(1 + \gamma_{3,i}\lambda) $$
   Where $\lambda$ is the market impact factor

These volatility adjustments are used to standardize returns for asset comparison purposes during portfolio construction and as an input to volatility targeting.

## 3. Performance Metrics

### 3.1 Sharpe Ratio Under Normal Distribution
For an individual asset *i*:
1. **Standard Sharpe Ratio (Arithmetic)**:
   $$ SR_i = \frac{R_{a,i} - R_f}{\sigma_i} $$

2. **Geometric Sharpe Ratio**:
   $$ SR_{g,i} = \frac{R_{g,i} - R_f}{\sigma_i} $$

3.  **Adjusted Arithmetic Sharpe Ratio:**
    $$ SR_{a,adj,i} = \frac{R_{a,i} - R_f}{\sigma_{adj,i}} $$



## 3.2 Sharpe Ratio Under Student-t Distribution
For an individual asset *i*:

1. **Student-t Sharpe Ratio**:
   $$ SR_{t,i} = \frac{R_{g,adj,i} - R_f}{\sigma_{t,i}} $$

2. **Skew-Adjusted Student-t Sharpe Ratio (Arithmetic)**:
   $$ SR_{t,adj,arith,i} = \frac{R_{a,i} - R_f}{\sigma_{t,adj,i}} $$

3. **Skew-Adjusted Student-t Sharpe Ratio**:
   $$ SR_{t,adj,i} = \frac{R_{g,adj,i} - R_f}{\sigma_{t,adj,i}} $$


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



## 4. Implementation Considerations

### 4.1 Sample Size Requirements
- Minimum samples needed for stable estimates
- Impact of estimation error on each metric
- Rolling window vs. expanding window estimates

### 4.2 Distribution Testing
- Tests for normality
- Determination of degrees of freedom
- Stability of higher moments

### 4.3 Practical Constraints
- Transaction costs
- Market impact
- Margin requirements
- Leverage limits

## 5. Implementation Example

```python
import numpy as np
from scipy.stats import norm

def variance_drag(volatility: float) -> float:
    """Calculate variance drag from volatility."""
    return (volatility ** 2) / 2

def kurtosis_drag(excess_kurtosis: float, volatility: float) -> float:
    """Calculate kurtosis drag from excess kurtosis and volatility."""
    return (excess_kurtosis * volatility ** 4) / 24

def heavy_tail_drag(volatility: float, df:float) -> float:
    """Calculate heavy tail drag from volatility and degrees of freedom
    under a student-t or skewed student-t distribution assumption."""
    return (volatility ** 2 / 2) * (2 / (df - 2))

def skew_drag(skewness: float, volatility: float) -> float:
    """Calculate skewness drag from skewness and volatility."""
    return (-skewness * volatility ** 3) / 6

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

def adjusted_geometric_return_student_t(arithmetic_return: float, volatility: float,
                            df: float, skewness: float) -> float:
    """Calculate adjusted geometric return accounting for all drag effects under Student-t distribution assumption."""
    geo_return = geometric_return(arithmetic_return, volatility)
    heavy_tail = heavy_tail_drag(volatility, df)
    skew_drag_val = skew_drag(skewness, volatility)
    return geo_return - heavy_tail - skew_drag_val


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



def adjusted_arithmetic_sharpe_ratio_normal(arithmetic_return: float, volatility: float,
                                  excess_kurtosis: float, skewness: float,
                                  risk_free_rate: float = 0) -> float:
    """Calculate adjusted arithmetic Sharpe ratio assuming normal distribution."""
    adj_vol = adjusted_volatility_normal(volatility, excess_kurtosis, skewness)
    return sharpe_ratio(arithmetic_return, adj_vol, risk_free_rate)




def student_t_sharpe_ratio(arithmetic_return: float, volatility: float,
                          excess_kurtosis: float, skewness: float,
                          df: float, risk_free_rate: float = 0) -> float:
    """Calculate Student-t Sharpe ratio."""
    adj_geo_return = adjusted_geometric_return(arithmetic_return, volatility,
                                             excess_kurtosis, skewness)
    student_t_vol = student_t_volatility(volatility, df)
    return sharpe_ratio(adj_geo_return, student_t_vol, risk_free_rate)

def adjusted_arithmetic_sharpe_ratio_student_t(arithmetic_return: float, volatility: float,
                                  excess_kurtosis: float, skewness: float,
                                  df:float, market_impact: float,
                                  risk_free_rate: float = 0) -> float:
    """Calculate adjusted arithmetic Sharpe ratio assuming Student-t distribution."""
    adj_vol = skew_adjusted_student_t_volatility(volatility, df, skewness, market_impact)
    return sharpe_ratio(arithmetic_return, adj_vol, risk_free_rate)

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
# replace this for portfolio_volatility_target(sharpe ratio, kelly_fraction)
#show Kelly fraction impact on geometric growth and MTD
#where fractional Kelly gives vol, convert arithmetic return implied in vol_target
#to geometric return (and adjusted, calculate MTD in fractional Kelly


def mtd_normal(sharpe_ratio: float) -> float:
    """Calculate Maximum Theoretical Drawdown under normal distribution."""
    return 0.5  # At full Kelly

 

# Example usage:
R_a = 0.075    # 7.5% arithmetic return
vol = 0.20     # 20% volatility
skew = -0.45
kurt_excess = 1.5
v = 9
lam = 0.2
rf = 0.0

R_g_adj_normal = geometric_return_adjusted(R_a, vol, skew, kurt_excess)
sigma_adj_normal = adjusted_volatility_normal(vol, skew, kurt_excess)
SR_g_adj_normal = sharpe_ratio(R_g_adj_normal, sigma_adj_normal, rf)
MTD_normal = max_theoretical_drawdown_normal_full_kelly()

R_g_adj_t = adjusted_geometric_return_student_t(R_a, vol, v, skew)
sigma_t_adj = skewed_t_vol(vol, v)
SR_t_adj = sharpe_ratio(R_g_adj_t, sigma_t_adj, rf)
vol_target_t = optimal_vol_target_skewed_t(SR_t_adj, v, skew, lam)
MTD_t = max_theoretical_drawdown_skewed_t(vol_target_t, SR_t_adj, v, skew, lam)
GDR_t = GDR(R_g_adj_t, MTD_t)

print("Normal Dist Adjusted Geometric Return:", R_g_adj_normal)
print("Normal Dist Adjusted Sharpe:", SR_g_adj_normal)
print("Normal Dist MTD (Full Kelly):", MTD_normal)

print("Skewed-t Adjusted Geometric Return:", R_g_adj_t)
print("Skewed-t Adjusted Sharpe:", SR_t_adj)
print("Skewed-t Optimal Vol Target:", vol_target_t)
print("Skewed-t MTD:", MTD_t)
print("Skewed-t GDR:", GDR_t)
```

## 13. Conclusions

- Converting arithmetic to geometric returns requires adjusting for volatility, skewness, and kurtosis.
- Higher moments significantly reduce achievable Sharpe ratios, increase maximum drawdowns, and lower the optimal Kelly volatility target.

By considering these realistic adjustments, investors can better target sustainable growth, manage extreme risk, and optimize their portfolios under more accurate return distribution assumptions.
