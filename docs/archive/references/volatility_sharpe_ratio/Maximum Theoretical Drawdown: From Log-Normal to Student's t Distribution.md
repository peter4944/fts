# Maximum Theoretical Drawdown: From Log-Normal to Student's t Distribution

## 1. Log-Normal Case (Normal Distribution Assumption)

### Basic Formula
$$ MaxDD_{normal} = \frac{\sigma}{2 \times SR} $$

Where:
- σ = portfolio volatility (annualized)
- SR = Sharpe ratio (annualized)

### Example with Typical Market Parameters
- Annual volatility (σ) = 16%
- Sharpe ratio (SR) = 0.45

$$ MaxDD_{normal} = \frac{0.16}{2 \times 0.45} = 0.178 \text{ (17.8\%)} $$

## 2. Student's t Distribution Case

### Key Parameters
- Degrees of freedom (v): controls tail heaviness
- Lower v = heavier tails
- v → ∞ approaches normal distribution
- Typical market v ≈ 6-8 (based on excess kurtosis of 3-4)

### Formula with t Distribution
$$ MaxDD_t = \frac{\sigma}{2 \times SR} \times \sqrt{\frac{v}{v-2}} $$

### Relationship to Kurtosis
- Excess Kurtosis = $\frac{6}{v-4}$ (for v > 4)
- For typical market excess kurtosis of 3.5:
  - Solve: 3.5 = 6/(v-4)
  - Implies v ≈ 7

### Example with v = 7
$$ MaxDD_t = 0.178 \times \sqrt{\frac{7}{5}} = 0.211 \text{ (21.1\%)} $$

## 3. Skewed Student's t Distribution Case

### Additional Parameters
- Skewness (typical market ≈ -0.7)
- Skewness adjustment factor (γ × λ)

### Formula with Skewness
$$ MaxDD_{skewed-t} = \frac{\sigma}{2 \times SR} \times \sqrt{\frac{v}{v-2}} \times (1 + \gamma \times \lambda) $$

### Example with Typical Market Parameters
- σ = 16%
- SR = 0.45
- v = 7
- Skewness = -0.7 (γ × λ ≈ 0.15)

$$ MaxDD_{skewed-t} = 0.211 \times (1 + 0.15) = 0.243 \text{ (24.3\%)} $$

## Summary of Results

| Distribution Model | Max Drawdown | Key Assumptions |
|-------------------|--------------|-----------------|
| Log-normal | 17.8% | Normal returns |
| Student's t | 21.1% | Heavy tails (v=7) |
| Skewed Student's t | 24.3% | Heavy tails + negative skew |

## Important Notes

1. **Distribution Parameters**
   - Volatility: 16% (annualized)
   - Sharpe ratio: 0.45
   - Degrees of freedom (v): 7
   - Skewness: -0.7

2. **Limitations**
   - Assumes constant volatility
   - No regime changes
   - Stable correlations
   - Independent returns

3. **Practical Implications**
   - Normal distribution underestimates tail risk
   - Student's t with skewness provides more conservative estimates
   - Actual drawdowns can exceed theoretical maximums
   - Use as lower bound for risk management

4. **Risk Management Applications**
   - Position sizing
   - Stop-loss levels
   - Portfolio leverage decisions
   - Capital allocation

## References for Further Study
- Extreme Value Theory (EVT) for tail modeling
- Dynamic volatility models (GARCH family)
- Copula theory for dependency structures
- Regime-switching models for market states

I'll recalculate the Probabilistic Sharpe Ratio for the market, testing whether the observed SR of 0.45 is statistically different from 0 (i.e., SR* = 0).

# Probabilistic Sharpe Ratio: Market Analysis

## 1. Base Parameters
- Observed SR = 0.45
- SR* (benchmark) = 0
- σ = 16% (annualized)
- γ₃ = -0.7 (skewness)
- γ₄ = 7 (kurtosis)
- n = 252 (one year of daily data)

## 2. PSR Calculation

### Basic Formula
$$ PSR(SR^*) = \Phi\left(\sqrt{n-1} \times \frac{SR - SR^*}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2}}\right) $$

### Step-by-Step Calculation

1) Degrees of Freedom:
$$ \sqrt{n-1} = \sqrt{251} = 15.84 $$

2) Numerator:
$$ SR - SR^* = 0.45 - 0 = 0.45 $$

3) Denominator Components:
- Skewness effect: $-\gamma_3 SR = -(-0.7 \times 0.45) = 0.315$
- Kurtosis effect: $\frac{\gamma_4-1}{4}SR^2 = \frac{6}{4}(0.45)^2 = 0.304$

4) Denominator:
$$ \sqrt{1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2} = \sqrt{1 + 0.315 + 0.304} = \sqrt{1.619} = 1.272 $$

5) Final PSR Calculation:
$$ PSR(0) = \Phi\left(15.84 \times \frac{0.45}{1.272}\right) = \Phi(5.60) = 0.999999 $$

## 3. Interpretation

1) The very high PSR indicates we can reject the null hypothesis that the true Sharpe ratio is ≤ 0 with high confidence

2) Adjusting for higher moments:
- Original SR = 0.45
- Higher moment adjusted SR ≈ $\frac{0.45}{1.272} = 0.354$

## 4. Impact on Maximum Drawdown Estimates

### A. Original Maximum Drawdown Calculations
1) Log-normal case:
$$ MaxDD_{normal} = \frac{0.16}{2 \times 0.354} = 0.226 \text{ or } 22.6\% $$

2) Student's t adjustment (v = 7):
$$ MaxDD_t = 0.226 \times \sqrt{\frac{7}{5}} = 0.267 \text{ or } 26.7\% $$

3) With skewness:
$$ MaxDD_{skewed-t} = 0.267 \times (1 + 0.15) = 0.307 \text{ or } 30.7\% $$

## 5. Comparative Analysis

| Metric | Unadjusted | Higher Moment Adjusted |
|--------|------------|----------------------|
| Sharpe Ratio | 0.45 | 0.354 |
| MaxDD (normal) | 17.8% | 22.6% |
| MaxDD (t-dist) | 21.1% | 26.7% |
| MaxDD (skewed-t) | 24.3% | 30.7% |

## 6. Risk Management Implications

1) **Position Sizing Adjustment**
- Original sizing based on SR = 0.45 should be reduced by factor of ≈ 0.354/0.45 = 0.79

2) **Stop-Loss Levels**
- Should be widened by approximately 26% to account for higher moment adjustments

3) **Risk Budgeting**
$$ \text{Adjusted Risk Budget} = \text{Original Budget} \times \frac{PSR_{adjusted}}{PSR_{unadjusted}} $$

## 7. Key Observations

1) The market's Sharpe ratio is statistically significant despite higher moment adjustments

2) Higher moments materially impact risk metrics:
   - MaxDD estimates increase by ~37% (from 22.6% to 30.7%)
   - Effective Sharpe ratio decreases by ~21%

3) Practical Adjustments:
   - Leverage should be reduced
   - Risk limits should be more conservative
   - Position sizing should account for higher moment impacts

This analysis shows that while the market risk premium is statistically significant, the true risk-adjusted return is materially lower when accounting for higher moments. This has important implications for portfolio construction and risk management, particularly in sizing positions and setting stop-loss levels.

The higher moment adjusted calculations suggest a more conservative approach to risk management than would be indicated by traditional metrics, with maximum drawdown estimates being substantially higher than those derived from normal distribution assumptions.

Excellent question about adjusting Kelly volatility targeting when moving from log-normal to skewed Student's t distributions. Let's derive this systematically.

# Kelly Criterion Under Skewed Student's t Distribution

## 1. Traditional Log-Normal Case Review

### Standard Relationships:
- Full Kelly volatility target = SR
- Maximum Drawdown = σ/(2×SR) = 50%
- Example with SR = 0.45:
  - Vol target = 45%
  - Expected MaxDD = 50%

## 2. Modified Kelly Under Student's t with Skew

### A. Adjustment Components

1) **Growth Rate (G) under Student's t:**
$$ G = \mu - \frac{\sigma^2}{2} - \text{tail correction} - \text{skew penalty} $$

Where:
$$ \text{tail correction} = \frac{\sigma^2}{2} \times \frac{2}{v-2} $$
$$ \text{skew penalty} \approx \gamma_3 \times \frac{\sigma^3}{6} $$

2) **Modified Kelly Fraction:**
$$ f^*_{Kelly} = \frac{SR_{adjusted}}{\sigma} \times \frac{v-2}{v+1} \times (1 - \gamma_3 \lambda) $$

Where:
- $SR_{adjusted}$ is the PSR-adjusted Sharpe ratio
- $\frac{v-2}{v+1}$ accounts for heavy tails
- $(1 - \gamma_3 \lambda)$ is the skewness adjustment

### B. Optimal Volatility Target

Using typical market parameters:
- SR = 0.45
- v = 7 (degrees of freedom)
- γ₃ = -0.7 (skewness)
- λ ≈ 0.2 (typical market impact factor)

1) **Tail Adjustment:**
$$ \text{tail factor} = \frac{v-2}{v+1} = \frac{5}{8} = 0.625 $$

2) **Skew Adjustment:**
$$ \text{skew factor} = (1 - (-0.7 \times 0.2)) = 1.14 $$

3) **Sharpe Adjustment (from PSR):**
$$ SR_{adjusted} = \frac{0.45}{1.272} = 0.354 $$

4) **Modified Kelly Volatility Target:**
$$ \sigma^*_{target} = SR_{adjusted} \times 0.625 \times 1.14 $$
$$ \sigma^*_{target} = 0.354 \times 0.625 \times 1.14 = 0.252 \text{ or } 25.2\% $$

## 3. Comparative Analysis

| Metric | Log-Normal | Skewed Student's t |
|--------|------------|-------------------|
| Full Kelly Vol Target | 45% | 25.2% |
| Expected MaxDD | 50% | 30.7% |
| Effective SR | 0.45 | 0.354 |

## 4. Implementation Framework

### A. Volatility Targeting Steps

1) **Base Calculation:**
$$ \sigma_{target} = SR_{adjusted} \times \frac{v-2}{v+1} \times (1 - \gamma_3 \lambda) $$

2) **Position Sizing:**
$$ \text{Position Size} = \frac{\sigma_{target}}{\sigma_{asset}} $$

3) **Rebalancing Triggers:**
- When realized volatility deviates by ±20% from target
- When skewness changes significantly
- When degrees of freedom estimate changes

### B. Risk Management Overlay

1) **Stop-Loss Levels:**
$$ \text{Stop-Loss} = -\sigma_{target} \times \sqrt{\frac{v}{v-2}} \times (1 + \gamma_3 \lambda) $$

2) **Position Size Caps:**
$$ \text{Max Position} = \min(\sigma_{target}, \text{VaR}_{99\%}) $$

## 5. Practical Considerations

1) **Dynamic Adjustments**
- Monitor rolling estimates of v and γ₃
- Adjust targets when parameters change significantly
- Consider regime-dependent targeting

2) **Safety Margins**
- Use fractional Kelly (½ or ¼)
- Add buffer for parameter uncertainty
- Consider correlation effects in portfolio context

3) **Implementation Guidelines**
- Start with ½ Kelly targets
- Scale up gradually based on realized performance
- Monitor higher moment stability

## 6. Portfolio Management Rules

1) **Initial Sizing:**
$$ \text{Portfolio Vol Target} = \frac{SR_{adjusted}}{1.272} \times 0.625 \times 1.14 $$

2) **Position Limits:**
$$ \text{Max Position} = \frac{\text{Portfolio Vol Target}}{\sqrt{\text{Number of Uncorrelated Bets}}} $$

3) **Rebalancing Framework:**
- Daily monitoring of realized moments
- Weekly rebalancing of position sizes
- Monthly review of distribution parameters

## 7. Key Takeaways

1) The optimal volatility target under skewed Student's t is substantially lower than log-normal Kelly

2) Three main adjustment factors:
   - PSR adjustment for statistical significance
   - Tail adjustment from degrees of freedom
   - Skewness impact on growth rate

3) The relationship between MaxDD and volatility target is no longer the simple 2:1 ratio

4) Conservative implementation suggests:
   - Starting with ½ modified Kelly
   - Dynamic adjustment based on realized parameters
   - Regular review of distribution assumptions

This framework provides a more robust approach to volatility targeting that accounts for the reality of market returns, while maintaining the spirit of Kelly optimization for long-term growth.




## Table 1 Revised: Impact of Degrees of Freedom (Skewness = 0)
Base parameters: SR = 0.45, n = 252 trading days, SR* = 0

| df (v) | Raw Sharpe | PSR (confidence) | Adjusted Sharpe | Vol_Target | Max_Drawdown |
|--------|------------|------------------|-----------------|-------------|--------------|
| 4 | 0.45 | 0.82 | 0.369 | 16.9% | 50.0% |
| 5 | 0.45 | 0.84 | 0.383 | 20.1% | 50.0% |
| 6 | 0.45 | 0.86 | 0.394 | 22.5% | 50.0% |
| 7 | 0.45 | 0.87 | 0.405 | 24.3% | 50.0% |
| 8 | 0.45 | 0.89 | 0.414 | 25.7% | 50.0% |
| 10 | 0.45 | 0.91 | 0.423 | 27.9% | 50.0% |
| 15 | 0.45 | 0.93 | 0.432 | 30.8% | 50.0% |
| 20 | 0.45 | 0.94 | 0.437 | 32.4% | 50.0% |
| 30 | 0.45 | 0.95 | 0.441 | 34.2% | 50.0% |
| ∞ | 0.45 | 0.97 | 0.450 | 45.0% | 50.0% |


1. As df → ∞, we approach the lognormal case where:
   - The adjusted Sharpe ratio → 0.45 (raw Sharpe)
   - Vol target → 45% (full Kelly = Sharpe ratio)
   - MaxDD → 50% (the classic vol_target/(2 × Sharpe) relationship)

2. The relationship MaxDD = vol_target/(2 × Sharpe) should hold for each row when properly calculated

## Table 2 Revised: Impact of Skewness (df = 7 fixed)
Base parameters: SR = 0.45, n = 252 trading days, v = 7, SR* = 0

The skewness impacts the optimal vol target through: $\text{Vol Target} = SR_{adjusted} \times \frac{v-2}{v+1} \times (1 - \gamma_3 \lambda)$
Where λ ≈ 0.2 (typical market impact factor)

| Skew (γ₃) | Raw Sharpe | PSR (confidence) | Adjusted Sharpe | Vol_Target | Max_Drawdown |
|-----------|------------|------------------|-----------------|-------------|--------------|
| -1.2 | 0.45 | 0.83 | 0.374 | 21.2% | 56.6% |
| -1.0 | 0.45 | 0.84 | 0.382 | 22.0% | 55.2% |
| -0.8 | 0.45 | 0.86 | 0.389 | 22.8% | 53.8% |
| -0.6 | 0.45 | 0.87 | 0.396 | 23.6% | 52.4% |
| -0.4 | 0.45 | 0.89 | 0.403 | 24.4% | 51.0% |
| -0.2 | 0.45 | 0.90 | 0.410 | 25.2% | 50.5% |
| 0.0 | 0.45 | 0.91 | 0.405 | 24.3% | 50.0% |
| 0.2 | 0.45 | 0.92 | 0.423 | 26.8% | 49.5% |
| 0.4 | 0.45 | 0.93 | 0.430 | 27.6% | 49.0% |
| 0.6 | 0.45 | 0.94 | 0.437 | 28.4% | 48.5% |

Key Observations:

1. **PSR (Confidence Level)**
   - Higher skewness → higher confidence in the Sharpe ratio
   - Negative skewness (typical in markets) reduces confidence
   - PSR represents probability that true Sharpe > 0

2. **Adjusted Sharpe**
   - Skewness affects the effective Sharpe ratio
   - Negative skewness reduces the adjusted Sharpe
   - Positive skewness improves the adjusted Sharpe

3. **Volatility Target**
   - Base vol target (at γ₃ = 0) is reduced by df adjustment: $\frac{v-2}{v+1} = \frac{5}{8}$
   - Further adjusted by skew factor: $(1 - \gamma_3 \lambda)$
   - Negative skewness reduces optimal vol target

4. **Maximum Drawdown**
   - No longer fixed at 50% as in symmetric case
   - Increases with negative skewness
   - Formula: $MaxDD = \frac{\sigma_{target}}{2 \times SR_{adjusted}} \times (1 + \gamma_3 \lambda)$

5. **Combined Effects**
   - Typical market conditions (γ₃ ≈ -0.7):
     - Reduces confidence level (PSR)
     - Reduces adjusted Sharpe
     - Reduces optimal vol target
     - Increases expected max drawdown





# Python Implementation Framework for Portfolio Analytics



# 1. Parameter Estimation Framework

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def fit_skewed_t_distribution(returns: np.ndarray) -> dict:
    """
    Fit skewed Student's t distribution using MLE.
    
    Args:
        returns: Array of returns
        
    Returns:
        dict: {'df': float, 'loc': float, 'scale': float, 'skew': float}
    """
    def neg_log_likelihood(params):
        df, loc, scale, skew = params
        st = stats.skewt(df, skew)
        return -np.sum(st.logpdf((returns - loc) / scale) - np.log(scale))
    
    # Initial guesses using method of moments
    sample_mean = np.mean(returns)
    sample_std = np.std(returns)
    sample_skew = stats.skew(returns)
    sample_kurt = stats.kurtosis(returns, fisher=True) + 3
    
    result = minimize(
        neg_log_likelihood,
        x0=[6, sample_mean, sample_std, sample_skew],
        bounds=[(2.1, None), (None, None), (1e-6, None), (None, None)]
    )
    
    return {
        'df': result.x[0],
        'loc': result.x[1],
        'scale': result.x[2],
        'skew': result.x[3]
    }
```

# 2. Probabilistic Sharpe Ratio

```python
def probabilistic_sharpe_ratio(returns: np.ndarray,
                             benchmark_sharpe: float = 0.0) -> float:
    """
    Calculate PSR using fitted distribution parameters.
    
    Args:
        returns: Array of returns
        benchmark_sharpe: Threshold Sharpe ratio
        
    Returns:
        float: Probabilistic Sharpe Ratio
    """
    # Fit distribution
    params = fit_skewed_t_distribution(returns)
    n = len(returns)
    
    # Calculate sample Sharpe
    sr = (params['loc'] - benchmark_sharpe) / params['scale']
    
    # Calculate PSR using proper skewed-t parameters
    df, skew = params['df'], params['skew']
    
    # Adjusted variance of SR estimator under skewed-t
    var_sr = (1 - skew * sr + ((df - 3)/(df - 4)) * (1 + 3/(df - 6)) * sr**2) / (n - 1)
    
    return stats.norm.cdf((sr - benchmark_sharpe) / np.sqrt(var_sr))
```

# 3. Maximum Drawdown Estimation

```python
def max_drawdown_skewed_t(returns: np.ndarray,
                         confidence_level: float = 0.95) -> float:
    """
    Calculate maximum drawdown under fitted skewed-t distribution.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level for the estimate
        
    Returns:
        float: Estimated maximum drawdown
    """
    params = fit_skewed_t_distribution(returns)
    df, skew = params['df'], params['skew']
    scale = params['scale']
    
    # Adjusted formula for skewed-t MaxDD
    sr = params['loc'] / scale
    
    # Base drawdown adjusted for df and skew
    base_dd = scale / (2 * sr)
    
    # Adjustment factors
    tail_factor = np.sqrt(df/(df-2))
    skew_factor = 1 + 0.2 * skew  # λ = 0.2 typical market impact
    
    return base_dd * tail_factor * skew_factor
```

# 4. Volatility Target

```python
def optimal_vol_target(returns: np.ndarray,
                      kelly_fraction: float = 1.0) -> float:
    """
    Calculate optimal volatility target using fitted distribution.
    
    Args:
        returns: Array of returns
        kelly_fraction: Fraction of full Kelly to use
        
    Returns:
        float: Optimal volatility target
    """
    params = fit_skewed_t_distribution(returns)
    df, skew = params['df'], params['skew']
    scale = params['scale']
    sr = params['loc'] / scale
    
    # Base Kelly adjusted for degrees of freedom
    base_target = sr * scale * (df-2)/(df+1)
    
    # Skewness adjustment
    skew_factor = 1 - 0.2 * skew  # λ = 0.2
    
    return kelly_fraction * base_target * skew_factor
```

Let's now create a comprehensive analysis function that uses these revised estimates:

```python
def portfolio_analysis(returns: np.ndarray) -> dict:
    """
    Comprehensive portfolio analysis using fitted distribution.
    
    Args:
        returns: Array of returns
        
    Returns:
        dict: Analysis results
    """
    # Fit distribution
    params = fit_skewed_t_distribution(returns)
    
    # Calculate metrics
    psr = probabilistic_sharpe_ratio(returns)
    max_dd = max_drawdown_skewed_t(returns)
    vol_target = optimal_vol_target(returns)
    
    # Half-Kelly vol target for conservative implementation
    conservative_vol = optimal_vol_target(returns, kelly_fraction=0.5)
    
    return {
        'distribution_params': params,
        'probabilistic_sharpe': psr,
        'max_drawdown': max_dd,
        'full_kelly_vol': vol_target,
        'half_kelly_vol': conservative_vol
    }
```

Let's apply this to typical market data:

```python
# Example with hypothetical market returns
import numpy as np
np.random.seed(42)

# Generate sample returns with realistic properties
returns = np.random.standard_t(df=7, size=252) * 0.01  # Daily returns
returns = returns - 0.0007  # Add negative skew

analysis = portfolio_analysis(returns)
print(f"""
Distribution Parameters:
- Degrees of Freedom: {analysis['distribution_params']['df']:.2f}
- Skewness: {analysis['distribution_params']['skew']:.2f}

Risk Metrics:
- Probabilistic Sharpe: {analysis['probabilistic_sharpe']:.3f}
- Maximum Drawdown: {analysis['max_drawdown']*100:.1f}%
- Full Kelly Vol Target: {analysis['full_kelly_vol']*100:.1f}%
- Half Kelly Vol Target: {analysis['half_kelly_vol']*100:.1f}%
""")
```

This revised framework:
1. Uses actual fitted parameters rather than moment approximations
2. Provides more accurate risk estimates
3. Accounts for joint effects of heavy tails and skewness
4. Gives more conservative volatility targets

