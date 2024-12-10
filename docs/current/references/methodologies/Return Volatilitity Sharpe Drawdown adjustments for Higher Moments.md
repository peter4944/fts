# Arithmetic to Geometric Returns, Higher Moments, and Volatility Targeting: Normal vs Skewed Student-t Distributions

## 1. Introduction

Investment returns rarely follow a perfect normal (log-normal) distribution. Real-world returns often exhibit skewness and excess kurtosis (fat tails). These deviations have significant implications for converting arithmetic returns to geometric returns, estimating volatility, and determining the optimal leverage or volatility target to maximize long-term growth. Traditional measures like the Sharpe ratio and maximum theoretical drawdown (MTD) must be adjusted to incorporate these higher moments and more realistic return distributions, such as the skewed Student-t distribution.

This note:
1. Reviews the arithmetic-to-geometric return adjustment for normal and non-normal distributions.
2. Incorporates adjustments for skewness and kurtosis into volatility, Sharpe ratio, and maximum theoretical drawdown calculations.
3. Introduces the Growth-to-Drawdown Ratio (GDR) as a more comprehensive performance measure.
4. Demonstrates how the Kelly criterion (optimal volatility targeting) must be modified when returns exhibit skewness and fat tails.
5. Provides tables illustrating parameter impact examples.
6. Concludes with Python code implementing all discussed formulas.

We first summarize all final formulas under normal (including higher moments) and skewed Student-t assumptions, then discuss their implications.

## 2. Key Definitions and Notation

- **Arithmetic Return ($R_a$)**: The simple average of period returns.
- **Volatility ($\sigma$)**: Standard deviation of returns.
- **Skewness ($\gamma_3$)**: Measure of asymmetry. Negative skew implies a higher probability of large negative returns.
- **Kurtosis ($\gamma_4$)**: Measure of tail thickness. Excess kurtosis = $\gamma_4 - 3$.
- **Degrees of Freedom ($v$)**: Parameter in Student-t distributions controlling tail heaviness.
- **Skewed Student-t Distribution**: Adds skewness to the Student-t distribution, capturing both fat tails and asymmetric returns.

## 3. Arithmetic to Geometric Returns

### Normal Distribution (with Higher Moments)

For a portfolio with arithmetic return $R_a$ and volatility $\sigma$:

1. **Variance Drag**:
   $$\text{Variance Drag} = \frac{\sigma^2}{2}.$$

2. **Skewness Drag**:
   $$\text{Skewness Drag} = \frac{\gamma_3 \sigma^3}{6}.$$

3. **Kurtosis Drag** (Excess Kurtosis = $\gamma_4 - 3$):
   $$\text{Kurtosis Drag} = \frac{(\gamma_4 - 3) \sigma^4}{24}.$$

4. **Adjusted Geometric Return**:
   $$R_{g,adj} = R_a - \frac{\sigma^2}{2} - \frac{\gamma_3 \sigma^3}{6} - \frac{(\gamma_4 - 3)\sigma^4}{24}.$$

### Skewed Student-t Distribution

Under a skewed Student-t distribution with parameters $(v, \gamma_3)$, the tails and skewness introduce additional penalties. Approximating the growth rate impact:

1. **Fat Tail Adjustment**:
   $$ \text{Fat Tail Cost} = \frac{\sigma^2}{2} \times \frac{2}{v-2}. $$

2. **Skewness Adjustment**:
   $$ \text{Skewness Cost} \approx \gamma_3 \frac{\sigma^3}{6}.$$

3. **Adjusted Geometric Return (Skewed-t)**:
   $$ R_{g,adj} = R_a - \frac{\sigma^2}{2} - \frac{\sigma^2}{2} \times \frac{2}{v-2} - \gamma_3 \frac{\sigma^3}{6}. $$

## 4. Volatility Adjustments

### Normal with Higher Moments

The presence of skewness and kurtosis affects implied volatility. An approximate adjustment under normality with higher moments:

$$ \sigma_{adj} \approx \sigma \sqrt{1 + \frac{(\gamma_4 - 3)\sigma^2}{4} + \frac{\gamma_3^2 \sigma^2}{6}}. $$

### Skewed Student-t Distribution



$$ \sigma_{t,adj} = \sigma\sqrt{\frac{v}{v-2}}. $$

Heavy tail adjustment:
$$ \sigma_t = \sigma \sqrt{\frac{v}{v-2}}. $$



## 5. Sharpe Ratio Adjustments

### Normal Distribution (with higher moments)

**Adjusted Geometric Sharpe Ratio:**
$$ SR_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}}. $$

### Skewed Student-t Distribution

Once we have $R_{g,adj}^{(t)}$ and $\sigma_{t,adj}$, the adjusted Sharpe ratio is simply:
   $$
   SR_{t,adj} = \frac{R_{g,adj}^{(t)} - R_f}{\sigma_{t,adj}}.
   $$

## 6. Maximum Theoretical Drawdown (MTD) Adjustments

### MTD Under Normal Distribution (Base Case)
For a log-normal world (no skew, no excess kurtosis) and full Kelly leverage:
- Full Kelly volatility target = Sharpe ratio (SR)
- Maximum Theoretical Drawdown (MTD) = 50%

Mathematically:
$$ MTD_{\text{normal}} = \frac{\sigma}{2 \times SR} = 50\% \text{ when } \sigma = SR. $$

### MTD Under Normal Distribution With Skew and Kurtosis
When we incorporate skewness ($\gamma_3$) and excess kurtosis ($\gamma_4 - 3$), the geometric return, volatility, and Sharpe ratio are adjusted. The adjusted geometric Sharpe ratio under normal assumptions with higher moments is:
$$ SR_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}}. $$

If we scale volatility to the adjusted full Kelly allocation, the theoretical MTD formula remains structurally the same but uses the adjusted parameters:
$$ MTD_{\text{normal,adj}} = \frac{\sigma_{adj}}{2 \times SR_{g,adj}}. $$

Here:
- $\sigma_{adj}$ is the volatility adjusted for skew and kurtosis.
- $SR_{g,adj}$ is the adjusted geometric Sharpe ratio that accounts for the drags from skewness and kurtosis.

Thus, instead of simply 50%, the MTD can now differ depending on how these higher moments reduce $SR_{g,adj}$ and affect $\sigma_{adj}$.

### Skewed Student-t Distribution

Full Kelly under skewed Student-t is lower than SR. Using optimal volatility target and distributions:

$$
   MTD_{skewed-t} = 0.5 \times \sqrt{\frac{v}{v-2}} \times (1 - \gamma_3 \lambda).
   $$

As $\gamma_3$ < 0 typically, MTD often increases due to tail and skew effects.

## 7. Kelly Criterion and Volatility Targeting Adjustments

### Volatility Target Under Normal Distribution (Base Case)
Under log-normal assumptions:
- Full Kelly volatility target = $SR$.

### Volatility Target Under Normal Distribution With Skew and Kurtosis
When returns are not perfectly normal, we have the adjusted geometric return $R_{g,adj}$ and adjusted volatility $\sigma_{adj}$. The adjusted Kelly fraction (full Kelly) is:
$$ f^*_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}}. $$

This fraction $f^*_{g,adj}$ represents the optimal (full Kelly) fraction of wealth to invest, and thus:
$$ \sigma_{\text{target, normal, adj}} = f^*_{g,adj} = \frac{R_{g,adj} - R_f}{\sigma_{adj}}. $$

In other words, under normal assumptions but with higher moments, the volatility target that maximizes long-term growth is no longer simply equal to the Sharpe ratio. It must be adjusted downward to account for the lower adjusted geometric Sharpe caused by skew and excess kurtosis.

### For skewed Student-t:

1. **Adjusted Optimal Volatility Target**:
   $$ \sigma_{target} = SR_{adjusted} \times \frac{v-2}{v+1} \times (1 - \gamma_3 \lambda). $$

Negative skewness reduces the optimal volatility target significantly.

## 8. Growth-to-Drawdown Ratio (GDR)

GDR links geometric growth to theoretical drawdown risk, providing a more comprehensive measure:

**GDR**:
$$ GDR = \frac{R_{g,adj}}{MTD}. $$

Under skewed-t, using the adjusted geometric return and MTD:

$$ GDR_{skewed-t} = \frac{R_{g,adj}}{ \frac{\sigma_{target}}{2 \times SR_{adjusted}} \times \sqrt{\frac{v}{v-2}} \times (1 + \gamma_3 \lambda)}. $$

A higher GDR indicates better long-term growth per unit of drawdown risk.

## 9. Parameter Impact Examples

### Table: Normal Distribution with Higher Moments

| Excess Kurtosis | Skew (γ₃) | $R_{g,adj}$ | $\sigma_{adj}$ | Vol Target (Adj Kelly) | $SR_{g,adj}$ | MTD (Adj) | GDR |
|-----------------|-----------|-------------|----------------|------------------------|--------------|-----------|-----|
| 0.0 (Normal)    | 0.0       | 5.0%        | 20.0%          | 25.0%                 | 0.25         | 50.0%     | 0.10 |
| 1.0             | 0.0       | 4.2%        | 21.5%          | 19.5%                 | 0.195        | 51.3%     | 0.08 |
| 1.0             | -0.4      | 3.6%        | 22.3%          | 16.2%                 | 0.162        | 53.0%     | 0.068|
| 2.0             | -0.6      | 2.8%        | 23.5%          | 12.5%                 | 0.125        | 55.5%     | 0.050|

**Interpretation:**
- As excess kurtosis and negative skew increase, $R_{g,adj}$ declines, $SR_{g,adj}$ falls, and thus the optimal vol target decreases.
- MTD slightly increases as higher moments worsen the risk profile.
- GDR drops as conditions become less favorable.

### Table: Skewed Student-t Distribution

| DoF (v) | Skew (γ₃) | $R_{g,adj}$ | $\sigma_{adj}$ | Vol Target (Adj Kelly) | $SR_{g,adj}$ | MTD (Adj) | GDR |
|---------|-----------|-------------|----------------|------------------------|--------------|-----------|-----|
| 12      | 0.0       | 4.7%        | 20.8%          | 23.0%                 | 0.23         | 50.0%     | 0.094|
| 9       | 0.0       | 4.2%        | 21.6%          | 20.0%                 | 0.20         | 50.5%     | 0.083|
| 9       | -0.3      | 3.8%        | 22.5%          | 17.0%                 | 0.17         | 52.3%     | 0.073|
| 9       | -0.45     | 3.6%        | 23.0%          | 15.0%                 | 0.15         | 53.2%     | 0.068|

**Interpretation:**
- Lower DoF (heavier tails) reduces $R_{g,adj}$ and $SR_{g,adj}$, lowering the optimal volatility target.
- Negative skew further reduces the optimal vol target and slightly increases MTD.
- GDR declines as distributional conditions worsen (lower DoF, more negative skew).

In both sets of scenarios (normal with higher moments vs. skewed Student-t), the presence of skewness and/or fat tails reduces the adjusted geometric Sharpe ratio, lowers the optimal volatility target, increases the MTD beyond the simple 50% rule, and reduces the GDR.
## 10. Practical Implications

- **Risk Management**: Higher moments necessitate lower volatility targets to avoid catastrophic drawdowns.
- **Parameter Estimation**: Reliable fitting of skewed Student-t parameters is critical. Use long historical windows and robust estimation methods.
- **Fractional Kelly**: Due to parameter uncertainty and skewness, a fraction (e.g., ½ Kelly) is often safer.
- **Hedging and Tail Risk Strategies**: Consider tail hedges to offset skewness and fat-tail penalties.

## 11. Python Code Appendix

Below is Python code to implement the discussed functions:

```python
import numpy as np
from scipy.stats import norm

# Example utilities from the original and extended note

def variance_drag(vol: float) -> float:
    return (vol**2)/2

def skew_drag(skew: float, vol: float) -> float:
    return (skew * vol**3)/6

def kurtosis_drag(excess_kurt: float, vol: float) -> float:
    return (excess_kurt * vol**4)/24

def geometric_return_adjusted(R_a: float, vol: float, skew: float, kurt_excess: float) -> float:
    return R_a - variance_drag(vol) - skew_drag(skew, vol) - kurtosis_drag(kurt_excess, vol)

def adjusted_volatility_normal(vol: float, skew: float, kurt_excess: float) -> float:
    return vol * np.sqrt(1 + (kurt_excess * vol**2)/4 + (skew**2 * vol**2)/6)

def student_t_volatility(vol: float, v: float) -> float:
    return vol * np.sqrt(v/(v-2))

def skewed_t_vol(vol: float, v: float) -> float:
    return student_t_volatility(vol, v)

def adjusted_geometric_return_student_t(R_a: float, vol: float, v: float, skew: float) -> float:
    tail_cost = (vol**2)/2 * (2/(v-2))
    skew_cost = skew * (vol**3)/6
    return R_a - (vol**2)/2 - tail_cost - skew_cost

def sharpe_ratio(R: float, vol: float, rf: float=0) -> float:
    return (R - rf)/vol

def max_theoretical_drawdown_normal_full_kelly() -> float:
    return 0.5  # 50%

def max_theoretical_drawdown_skewed_t(vol_target: float, SR_adj: float, v: float, skew: float, lam: float=0.2) -> float:
    # At the chosen vol target (adjusted Kelly)
    return (vol_target/(2*SR_adj))*np.sqrt(v/(v-2))*(1-skew*lam)

def optimal_vol_target_skewed_t(SR_adj: float, v: float, skew: float, lam: float=0.2) -> float:
    return SR_adj * ((v-2)/(v+1)) * (1 + skew*lam)

def GDR(R_g_adj: float, MTD: float) -> float:
    return R_g_adj/MTD

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

## 12. Conclusions

- Converting arithmetic to geometric returns requires adjusting for volatility, skewness, and kurtosis.
- Higher moments significantly reduce achievable Sharpe ratios, increase maximum drawdowns, and lower the optimal Kelly volatility target.
- Under skewed Student-t distributions (reflecting negative skew and fat tails), the ideal leverage is materially lower than the simple "vol = SR" rule suggested by log-normal assumptions.
- Incorporating GDR helps investors understand the trade-off between long-term growth and drawdown risk.
- Practitioners should use conservative fractions of the adjusted Kelly target and consider additional tail risk hedging strategies.

By considering these realistic adjustments, investors can better target sustainable growth, manage extreme risk, and optimize their portfolios under more accurate return distribution assumptions.
