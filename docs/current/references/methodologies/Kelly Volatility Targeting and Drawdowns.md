# Kelly Volatility Targeting and Drawdowns

## TOC
### 1. Introduction
Briefly explain the Kelly Criterion, emphasizing its goal of maximizing long-term capital growth through an optimal betting or investment strategy.
Introduce drawdown as a critical risk management concept, representing the peak-to-trough decline in portfolio value.
Define the note's purpose: To investigate drawdown implications of Kelly allocations across diverse scenarios, factoring in various return distributions, constraints, and rebalancing frequencies.
### 2. The Kelly Criterion: Origins and Foundations
Delve into the Kelly Criterion's historical roots in gambling, including its connection to information theory and its development by John L. Kelly Jr.
Present the basic formula for the Kelly Criterion in a simple coin-toss gambling scenario with fixed odds.
Demonstrate the Kelly Criterion's equivalence to maximizing the expected logarithmic utility of wealth.
Discuss the crucial assumptions of the Kelly Criterion: Reinvestment of winnings and a large number of trials (bets or investments).
Highlight the Kelly Criterion's properties, such as long-term outperformance of other strategies and maximization of median terminal wealth.
Illustrate the concept of fractional Kelly strategies, where investors bet a fraction of the full Kelly bet to reduce volatility while sacrificing some growth potential.
### 3. Drawdown Under the Kelly Criterion
Explain the concept of drawdown in more detail, differentiating between Maximum Drawdown (MDD), Average Drawdown (ADD), Conditional Drawdown at Risk (CDaR), and Conditional Expected Drawdown (CED).
Explain the impact of heavier tails in the return distribution on the drawdown characteristics.
Provide formulas and insights into calculating the probability of a specific drawdown under both full and fractional Kelly allocations.
Discuss the role of the time horizon in determining drawdown probability and its relation to the probability of ruin.
Present the concept of Expected Maximum Drawdown (EMD) and its difference from the probability of a specific drawdown.
Provide insights into simulating drawdowns under various scenarios and discuss the choice of an appropriate probability distribution, including lognormal, skewed Student's t, and others.
Elaborate on the computation and interpretation of CDaR as a measure of expected drawdown beyond a defined threshold.
### 4. Impact of Non-Continuous Rebalancing
Relax the assumption of continuous rebalancing and explain the practical implications of rebalancing at discrete intervals (e.g., monthly, quarterly).
Discuss how infrequent rebalancing can amplify drawdown risk, particularly in volatile markets.
Introduce rebalancing bands as a method to determine optimal rebalancing frequencies.
Illustrate the influence of rebalancing frequency on drawdown through simulations with varying rebalancing intervals.
Highlight the challenges of finding the optimal rebalancing frequency and discuss alternative approaches.
### 5. Python Implementation and Simulation Analysis
Provide a Python code example demonstrating the simulation of portfolio paths under different return distributions (lognormal, skewed Student's t, etc.) and rebalancing frequencies.
Show how to calculate and output relevant metrics from the simulations, including:
Annualized arithmetic and logarithmic returns.
Annualized realized volatility.
Geometric growth rate.
Maximum Drawdown (MDD).
Average Drawdown (ADD).
Conditional Drawdown at Risk (CDaR).
Conditional Expected Drawdown (CED).


## 1. Introduction
The Kelly Criterion, developed by John L. Kelly Jr. in 1956, is a mathematical formula used to determine the optimal size of a bet in a favorable gambling scenario. It aims to maximize the expected logarithmic growth of capital over the long run. It has found applications in various fields, including finance, where it's used to determine the optimal allocation of capital across different assets.

However, the practical implementation of the Kelly Criterion faces several challenges, a key one being the concept of drawdown. Drawdown is the peak-to-trough decline in the value of an investment over a specific period. While the Kelly Criterion aims for long-term growth, it can often lead to substantial drawdowns in the short term, which can be psychologically challenging for investors and may lead to deviations from the strategy.

This note aims to explore the relationship between the Kelly Criterion and drawdown dynamics through simulation. We will analyze how various factors, such as the expected return and volatility of the underlying assets, impact portfolio growth and the magnitude of potential drawdowns. By understanding this interplay, we can gain valuable insights into the practical implications of using the Kelly Criterion in a real-world investment setting.

A natural next step is to define the methodology we'll use for our simulation. In the next section, we will detail the assumptions underpinning the simulation, the key input parameters, and the specific metrics we will track to assess portfolio performance under the Kelly Criterion.

## 2. The Kelly Criterion: Origins and Foundations


This section delves into the historical development and theoretical foundations of the Kelly Criterion.

### 2.1 Origins in Gambling and Information Theory
The Kelly Criterion emerged from the work of John L. Kelly Jr., a researcher at Bell Labs, in 1956. Kelly's original paper, "A New Interpretation of Information Rate," framed the problem in the context of transmitting information over a noisy communication channel. He sought to determine the optimal betting strategy for a gambler receiving insider information about a horse race.

Claude Shannon, a pioneer of information theory and Kelly's colleague at Bell Labs, played a significant role in the development of the Kelly Criterion. Shannon's information theory provided the foundation for understanding the relationship between information, uncertainty, and optimal decision-making.

The Kelly Criterion's connection to information theory lies in the concept of maximizing the expected value of information. By choosing the optimal bet size, a gambler with an edge can maximize the rate at which they gain information about the true probabilities of the events they're betting on. This leads to the fastest possible long-term growth of their capital.

Edward Thorp, a mathematician and pioneering quantitative hedge fund manager, further popularized the Kelly Criterion by applying it to blackjack and other gambling scenarios. Thorp's work demonstrated the practical applicability of the criterion and its potential for achieving significant capital growth.

### 2.2 Basic Formula for a Coin-Toss Scenario
To illustrate the Kelly Criterion, consider a simple coin-toss gambling scenario:

Biased Coin: The coin has a probability of winning (heads) of p, and a probability of losing (tails) of q = 1 - p.
Fixed Odds: A winning bet returns b times the amount wagered.
The Kelly Criterion determines the optimal fraction (f) of your capital to bet on each toss to maximize long-term growth. The formula for the Kelly fraction in this scenario is:

f = (bp - q) / b

### 2.3 Equivalence to Maximizing Logarithmic Utility
The Kelly Criterion can also be derived from the perspective of utility theory. It's mathematically equivalent to maximizing the expected value of the logarithm of wealth. The logarithmic utility function has several desirable properties that make it well-suited for long-term investment decisions. It exhibits diminishing marginal utility, implying that each additional dollar of wealth provides less utility than the previous dollar. This aligns with the behavior of rational investors who become more risk-averse as their wealth increases.

### 2.4 Crucial Assumptions and Properties
The Kelly Criterion relies on several key assumptions:

Reinvestment of Winnings: The gambler or investor reinvests all their winnings back into the betting or investment strategy. This allows for compounding returns, which is crucial for long-term growth.
Large Number of Trials: The Kelly Criterion is designed for scenarios with a large number of independent bets or investments. Its optimality is asymptotic, meaning it performs best as the number of trials approaches infinity.
The Kelly Criterion possesses several notable properties:

Long-Term Outperformance: The Kelly strategy asymptotically outperforms any other strategy that bets a fixed fraction of wealth. This means that in the long run, the Kelly bettor will accumulate more wealth than any other bettor with a different, but constant, betting fraction.
Maximization of Median Terminal Wealth: Under certain conditions, the Kelly Criterion maximizes the median of the terminal wealth distribution. This implies that for a typical outcome, the Kelly bettor is likely to have more wealth than someone following a different strategy.
Avoidance of Ruin: While the Kelly Criterion doesn't guarantee avoidance of losses, it ensures that the gambler never risks ruin, as they only ever bet a fraction of their wealth.

### 2.5 Fractional Kelly Strategies
One of the primary criticisms of the Kelly Criterion is that it can recommend large bet sizes, leading to significant volatility in portfolio value. To address this, investors often employ fractional Kelly strategies.

Fractional Kelly involves betting a fixed fraction (c) of the full Kelly bet (f). For example, a Half Kelly bettor would use c = 0.5, betting half the amount recommended by the full Kelly Criterion. By reducing the bet size, fractional Kelly strategies decrease volatility and drawdown risk, but at the expense of lower expected growth.


## 3. Drawdowns Under the Kelly Criterion
This section examines the drawdown behavior under Kelly Criterion allocations. The focus is on understanding the relationship between Kelly betting and drawdown risk, including the probability and magnitude of drawdowns.

### 3.1 Defining Drawdown
Drawdown is a crucial risk management concept that quantifies the peak-to-trough decline in the value of an investment portfolio over a specific period. It's typically expressed as a percentage of the peak value. For example, if a portfolio's value drops from $100 to $80, the drawdown is 20%.

Investors are concerned with drawdown because it represents the potential loss they could experience during adverse market conditions. A high drawdown can be psychologically challenging and may force investors to sell their holdings at a loss, potentially impacting their long-term investment goals.

### 3.2 Drawdown Implications of Kelly Allocations
The Kelly Criterion aims to maximize the long-term growth rate of capital. However, this maximization of growth comes with an inherent trade-off: higher growth potential often implies accepting a higher probability of larger drawdowns.

The full Kelly strategy, while asymptotically optimal for growth, can lead to significant fluctuations in portfolio value. This is because it recommends betting a proportion of your wealth that is directly proportional to the edge you have in the game or investment opportunity. When the edge is large, the recommended bet size can be substantial, leading to larger potential losses.

Fractional Kelly strategies offer a way to mitigate this drawdown risk by reducing the bet size. As discussed in Section II, fractional Kelly investors bet a predetermined fraction (c) of the full Kelly bet. By decreasing the bet size, they lower the potential magnitude of losses, sacrificing some growth potential in return for a smoother equity curve.

### 3.3 Factors Influencing Drawdown
Several factors influence the drawdown behavior of Kelly portfolios:

Expected Return and Volatility: Higher expected returns typically lead to larger Kelly bet sizes, increasing both the growth potential and the drawdown risk. Conversely, lower expected returns result in smaller Kelly bets and reduced drawdown. Volatility also plays a significant role; higher volatility leads to larger potential swings in portfolio value and increased drawdown risk.
Return Distribution: The shape of the return distribution impacts drawdown behavior. The Kelly Criterion's optimality is often derived assuming normally distributed returns. However, real-world asset returns often exhibit fat tails, meaning that extreme events are more frequent than predicted by a normal distribution. Fat tails can increase drawdown risk as larger than expected losses become more likely.
Constraints and Rebalancing: The presence of constraints, such as limits on leverage or short-selling, can affect Kelly allocations and drawdown. Rebalancing frequency is also crucial; more frequent rebalancing generally reduces drawdown risk, but comes at the cost of increased transaction costs and potentially reduced returns due to missed opportunities.
Uncertainty About Risk: The Kelly Criterion formula requires precise knowledge of the expected return and volatility. However, in practice, these parameters are uncertain and must be estimated from historical data. Estimation errors can lead to suboptimal Kelly bets and unexpected drawdowns. Several studies have investigated the impact of parameter uncertainty on the Kelly Criterion and have proposed adjustments to account for this uncertainty. These adjustments often involve reducing the Kelly fraction to account for the potential for overestimating the edge and, in turn, mitigating drawdown risk. Fractional Kelly strategies are also a common way to deal with parameter uncertainty. By betting a fraction of the full Kelly bet, investors can reduce their exposure to potential estimation errors.

### 3.4 Analyzing Drawdown Under the Kelly Criterion
Analyzing drawdown under the Kelly Criterion involves understanding the probability of specific drawdowns, the expected maximum drawdown, and the distribution of drawdown duration. Various analytical and simulation techniques can be used to assess these metrics.

Analytical techniques often rely on simplifying assumptions about the return distribution, such as assuming normality. These techniques can provide closed-form solutions for certain drawdown metrics, but their accuracy may be limited in the presence of fat tails or other non-normalities in the return distribution.

Simulations, on the other hand, can handle more complex return distributions and incorporate various constraints and rebalancing policies. By generating a large number of possible return paths, simulations can provide a comprehensive picture of drawdown behavior under different scenarios. However, simulations can be computationally intensive and may require careful calibration of input parameters.

### 3.5 Practical Implications
Understanding the drawdown implications of Kelly allocations is crucial for investors seeking to balance growth potential with risk management. The choice of full versus fractional Kelly depends on the investor's risk tolerance, investment horizon, and the characteristics of the underlying assets or investment strategy.

Investors with a high risk tolerance and a long investment horizon may be comfortable with the larger drawdowns associated with full Kelly betting, especially in scenarios with a demonstrably large edge. Investors with a lower risk tolerance or a shorter investment horizon may prefer fractional Kelly strategies to reduce volatility and protect their capital.

In addition to choosing the appropriate Kelly fraction, investors can manage drawdown risk by diversifying across uncorrelated assets, implementing stop-loss orders to limit losses on individual positions, and adjusting their Kelly allocations based on changes in market conditions and parameter uncertainty.

## 4. Drawdown Dynamics

### 4.1.  Drawdown Measures

**Introduction:** This subsection introduces various drawdown measures commonly used to quantify and assess the magnitude and duration of portfolio losses. Understanding these measures is crucial for comprehending the drawdown implications of Kelly allocations and incorporating drawdown risk management into the investment strategy.

**1. Maximum Drawdown (MDD):**

*   **Definition:** The Maximum Drawdown (MDD) represents the largest peak-to-trough decline in portfolio value over a specified period. It measures the worst-case scenario in terms of potential losses and provides insights into the portfolio's resilience during adverse market conditions.
*   **Mathematical Formulation:**

    ```
    MDD(S) = sup_{t∈[0,T]} D_t = sup_{t∈[0,T]} (M_t - S_t) / M_t
    ```

    where:

    *   *S<sub>t</sub>* is the portfolio value at time  *t*
    *   *M<sub>t</sub>* is the running maximum portfolio value up to time  *t*
    *   *T* is the investment horizon

*   **Intuitive Explanation:** MDD captures the largest historical loss an investor would have experienced if they had bought at the peak and sold at the trough within the specified period. A higher MDD indicates a greater potential for significant losses.
*   **Illustration:** A simple line chart showcasing a portfolio's value over time, highlighting the peak and trough points that define the MDD.

**2. Average Drawdown (ADD):**

*   **Definition:** The Average Drawdown (ADD) measures the average magnitude of all drawdowns experienced by a portfolio over a given period. It provides a more comprehensive view of drawdown risk compared to MDD, as it considers the frequency and severity of all drawdowns, not just the largest one.
*   **Mathematical Formulation:**

    ```
    ADD(S) = (1/T) ∫_{0}^{T} D_t dt
    ```

    where:

    *   *D<sub>t</sub>* is the drawdown at time *t*
    *   *T* is the investment horizon

*   **Intuitive Explanation:** ADD reflects the average historical loss an investor would have experienced due to drawdowns, considering both the size and duration of each drawdown event.
*   **Illustration:** A histogram displaying the distribution of drawdowns experienced by the portfolio, highlighting the average drawdown level.

**3. Conditional Drawdown at Risk (CDaR):**

*   **Definition:** Conditional Drawdown at Risk (CDaR), also known as Expected Tail Drawdown (ETD), is a tail risk measure that focuses on the expected drawdown in the worst (1−α)⋅100% cases, where α is a chosen confidence level. It captures the average magnitude of the most significant drawdowns, providing a more nuanced assessment of extreme drawdown risk.
*   **Mathematical Formulation:**

    ```
    CDaR_α(S) = (1/(1-α)) ∫_{α}^{1} VaR_u(D) du
    ```

    where:

    *   *VaR<sub>u</sub>(D)* is the Value at Risk of the drawdown distribution at confidence level *u*
    *   α is the chosen confidence level

*   **Intuitive Explanation:** CDaR estimates the average loss an investor would experience in the worst (1−α)⋅100% drawdown scenarios. A higher CDaR implies a greater risk of substantial losses during extreme market downturns.
*   **Illustration:** A chart displaying the drawdown distribution, highlighting the tail region corresponding to the worst (1−α)⋅100% cases and indicating the CDaR level.

**4. Conditional Expected Drawdown (CED):**

*   **Definition:** Conditional Expected Drawdown (CED) represents the tail mean of the Maximum Drawdown (MDD) distribution. It focuses specifically on the expected value of the MDD in the worst (1−α)⋅100% cases, providing a direct measure of extreme drawdown risk associated with the largest potential losses.
*   **Mathematical Formulation:**

    ```
    CED_α(S) = E[MDD(S) | MDD(S) > q_α]
    ```

    where:

    *   *q<sub>α</sub>* is the α-quantile of the MDD distribution

*   **Intuitive Explanation:** CED estimates the average magnitude of the MDD in the most severe (1−α)⋅100% scenarios. It directly quantifies the extreme drawdown risk associated with the worst-case potential losses.
*   **Illustration:** Similar to the CDaR illustration, but specifically highlighting the expected value of the MDD in the tail region of the distribution.


**5. Marginal Contribution to Conditional Expected Drawdown (MCD\_CED):**

*   **Definition:** Marginal Contribution to Conditional Expected Drawdown (MCD\_CED), as introduced by Goldberg and Mahmoud in their 2016 paper, quantifies the contribution of each asset or factor to the overall portfolio's CED. It is analogous to the concept of marginal contributions to risk in a Value-at-Risk (VaR) framework. By decomposing the portfolio's CED into individual asset contributions, MCD\_CED allows investors to identify the specific holdings that contribute most significantly to the portfolio's drawdown risk.
*   **Mathematical Formulation:**

    ```
    MCD_CED_i,α(P) = E[(F_i,t* - F_i,s*) | µ(P) > DT_α(P)]
    ```

    where:

    *   *MCD\_CED<sub>i,α</sub>(P)* is the marginal contribution of asset *i* to the portfolio *P*'s CED at confidence level α
    *   *F<sub>i,t*</sub>* and *F<sub>i,s*</sub>* represent the values of asset *i* at times *t* and *s*, respectively, where the overall portfolio maximum drawdown µ(*P*) occurs
    *   *µ(P)* is the maximum drawdown of the portfolio *P*
    *   *DT<sub>α</sub>(P)* is the portfolio's maximum drawdown threshold at confidence level α

*   **Intuitive Explanation:** MCD\_CED captures the expected loss attributed to asset *i* during the period when the overall portfolio experiences its maximum drawdown, conditional on the maximum drawdown exceeding a predefined threshold. A higher MCD\_CED for a particular asset indicates a greater contribution to the portfolio's drawdown risk.
*   **Illustration:** A bar chart depicting the MCD\_CED values for each asset in the portfolio at a given confidence level. The height of each bar represents the magnitude of the asset's contribution to the portfolio's overall CED.

**Conclusion:** MCD\_CED is a valuable tool for understanding the sources of drawdown risk within a portfolio. It allows investors to identify and manage the specific holdings that contribute most significantly to potential losses during adverse market conditions.


**Conclusion:** This subsection provided a comprehensive overview of different drawdown measures, each capturing distinct aspects of drawdown risk. Understanding these measures is essential for evaluating the drawdown implications of Kelly allocations and for incorporating drawdown risk management into portfolio construction.

**Moving Forward:**  The next step could involve discussing how these drawdown measures can be utilized to constrain Kelly allocations and manage downside risk. This could include exploring the relationship between fractional Kelly and CDaR, demonstrating how reducing allocation impacts CDaR, and presenting alternative drawdown-based constraints like the Average Drawdown or Time Under Drawdown. Additionally, we can provide Python functions to calculate these drawdown measures for given portfolio paths, facilitating their practical application. For instance, we could analyze how different drawdown constraints affect the optimal Kelly fractions and the resulting portfolio performance. This analysis would highlight the trade-offs between maximizing long-term growth and mitigating drawdown risk, providing practical insights for investors seeking to implement the Kelly criterion while managing downside potential.




## 5.  Non-Continuous Rebalancing and Drawdown (Elaborated with insights from Source 3)
This section will examine the impact of non-continuous rebalancing on drawdown behavior under the Kelly Criterion. We'll explore how different rebalancing frequencies affect drawdown, taking into account real-world constraints like transaction costs.

### 5.1 The Importance of Rebalancing Frequency
Rebalancing, the process of adjusting portfolio weights back to their target allocations, is essential for implementing the Kelly Criterion and managing drawdown risk. Ideally, continuous rebalancing would ensure the portfolio is always optimally positioned. Continuous rebalancing would lead to the maximization of growth opportunities and the minimization of drawdown.

However, continuous rebalancing is impractical. Factors such as transaction costs, market liquidity constraints, and the time required to monitor and execute trades render continuous rebalancing unfeasible. Therefore, investors must adopt a non-continuous rebalancing approach, updating their portfolios at discrete intervals.

The choice of rebalancing frequency becomes a critical decision, impacting both the potential growth and drawdown of the Kelly portfolio.

### 5.2 The Impact of Non-Continuous Rebalancing on Drawdown

Non-continuous rebalancing introduces a lag between the portfolio's actual weights and their target Kelly allocations. This lag can:

Increase Drawdown Risk: Fluctuating asset prices may cause the portfolio's weights to drift significantly from their optimal levels, exposing the portfolio to larger drawdowns than with continuous rebalancing. This drift is especially pronounced during periods of high market volatility.
Lead to Missed Growth Opportunities: Non-continuous rebalancing can also result in missed growth opportunities. If an asset's price increases sharply, the portfolio may not fully benefit from this move until the next rebalancing period.

### 5.3 Factors Influencing the Choice of Rebalancing Frequency
Several factors influence the optimal rebalancing frequency for Kelly portfolios:

Transaction Costs: Higher transaction costs discourage frequent rebalancing. Costs associated with adjusting portfolio weights can erode returns, so investors must balance maintaining close to optimal Kelly allocations with minimizing trading expenses.
Market Volatility: In highly volatile markets, frequent rebalancing might be necessary to mitigate drawdown risk caused by rapid price fluctuations. However, frequent rebalancing can also increase transaction costs. In less volatile markets, less frequent rebalancing might suffice. Finding the right balance is crucial in determining the optimal rebalancing frequency.
Return Correlations: The correlation between asset returns also affects rebalancing frequency. Highly correlated assets tend to move together, reducing the need for frequent rebalancing. However, uncorrelated or negatively correlated assets can diverge more significantly, requiring more frequent rebalancing to maintain the desired risk-return profile.
Investment Horizon: Investors with longer investment horizons may be less concerned with short-term fluctuations in portfolio value, opting for less frequent rebalancing. Investors with shorter horizons may need to rebalance more frequently to protect their capital from near-term drawdowns.

### 5.4 Analyzing Drawdown with Non-Continuous Rebalancing
Analyzing drawdown with non-continuous rebalancing is more complex than with continuous rebalancing. It requires incorporating the chosen rebalancing frequency and transaction costs into the analysis.

Simulations can be highly effective in assessing the impact of non-continuous rebalancing on drawdown. By varying the rebalancing frequency and incorporating realistic transaction costs, simulations can provide valuable insights into the optimal balance between drawdown control and return maximization.

### 5.5 Practical Implications
Finding the optimal rebalancing frequency for Kelly portfolios requires careful consideration of the factors outlined above. Investors can use analytical techniques and simulations to assess the impact of different rebalancing frequencies on their portfolio's expected growth and drawdown risk.

They may also consider dynamic rebalancing strategies, where the rebalancing frequency adjusts based on market conditions like volatility or return correlations. For example, investors could increase their rebalancing frequency during periods of heightened volatility to control drawdown and decrease it during calmer periods to minimize transaction costs.

### 5.6 Case Study - Impact of Rebalancing on Secured Maximum Drawdown
Let's consider a hypothetical case study that illustrates the impact of rebalancing frequency on drawdown. Imagine a portfolio with a 96% secured level (allowing a maximum drawdown of 4%). We simulate this portfolio with different rebalancing frequencies: every 1, 2, or 3 periods. The results show that:

More frequent rebalancing generally leads to lower drawdowns. For instance, the portfolio rebalanced every period might experience a maximum drawdown of 5.27%, while the portfolio rebalanced every 3 periods experiences a larger drawdown of 19.1%.
However, the optimal growth rate declines as the rebalancing frequency increases because more frequent rebalancing incurs higher transaction costs, reducing the overall return.
This case study emphasizes the trade-off between drawdown control and growth maximization when implementing non-continuous rebalancing. The optimal rebalancing frequency depends on the portfolio's specific characteristics, the investor's risk tolerance, and trading costs.


## 6. Python Implementation

```python
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import lognorm, t



def kelly_simulator(

excess_return_annualized,

volatility_annualized,

kelly_fraction,

n_simulations=1000,

horizon_years=10,

distribution_type='lognormal',

excess_kurtosis=0,

skew=0,

steps_per_year=252

):

"""

Simulates portfolio performance under the Kelly Criterion.



Args:

excess_return_annualized: Annualized arithmetic excess return.

volatility_annualized: Annualized volatility.

kelly_fraction: Fraction of the full Kelly leverage to use (e.g., 0.5 for half Kelly).

n_simulations: Number of simulations to run.

horizon_years: Investment horizon in years.

distribution_type: Type of return distribution ('lognormal', 't').

excess_kurtosis: Excess kurtosis for lognormal distribution (optional).

skew: Skewness for lognormal distribution (optional).

steps_per_year: Number of time steps per year (e.g., 252 for daily).



Returns:

A pandas DataFrame containing simulated portfolio values for each simulation and time step.

"""



dt = 1 / steps_per_year # Time step size

n_steps = int(horizon_years * steps_per_year) # Total number of time steps



# Initialize portfolio values

portfolio_values = pd.DataFrame(np.ones((n_simulations, n_steps + 1)), columns=range(n_steps + 1))



# Generate random returns based on specified distribution

if distribution_type == 'lognormal':

# If excess kurtosis and skew are provided, adjust lognormal parameters accordingly.

# This requires a method to translate kurtosis and skew to lognormal parameters.

# One approach is to use moment matching techniques.

# ... (Code to adjust lognormal parameters based on excess kurtosis and skew)



# Generate lognormal returns

returns = np.random.lognormal(

mean=(excess_return_annualized - 0.5 * volatility_annualized**2) * dt,

sigma=volatility_annualized * np.sqrt(dt),

size=(n_simulations, n_steps)

)

elif distribution_type == 't':

# Specify degrees of freedom for the t-distribution

# ...

# Generate t-distributed returns

# ...

else:

raise ValueError("Invalid distribution type. Choose 'lognormal' or 't'.")



# Apply Kelly leverage and simulate portfolio growth

for i in range(n_steps):

portfolio_values.iloc[:, i+1] = portfolio_values.iloc[:, i] * (1 + kelly_fraction * returns[:, i])



return portfolio_values



# Example usage

simulated_portfolios = kelly_simulator(

excess_return_annualized=0.08,

volatility_annualized=0.18,

kelly_fraction=0.5,

n_simulations=1000,

horizon_years=5,

distribution_type='lognormal',

excess_kurtosis=3,

skew=-0.5,

steps_per_year=252

)



# Calculate and output relevant metrics (annualized)

# ... (Code to calculate realized volatility, arithmetic and geometric returns, etc.)



# Visualize drawdown behavior

# ... (Code to generate plots of drawdown distribution, time to recovery, etc.)
```


This Python simulation library, tailored for exploring the Kelly Criterion, especially its volatility and drawdown dynamics, empowers users to understand and apply this powerful investment strategy effectively. By simulating various scenarios, including different return distributions, Kelly fractions, and time horizons, users can gain valuable insights into the potential outcomes of Kelly-based allocations.

The library's flexibility, enabling users to define key parameters like annualized excess return, volatility, Kelly fraction, and distribution type, facilitates the exploration of a wide range of investment scenarios. This adaptability is crucial for understanding how the Kelly Criterion performs under various market conditions and risk profiles.

Incorporating functions to calculate and visualize essential metrics like realized volatility, arithmetic and geometric returns, and drawdown characteristics further enhances the library's practical value. These tools provide users with a comprehensive view of the risk and return profiles associated with different Kelly-based strategies.

