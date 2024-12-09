# Combining Asset Return Histories of Different Lengths
The paper "Analysing investments whose histories differ in length" [1] tackles the issue of utilising asset return histories that have varying lengths in investment analysis, particularly within a multivariate setting. This situation frequently arises when dealing with assets traded on different exchanges or in different countries, especially when incorporating investments from emerging markets [2].

Challenges of Unequal-Length Histories
The paper highlights the common practice of truncating the longer histories to match the shortest one, creating an equal-length sample for analysis [2]. While seemingly straightforward, this approach discards potentially valuable information contained within the discarded returns of the longer-history assets [3]. Those discarded returns could provide further insight into the behaviour of longer-history assets and potentially offer valuable information about the shorter-history assets [3].

Proposed Approach: Combined-Sample Analysis
The author proposes a combined-sample approach that leverages the entirety of the available data, despite the varying lengths of asset return histories [4]. This method uses all available data for each asset, leading to a more informed and comprehensive analysis.

Methodology
To illustrate the combined-sample approach and its benefits, the paper assumes a simplified framework where asset returns follow an i.i.d. multivariate Normal distribution [4]. The analysis initially focuses on a scenario with two possible starting dates for asset histories, making the central concepts easier to grasp [5]. This scenario is later generalised to accommodate an arbitrary number of starting dates [6].

Key Steps
Maximum Likelihood Estimation (MLE): The paper derives maximum-likelihood estimators (MLEs) for the first and second moments (means and variances) of asset returns using the combined sample [7]. These combined-sample MLEs are compared to the traditional truncated-sample MLEs, demonstrating that the former utilises more information and can yield more accurate estimates.
Bayesian Predictive Distribution: Acknowledging the inherent uncertainty associated with estimating parameters from finite samples, the paper introduces the Bayesian predictive distribution of future returns [8]. This distribution accounts for estimation risk, which is the additional uncertainty investors face due to the imprecision in parameter estimates [9].
Moments of the Predictive Distribution: The first and second moments of the Bayesian predictive distribution, conditional on the combined sample, are derived [8]. These moments incorporate both the sample information and the uncertainty associated with parameter estimation, offering a more complete picture of potential future returns compared to using point estimates from MLEs.
Extension to Multiple Starting Dates: The initial two-starting-date scenario is extended to handle cases with an arbitrary number of starting dates for asset histories [6]. This generalisation expands the applicability of the combined-sample method to more realistic situations.
Practical Implications: Portfolio Optimisation
The combined-sample approach is applied to two portfolio optimisation examples involving emerging markets:

Mean-Variance Optimisation with Emerging Markets: This example involves a portfolio of US Treasury bills (assumed riskless), S&P 500 index, MSCI EAFE index, and IFC Emerging Markets index [5, 10]. Using data from 1970 onwards for developed markets and 1985 onwards for emerging markets, the combined-sample approach reveals the value of incorporating the longer histories in estimating expected returns and constructing optimal portfolios. The analysis shows that a portfolio constructed using the combined sample and accounting for estimation risk significantly outperforms one built using truncated data and ignoring estimation risk [5].
Minimum-Variance Portfolio of Emerging Market Indices: This example focuses on building a minimum-variance portfolio from 22 emerging market indices with varying starting dates between 1989 and 1993 [11]. The paper demonstrates that using the combined sample and accounting for estimation risk leads to a significantly lower portfolio variance compared to methods that either truncate the sample or ignore estimation risk [12].
Frequentist Analysis: Performance in Repeated Samples
In addition to the conditional Bayesian decision approach, the paper examines the performance of various portfolio construction methods in repeated hypothetical random samples [13]. This frequentist analysis, using 'true' moments of returns, further confirms the substantial benefits of utilising the combined sample and accounting for estimation risk [13].

Key Takeaways
The paper advocates for a paradigm shift in handling asset return histories of different lengths:

Information Retention: Don't discard valuable information present in longer histories.
Combined-Sample Analysis: Embrace the combined-sample approach for more comprehensive analysis.
Estimation Risk: Always account for estimation risk when dealing with finite samples.
These takeaways highlight the importance of utilising all available information, even when dealing with datasets of varying lengths. The combined-sample approach coupled with the recognition of estimation risk leads to more informed and robust portfolio construction, ultimately resulting in more efficient investment decisions.

Based on the search results, I'll outline Stambaugh's methodology for handling assets with different historical lengths in portfolio construction.

## Core Methodology

**Basic Framework**
The approach uses a multivariate method that leverages both long and short history assets to improve estimation of returns and covariances[4]. Rather than truncating all histories to match the shortest one, it extracts information from longer histories to benefit the analysis of shorter-history assets.

## Implementation Steps

1. **Data Organization**
- Separate assets into groups based on history length (X for long history, Y for short history)[2]
- Ensure all assets share a common ending date but can have different starting dates

2. **Parameter Estimation**
For assets with shorter histories (Y), estimate:

**Mean Returns**
$$\hat{\mu}_{Y,L} = \mu_{Y,S} + \beta(\mu_{X,L} - \mu_{X,S})$$[2]

**Covariance Matrix**
$$\hat{\Sigma}_{YY,L} = \Sigma_{YY,S} + \beta(\Sigma_{XX,L} - \Sigma_{XX,S})\beta^t$$[2]

3. **Multiple Starting Dates Handling**
When assets have various starting dates:
- Group assets by history length (G1, G2, ..., GJ)
- Apply the method iteratively, starting with longest histories[2]
- Progressively incorporate backfilled data into subsequent estimations

## Practical Considerations

**Implementation Challenges**
- Covariance matrices may not be invertible for short history periods[2]
- Need to ensure positive semi-definiteness of estimated covariance matrices
- Account for estimation risk through Bayesian predictive distributions[4]

## Portfolio Construction Application

The methodology integrates with standard portfolio optimization frameworks:
- Use estimated parameters in a Markowitz-style optimization
- Account for estimation uncertainty in the portfolio weights
- Consider trading costs and rebalancing frequency[1]

This approach provides more accurate estimates of portfolio parameters compared to simply truncating histories or using maximum-likelihood estimates alone[4].

Citations:
[1] https://stanford.edu/~boyd/papers/pdf/lrsm_portfolio.pdf
[2] https://portfoliooptimizer.io/blog/managing-missing-asset-returns-in-portfolio-analysis-and-optimization-backfilling-through-residuals-recycling/
[3] https://fnce.wharton.upenn.edu/profile/stambaug/
[4] https://repository.upenn.edu/items/18ea3b91-83de-49a1-98cc-2f65ab827a7c
[5] https://ideas.repec.org/p/nbr/nberwo/5918.html