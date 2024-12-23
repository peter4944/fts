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

# Example usage
R_a = 0.1  # 7.5% arithmetic return
vol = 0.10
skew = -0.45
v = 7
lam = 0.2
rf = 0.05

R_g_adj_t = adjusted_geometric_return_student_t(R_a, vol, v, skew)
vol_t_adj = skewed_t_vol(vol, v)
SR_t_adj = skewed_t_sharpe(R_g_adj_t, vol_t_adj, rf)
opt_vol = skewed_t_optimal_vol_target(SR_t_adj, v, skew, lam)
mtd_skewt = skewed_t_MTD(v, skew, lam)
GDR = R_g_adj_t / mtd_skewt

print("Skewed-t Adjusted Geometric Return:", R_g_adj_t)
print("Skewed-t Adjusted Volatility:", vol_t_adj)
print("Skewed-t Adjusted Sharpe:", SR_t_adj)
print("Skewed-t Optimal Vol Target:", opt_vol)
print("Skewed-t MTD:", mtd_skewt)
print("Skewed-t GDR:", GDR)

# Try changing lambda to see the expected behavior:
lam = 1.0
vol_t_adj_lam = skewed_t_adj_vol(vol, v)
SR_t_adj_lam = skewed_t_sharpe(R_g_adj_t, vol_t_adj_lam, rf)
opt_vol_lam = skewed_t_optimal_vol_target(SR_t_adj_lam, v, skew, lam)
mtd_skewt_lam = skewed_t_MTD(v, skew, lam)
GDR_lam = R_g_adj_t / mtd_skewt_lam

print("\nWith lambda=1.0:")
print("Skewed-t Adjusted Geometric Return:", R_g_adj_t)
print("Skewed-t Adjusted Volatility:", vol_t_adj_lam)
print("Skewed-t Adjusted Sharpe:", SR_t_adj_lam)
print("Skewed-t Optimal Vol Target:", opt_vol_lam)
print("Skewed-t MTD:", mtd_skewt_lam)
print("Skewed-t GDR:", GDR_lam)
