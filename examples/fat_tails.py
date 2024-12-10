import numpy as np

def student_t_volatility(vol: float, v: float) -> float:
    return vol * np.sqrt(v/(v-2))

def adjusted_geometric_return_student_t(R_a: float, vol: float, v: float, skew: float) -> float:
    # Fat-tail cost
    tail_cost = (vol**2)/2 * (2/(v-2))
    # Skew cost
    skew_cost = skew*(vol**3)/6
    return R_a - (vol**2)/2 - tail_cost - skew_cost

def skewed_t_adj_vol(vol: float, v: float) -> float:
    # Example approach: If lambda scales the skew effect on vol:
    # If we assume skew affects volatility linearly:
    return student_t_volatility(vol, v)

def skewed_t_sharpe(R_g_adj_t: float, vol_t_adj: float, rf: float=0.0) -> float:
    return (R_g_adj_t - rf) / vol_t_adj

def skewed_t_optimal_vol_target(SR_t_adj: float, v: float, skew: float, lam: float=0.2) -> float:
    # Use the theoretical formula
    # Negative skew reduces target:
    # If skew < 0, (1 + skew*lam) < 1, reduces fraction
    return SR_t_adj * ((v-2)/(v+1)) * (1 + skew*lam)

def skewed_t_MTD(v: float, skew: float, lam: float=0.2) -> float:
    # Under full Kelly conditions
    # Negative skew increases MTD:
    return 0.5 * np.sqrt(v/(v-2)) * (1 - skew*lam)

# Example usage
R_a = 0.1  # 7.5% arithmetic return
vol = 0.10
skew = -0.45
v = 7
lam = 0.2
rf = 0.05

R_g_adj_t = adjusted_geometric_return_student_t(R_a, vol, v, skew)
vol_t_adj = skewed_t_adj_vol(vol, v)
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
