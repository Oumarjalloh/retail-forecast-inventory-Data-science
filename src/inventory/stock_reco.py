import math
from scipy.stats import norm

def recommend_order(forecast_next_L_days, on_hand: float, lead_time_days: int, service_level: float, residual_std: float):
    """
    base-stock: reorder_point = mu_L + z*sigma_L
    sigma_L approx = sqrt(L)*residual_std
    """
    L = max(1, int(lead_time_days))
    mu_L = float(sum(forecast_next_L_days[:L]))
    z = float(norm.ppf(service_level))
    sigma_L = math.sqrt(L) * float(max(0.0, residual_std))
    safety = z * sigma_L
    reorder_point = mu_L + safety
    order_qty = max(0.0, reorder_point - float(on_hand))

    # approx prob stockout (si demande ~ N(mu_L, sigma_L))
    if sigma_L > 1e-9:
        prob_stockout = float(1 - norm.cdf((on_hand - mu_L) / sigma_L))
    else:
        prob_stockout = float(1.0 if on_hand < mu_L else 0.0)

    return {
        "mu_L": mu_L,
        "sigma_L": sigma_L,
        "safety_stock": safety,
        "reorder_point": reorder_point,
        "recommended_order_qty": order_qty,
        "prob_stockout": prob_stockout
    }