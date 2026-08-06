"""
energy.py คือโมดูลสำหรับคำนวณพลังงานลม และความคุ้มทุนของโครงการ
Weibull → Power Curve → AEP → P50/P90
"""

# import libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_function

DEFAULT_TURBINE = {
    "model": "สมมติ 4.5 MW",
    "rated_kw": 4500.0,
    "cut_in_ms": 3.0,
    "rated_ms": 12.0,
    "cut_out_ms": 25.0,
    "n_turbines": 10
    }

DEFAULT_LOSSES = {
    "wake": 0.05,
    "availability": 0.03,
    "electrical": 0.02,
    "other": 0.02
    }

def fit_weibull(wind_speed) -> tuple[float, float]:
    """
    OUTPUT: (k = รูปร่างลม, A = สเกลลม m/s) 
    โดยค่า k แบบปกติของลมบกจะอยู่ที่ 1.8-3.0
    """
    v = np.asarray(wind_speed)
    v = v[np.isfinite(v) & (v > 0)]
    k, _, A = stats.weibull_min.fit(v, floc=0)
    return float(k), float(A)


def weibull_mean(k, A) -> float:
    return A * gamma_function(1 + 1/k)


def air_density(temp_c, pressure_hpa) -> float:
    """
    ρ = P/(R*T): คือความหนาแน่นของอากาศ (kg/m³)
    P = ความดัน (Pa), R = 287.05 J/(kg*K), T = อุณหภูมิ (K)
    """
    return (pressure_hpa * 100) / (287.05 * (temp_c + 273.15))


def power_curve(wind_speed, turbine = None):
    """
    INPUT: array เป็นความเร็วลม (m/s)
    OUTPUT: array เป็นกำลังไฟ (kW)
    """
    t = turbine or DEFAULT_TURBINE
    ws = np.asarray(wind_speed, dtype=float)
    power = np.zeros_like(ws)
    ramp = (ws >= t["cut_in_ms"]) & (ws < t["rated_ms"])
    power[ramp] = t["rated_kw"] * ((ws[ramp]**3 - t["cut_in_ms"]**3) / (t["rated_ms"]**3 - t["cut_in_ms"]**3))
    power[(ws >= t["rated_ms"]) & (ws <= t["cut_out_ms"])] = t["rated_kw"]
    return power


def calculate_aep(k, A, rho, turbine = None, losses = None) -> dict:
    """
    AEP = Σ [P(v) * f(v) * 8760]
    OUTPUT: dict {gross_per_turbine, net_per_turbine, net_farm, capacity_factor}
    """
    t = turbine or DEFAULT_TURBINE
    losses = losses or DEFAULT_LOSSES
    edges = np.arange(0, 30.5, 0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    freq = np.diff(stats.weibull_min.cdf(edges, k, 0, A))
    freq = freq / freq.sum()
    corrected = centers * (rho / 1.225)**(1/3) # ปรับตามค่า air density
    gross = float(np.sum(power_curve(corrected, t) * freq * 8760) / 1e6)
    net = gross

    for loss in losses.values():
        net *= (1 - loss)

    net_farm = net * t["n_turbines"]
    cf = net_farm * 1e6 / (t["rated_kw"] * t["n_turbines"] * 8760) * 100
    return {
        "gross_per_turbine_gwh": gross,
        "net_per_turbine_gwh": net,
        "net_farm_gwh": float(net_farm),
        "capacity_factor_pct": float(cf)
        }


def exceedance_levels(aep_net_gwh, uncertainty: dict) -> dict:
    """
    การรวมค่าความไม่แน่นอนแบบ root-sum-square (RSS) แล้วหา P50/P75/P90 ของ AEP
    INPUT : aep_net_gwh = AEP และ uncertainty = dict {uncertainty_name: sigma_fraction}
    OUTPUT: dict {sigma_total, P50, P75, P90}
    """
    sigma = float(np.sqrt(sum(u**2 for u in uncertainty.values())))
    return {
        "sigma_total": sigma,
        "P50": float(aep_net_gwh),
        "P75": float(aep_net_gwh * (1 - stats.norm.ppf(0.75) * sigma)),
        "P90": float(aep_net_gwh * (1 - stats.norm.ppf(0.90) * sigma))
        }