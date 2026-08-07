"""
finance.py คือโมดูลเอาไว้วิเคราะห์ความคุ้มทุน (NPV / IRR / Payback / LCOE)
เอาพลังงานที่ผลิตได้มาแปลงเป็นเงิน เพื่อดูความคุ้มค่าในการลงทุนว่าคุ้มไหม
"""

# import libraries
import numpy as np

DEFAULT_ECONOMICS = {
    "capex_musd_per_mw": 1.30,
    "opex_musd_per_mw_year": 0.045,
    "ppa_usd_per_mwh": 85.0,
    "project_life_years": 20,
    "discount_rate": 0.08,
    "annual_degradation": 0.005,
}

# เอาพลังงานต่อปีและกำลังติดตั้ง มาคำรวณเป็นเงิน 20 ปี
def financial_analysis(aep_gwh: float, capacity_mw: float, econ: dict | None = None) -> dict:
    """
    INPUT : aep_gwh = พลังงานต่อปี, capacity_mw = กำลังติดตั้ง
    OUTPUT: dict {NPV_musd, IRR_pct, payback_years, LCOE_usd_per_mwh}
    """
    e = {**DEFAULT_ECONOMICS, **(econ or {})}
    capex = e["capex_musd_per_mw"] * capacity_mw
    opex = e["opex_musd_per_mw_year"] * capacity_mw
    life, rate, degradation = e["project_life_years"], e["discount_rate"], e["annual_degradation"]

    cash_flow = [-capex]
    for year in range(1, life + 1):
        energy_mwh = aep_gwh * 1000 * (1 - degradation)**(year - 1)
        cash_flow.append(energy_mwh * e["ppa_usd_per_mwh"] / 1e6 - opex)
    cash_flow = np.array(cash_flow)

    npv = float(sum(c / (1 + rate)**i for i, c in enumerate(cash_flow)))

    roots = np.roots(cash_flow[::-1])
    roots = roots[np.isreal(roots) & (roots.real > 0)].real
    irr = float((1 / roots.max() - 1) * 100) if len(roots) else float("nan")

    cumulative = np.cumsum(cash_flow)
    payback = int(np.argmax(cumulative > 0)) if (cumulative > 0).any() else None

    disc_cost = capex + sum(opex / (1 + rate)**y for y in range(1, life + 1))
    disc_energy = sum(aep_gwh * 1000 * (1 - degradation)**(y - 1)/(1 + rate)**y for y in range(1, life + 1))
    lcoe = float(disc_cost / disc_energy * 1e6)

    return {
        "NPV_musd": npv, 
        "IRR_pct": irr,
        "break_even_years": payback, 
        "LCOE_usd_per_mwh": lcoe
        }