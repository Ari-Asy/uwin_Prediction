"""
qc.py โมดูลที่เอาไว้ทำความสะอาดข้อมูล และจัดการ tower shadow
"""

# import libraries
import numpy as np
import pandas as pd
from .config import QC_LIMITS, get_site

# จัดการกรองค่าความเร็วลม
def clean_wind_speed(series: pd.Series, v_min = None, v_max = None, stuck_steps = None, spike_ratio = None) -> pd.Series:
    """
    กรองค่าผิดปกติออกจากคอลัมน์ wind speed
    INPUT : series = ความเร็วลมราย 10 นาที ,1 คอลัมน์
    OUTPUT: Series ค่าเดิม ค่าน่าผิดปกติเปลี่ยนเป็น NaN แทนการลบ
    """
    lim = QC_LIMITS
    v_min = lim["v_min"] if v_min is None else v_min
    v_max = lim["v_max"] if v_max is None else v_max
    stuck_steps = lim["stuck_steps"] if stuck_steps is None else stuck_steps
    spike_ratio = lim["spike_ratio"] if spike_ratio is None else spike_ratio

    # ตัดค่านอกช่วงทิ้ง
    cleaned = series.where((series >= v_min) & (series <= v_max))

    # ตัดค่าค้าง
    is_same = cleaned.diff().abs() < 1e-6
    run_length = is_same.groupby((~is_same).cumsum()).cumcount() + 1
    cleaned = cleaned.mask(run_length >= stuck_steps)

    # ตัดค่า spike เมื่อเทียบกับค่ากลาง
    rolling_median = cleaned.rolling(18, center = True, min_periods = 6).median()
    return cleaned.mask(cleaned > rolling_median * spike_ratio + 2.0)

# เป็นการเรียก clean_wind_speed ให้กับทุก sensor
def run_qc(df: pd.DataFrame, site_code = None) -> pd.DataFrame:
    """
    INPUT : df ที่เปลี่ยนชื่อคอลัมน์แล้ว
    OUTPUT: df ที่ผ่านการ QC + คอลัมน์ TI และ WS<mast> แก้เสร็จแล้ว
    """
    site = get_site(site_code)
    out = df.copy()
    for sensor in site["sensor_heights"]:
        if sensor in out.columns:
            out[sensor] = clean_wind_speed(out[sensor])

    top = f"WS{site['mast_height_m']}"
    booms = {k: v for k, v in site["boom_bearing_deg"].items() if k in out.columns}

    if "SD" in out.columns and booms:
        out["TI"] = out["SD"] / out[list(booms)[0]]

    if len(booms) == 2 and "WD" in out.columns:
        (a, bearing_a), (b, bearing_b) = booms.items()
        use_a = angular_gap(out["WD"], bearing_a) <= angular_gap(out["WD"], bearing_b)
        out[top] = pd.Series(np.where(use_a, out[a], out[b]), index = out.index)
        out[top] = out[top].fillna(out[a].fillna(out[b]))
        out.attrs["pct_using_first_boom"] = float(use_a.mean() * 100)
    elif len(booms) == 1:
        out[top] = out[list(booms)[0]]
    return out

# คำนวณผลต่างของมุม 2 ทิศ
def angular_gap(angle_a, angle_b):
    """ผลต่างเชิงมุม 0-180 องศา (จัดการการวนรอบทิศ 0/360 แล้ว)"""
    return np.abs((angle_a - angle_b + 180) % 360 - 180)

# สรุปเป็นเปอร์เซ็นความสมบูรณ์ของข้อมูลแต่ละ คอลัมน์
def coverage_report(df: pd.DataFrame, site_code=None) -> pd.DataFrame:
    """OUTPUT: DataFrame coverage % รายคอลัมน์ บอกสถานะว่าผ่านเกณฑ์ 95% หรือไม่"""
    site = get_site(site_code)
    cols = [c for c in site["sensor_heights"] if c in df.columns]
    cov = (df[cols].notna().mean() * 100).round(2)
    return pd.DataFrame({
        "coverage_pct": cov, 
        "status": np.where(cov >= 95, "ผ่านเกณฑ์", "ต่ำกว่าเกณฑ์")
        })

# บอกสาเหตุการหายไปของข้อมูล
def gap_structure(series: pd.Series, records_per_day=144) -> dict:
    """
    วิเคราะห์ว่าช่องว่างเป็นแบบกระจาย หรือแบบ logger
    OUTPUT: dict สรุปความยาวช่องว่างที่ข้อมูลหายไป
    """
    missing = series.isna()
    if not missing.any():
        return {"num_gaps": 0}
    runs = missing.groupby((~missing).cumsum()[missing]).sum()
    step_min = 24 * 60 / records_per_day
    return {
        "num_gaps": int(len(runs)),
        "longest_gap_hours": float(runs.max() * step_min / 60),
        "gaps_over_1day": int((runs > records_per_day).sum()),
        "monthly_coverage_pct": (series.notna().resample("MS").mean() * 100).round(1).to_dict(),
    }