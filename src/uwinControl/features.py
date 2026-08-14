"""
features.py เอาไว้ตัวแปรให้โมเดล ML ใช้เรียนรู้
สร้าง feature และคำนวณ wind shear (α)
"""

# import libraries
import numpy as np
import pandas as pd
from .config import MIN_WIND_FOR_ALPHA, TRAIN_FRACTION, get_site

# feature ที่ไม่ได้มาจากเซนเซอร์
DERIVED_COLUMNS = ["shear_low", "wd_sin", "wd_cos", "Temp", "Pres", "RH", "hour_sin", "hour_cos", "season_sin", "season_cos"]

# รายชื่อ feature ของไซต์
def feature_columns(site_code = None) -> list[str]:
    """
    OUTPUT: list ชื่อคอลัมน์ feature ของไซต์นั้น (เซนเซอร์มาจาก feature_sensors ใน config)
    """
    return list(get_site(site_code)["feature_sensors"]) + DERIVED_COLUMNS

# ค่าตั้งต้นของไซต์ที่ ACTIVE ตอน import (ถ้าจะสลับไซต์ ให้เรียก feature_columns() ใหม่)
FEATURE_COLUMNS = feature_columns()

# สร้างคอลัมน์ feature ใหม่จาก Raw Data ที่ผ่าน QC แล้ว
def build_features(df: pd.DataFrame, site_code = None) -> pd.DataFrame:
    """
    INPUT : df ที่ผ่าน QC แล้ว (ต้องมี WD และเซนเซอร์ตาม feature_sensors ของไซต์)
    OUTPUT: df + คอลัมน์ feature ที่คำนวณเพิ่ม
    """
    site = get_site(site_code)
    out = df.copy()
    rad = np.deg2rad(out["WD"])
    out["wd_sin"], out["wd_cos"] = np.sin(rad), np.cos(rad)

    hour = out.index.hour + out.index.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

    doy = out.index.dayofyear
    out["season_sin"], out["season_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)

    # ผลต่างลมช่วงล่าง = base_sensor - เซนเซอร์ที่ต่ำสุดใน feature_sensors
    heights = site["sensor_heights"]
    lowest = min(site["feature_sensors"], key = lambda s: heights[s])
    out["shear_low"] = out[site["base_sensor"]] - out[lowest]
    return out

# คำนวณ wind shear exponent
def compute_alpha(df: pd.DataFrame, lower = None, upper = None, site_code = None):
    """
    คำนวณ wind shear exponent จากข้อมูล ด้วยสูตร  α = ln(v2/v1) / ln(h2/h1)
    lower/upper ถ้าไม่ส่งมา จะใช้ base_sensor และเซนเซอร์ยอดเสาของไซต์
    OUTPUT: (df + คอลัมน์ alpha_observed, ค่า median ของ α)
    """
    site = get_site(site_code)
    lower = lower or site["base_sensor"]
    upper = upper or f"WS{site['mast_height_m']}"
    h_low = site["sensor_heights"][lower]
    h_up = site["mast_height_m"]

    out = df.copy()
    valid = (out[lower] > MIN_WIND_FOR_ALPHA) & (out[upper] > MIN_WIND_FOR_ALPHA)
    out["alpha_observed"] = np.nan
    out.loc[valid, "alpha_observed"] = (np.log(out.loc[valid, upper] / out.loc[valid, lower]) / np.log(h_up / h_low))
    return out, float(out["alpha_observed"].median())

# แบ่งข้อมูล
def split_by_time(df: pd.DataFrame, feature_cols = None, target = None, train_fraction = TRAIN_FRACTION, site_code = None):
    """
    แบ่งข้อมูลเป็น train/test ตามเวลา
    OUTPUT: (train, test) -> DataFrame
    """
    site = get_site(site_code)
    target = target or f"WS{site['mast_height_m']}"
    feature_cols = feature_cols or feature_columns(site_code)
    data = df[feature_cols + [target, "alpha_observed"]].dropna()
    cut = data.index[int(len(data) * train_fraction)]
    return data.loc[:cut], data.loc[cut:]