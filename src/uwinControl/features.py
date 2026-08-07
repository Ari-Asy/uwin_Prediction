"""
features.py เอาไว้ตัวแปรให้โมเดล ML ใช้เรียนรู้
สร้าง feature และคำนวณ wind shear (α)
"""

# import libraries
import numpy as np
import pandas as pd
from .config import MIN_WIND_FOR_ALPHA, TRAIN_FRACTION, get_site

FEATURE_COLUMNS = ["WS60", "WS80", "WS100", "shear_low", "wd_sin", "wd_cos", "Temp", "Pres", "RH", "hour_sin", "hour_cos", "season_sin", "season_cos"]

# สร้างคอลัมน์ feature ใหม่จาก Raw Data ที่ผ่าน QC แล้ว
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT : df ที่ผ่าน QC แล้ว (ต้องมี WD, WS60, WS100)
    OUTPUT: df + คอลัมน์ feature 7 ตัว
    """
    out = df.copy()
    rad = np.deg2rad(out["WD"])
    out["wd_sin"], out["wd_cos"] = np.sin(rad), np.cos(rad)

    hour = out.index.hour + out.index.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

    doy = out.index.dayofyear
    out["season_sin"], out["season_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    out["shear_low"] = out["WS100"] - out["WS60"]
    return out

# คำนวณ wind shear exponent
def compute_alpha(df: pd.DataFrame, lower="WS100", upper=None, site_code=None):
    """
    คำนวณ wind shear exponent จากข้อมูล ด้วยสูตร  α = ln(v2/v1) / ln(h2/h1)
    OUTPUT: (df + คอลัมน์ alpha_observed, ค่า median ของ α)
    """
    site = get_site(site_code)
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
    feature_cols = feature_cols or FEATURE_COLUMNS
    data = df[feature_cols + [target, "alpha_observed"]].dropna()
    cut = data.index[int(len(data) * train_fraction)]
    return data.loc[:cut], data.loc[cut:]