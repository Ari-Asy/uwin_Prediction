"""
simulate.py โมดูลสำหรับการสร้างข้อมูลจำลองสำหรับทดสอบ pipeline
"""

# import libraries
import numpy as np
import pandas as pd
from .config import get_site

# สร้าง ลม รายชั่วโมง
def make_regional_wind(start, end, seed = 42) -> pd.Series:
    """OUTPUT: Series ลมภูมิภาครายชั่วโมง ร่วมทั้งข้อมูล ERA5 และข้อมูลเสาที่ไซต์"""
    random = np.random.default_rng(seed)
    time = pd.date_range(start, end, freq = "1h")
    smooth = pd.Series(random.normal(0, 1, len(time))).ewm(span = 24).mean().to_numpy()
    smooth = (smooth - smooth.mean()) / smooth.std()
    season = 1 + 0.22 * np.sin(2 * np.pi * (time.dayofyear - 30) / 365)
    year_map = pd.Series(random.normal(1, 0.05, time.year.nunique()), index = sorted(time.year.unique()))
    return pd.Series(np.clip((6.2 + 2.3 * smooth) * season * time.year.map(year_map).to_numpy(), 0.1, 30), index = time)

# สร้างข้อมูลจำลอง ERA5
def make_era5(regional: pd.Series, seed = 3) -> pd.DataFrame:
    """ข้อมูล ERA5 แบบจำลอง = ลมภูมิภาค + bias"""
    random = np.random.default_rng(seed)
    return pd.DataFrame({"era_ws": np.clip(0.82 * regional.to_numpy() + random.normal(0, 0.9, len(regional)) - 0.3, 0.1, 30)}, index = regional.index)

# สร้างข้อมูลจำลองของเสาวัดลมราย 10 นาที
def make_mast_data(regional, start, end, site_code = None, seed = 7) -> pd.DataFrame:
    """
    สร้างข้อมูลเสาวัดลมราย 10 นาที มีทั้งหมด 144 แถว/วัน พร้อม Bias ของข้อมูล:
    calibration drift, offset, เงาเสา, spike, เซนเซอร์ที่ค้าง, ข้อมูลหาย
    OUTPUT: DataFrame คอลัมน์ชื่อตาม sensor_heights + WD, SD, Temp, Pres, RH
    """
    site = get_site(site_code)
    random = np.random.default_rng(seed)
    interval_min = round(24 * 60 / site["records_per_day"]) # ความถี่มาจาก records_per_day ใน config (144 = 10min, 96 = 15min)
    time = pd.date_range(start, end, freq = f"{interval_min}min")
    num = len(time)

    reference = (regional.reindex(time.union(regional.index)).interpolate("time").reindex(time).to_numpy())
    reference = np.clip(reference + random.normal(0, 0.45, num), 0.1, 32)

    hour = time.hour + time.minute / 60
    doy = time.dayofyear
    shear_alpha = np.clip(0.20 + 0.075 * np.cos(2 * np.pi * (hour - 3) / 24) + random.normal(0, 0.02, num), 0.02, 0.5)
    stability = pd.Series(random.normal(0, 1, num)).ewm(span = 18).mean().to_numpy()
    direction = (58 + 30 * np.sin(2 * np.pi * (doy - 40) / 365) + random.normal(0, 28, num)) % 360
    turb_frac = np.clip(0.10 + 0.05 * np.exp( - reference / 8) + random.normal(0, 0.015, num), 0.03, 0.5)

    data = pd.DataFrame(index=time)
    data.index.name = "timestamp"
    for sensor, height in site["sensor_heights"].items():
        deviation = 1 + 0.012 * stability * np.log(height / 100)
        data[sensor] = np.clip(reference * (height / 100) ** shear_alpha * deviation + random.normal(0, 0.10, num), 0, None)

    # เงาเสา ~6%
    for sensor, boom in site["boom_bearing_deg"].items():
        gap = np.abs((direction - boom + 180) % 360 - 180)
        data[sensor] *= (1 - np.clip((gap-120) / 60, 0, 1) * 0.06)

    data["WS140"] *= (1 + np.linspace(0, 0.025, num)) # calibration drift
    data["WS80"] += 0.08 # offset ค้าง

    top = f"WS{site['mast_height_m']}_NW"
    data["WD"] = direction
    data["SD"] = np.clip(data[top] * turb_frac, 0.01, None)
    data["Temp"] = (27 + 4.5 * np.sin(2 * np.pi * (hour - 9) / 24) + 3 * np.sin(2 * np.pi * (doy - 100) / 365) + random.normal(0, 0.4, num))
    data["Pres"] = 1009 + 3 * np.sin(2 * np.pi * (doy - 15) / 365) + random.normal(0, 1.5, num)
    data["RH"] = np.clip(72 - 16 * np.sin(2 * np.pi * (hour - 9) / 24) + random.normal(0, 4, num), 20, 100)

    spike = random.random(num) < 0.0008
    data.loc[spike, "WS100"] *= random.uniform(1.8, 3.0, spike.sum())
    for _ in range(6):
        i = random.integers(0, num - 40)
        data.iloc[i:i+random.integers(8, 40), data.columns.get_loc("WS120")] = data["WS120"].iloc[i]

    missing = random.random(num) < 0.12
    for outage, days in [("2026-01-15", 5), ("2026-04-02", 9)]:
        i = time.get_indexer([pd.Timestamp(outage)], method = "nearest")[0]
        missing[i:i + days * site["records_per_day"]] = True
    data.loc[missing, :] = np.nan
    return data