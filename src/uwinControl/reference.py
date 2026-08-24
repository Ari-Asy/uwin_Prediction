"""
reference.py โมดูลโหลดข้อมูล reanalysis ERA5 จาก Copernicus CDS มาใช้กับ MCP
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .config import REFERENCE_DIR, get_site, era5_area

# dataset ที่ใช้ {ERA5 hourly data on single levels from 1940 to present}
DATASET = "reanalysis-era5-single-levels"
DEFAULT_TIMEZONE = "Asia/Bangkok"

# สร้าง Request ยิง API ไปให้ CDS
def build_request(year: int, month: int, site_code: str | None = None) -> dict:
    """
    OUTPUT: dict ที่ส่งให้ client.retrieve() ได้เลยครั้งละ 1 เดือน
    """
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": era5_area(site_code),
    }

# ตรวจข้อมูลภายใน ERA5
# ตรวจระยะพิกัด
# TODO สูตรคืออะไร แล้วมาได้ยังไง? เอามาใช้วัดอะไร
def _distance_km(lati1, longi1, lati2, longi2) -> float:
    """ตรวจระยะทางระหว่างสองพิกัดแบบ haversine (กม.)"""
    radius = 6371.0
    distance_lati = np.radians(lati2 - lati1)
    distance_longi = np.radians(longi2 - longi1)
    a = (np.sin(distance_lati / 2)**2 + np.cos(np.radians(lati1)) * np.cos(np.radians(lati2)) * np.sin(distance_longi / 2)**2)
    return float(2 * radius * np.arcsin(np.sqrt(a)))

# ที่อยู่โฟลเดอร์ ERA5
# TODO ดูว่าทำไมถึงเขียนแบบนี้ มีความหมายอะไร
def era5_folder(site_code: str | None = None) -> Path:
    """OUTPUT: เป็น Folder ที่เก็บ ERA5 ของไซต์นั้นๆ"""
    site = get_site(site_code)
    folder = REFERENCE_DIR / site["station_code"]
    folder.mkdir(parents = True, exist_ok = True)
    return folder

# ที่อยู่ไฟล์ ERA5
# TODO ดูว่าทำไมถึงเขียนแบบนี้ มีความหมายอะไร
def era5_path(year: int, month: int, site_code: str | None = None) -> Path:
    """OUTPUT: Path ที่เก็บข้อมูล ERA5 ของไซต์นั้นๆ"""
    return era5_folder(site_code) / f"era5_{year}_{month:02d}.nc"

# โหลดข้อมูล ERA5
# TODO ศึกษามาว่าทำไมต้องเขียนฟังก์ชันแบบนี้
def download_era5(year: int, month: int, site_code: str | None = None, force: bool = False) -> Path:
    """
    ดาวน์โหลด ERA5 1 ปี ถ้ามีไฟล์อยู่ใน Path แล้วจะข้าม
    OUTPUT: Path ของไฟล์
    """
    path = era5_path(year, month, site_code)
    if path.exists() and not force:
        print(f"ข้าม {year}-{month:02d} (มีไฟล์แล้ว {path.stat().st_size / 1e6:.1f} MB)")
        return path

    # import cdsapi ตรงนี้ เพื่อให้อ่านไฟล์ที่โหลดไว้แล้ว โดยไม่ต้องมี credential
    import cdsapi
    print (f"Start {year}-{month:02d}")
    client = cdsapi.Client()
    client.retrieve(DATASET, build_request(year, month, site_code)).download(str(path))
    print(f"Finish {year}-{month:02d} : {path.stat().st_size / 1e6:.1f} MB")
    return path

# โหลดข้อมูล ERA5 เป็นช่วง
# TODO ศึกษามาว่าทำไมต้องเขียนฟังก์ชันแบบนี้
def download_era5_range(site_code: str | None = None, force: bool = False) -> list[Path]:
    """
    วนโหลดข้อมูลตามช่วงใน config
    OUTPUT: list ของไฟล์ที่โหลดสำเร็จ
    """
    site = get_site(site_code)
    first_year, last_year = site["era5"]["years"]
    done = []
    failed = []

    for year in range(first_year, last_year + 1):
        for month in range(1 ,13):
            try:
                done.append(download_era5(year, month, site_code, force))
            except Exception as error:
                print(f"{year}-{month:02d} Fail : {error}")
                failed.append((year, month))

    print(f"\nข้อมูลที่โหลดสำเร็จ: {len(done)}")
    print(f"ข้อมูลที่โหลดไม่สำเร็จ: {len(failed)}")
    if failed:
        print(f"ข้อมูลที่ต้องมีการโหลดซ้ำ: {failed}")
    return done

# อ่านข้อมูลเพื่อเอามาใช้งาน
# TODO ศึกษามาว่าทำไมต้องเขียนฟังก์ชันแบบนี้
def load_era5(site_code: str | None = None, timezone: str | None = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """
    ฟังก์ชันอ่านไฟล์ ERA5 ทุกปีของไซต์ โดยจะเลือกกริดที่ใกล้ที่สุด แล้วคืนค่าเป็นตารางรายชั่วโมง
    INPUT: timezone = คือเขตเวลาปลายทาง
    OUTPUI: DataFrame index รายชั่วโมง โดยจะมีคอลัมน์คือ era_ws, era_wd, era_temp, era_pres
    """
    import xarray as xr
    site = get_site(site_code)
    files = sorted(era5_folder(site_code).glob("era5_*.nc"))
    if not files: # กรณีไม่พบไฟล์
        raise FileNotFoundError(f"ไม่พบไฟล์ ERA5 ของไซต์ {site['station_code']} ใน {REFERENCE_DIR}")
    dataset = xr.open_mfdataset(files, combine = "by_coords")

    # เลือกจุดที่ใกล้ไซต์ที่สุด
    point = dataset.sel(latitude = site["latitude"], longitude = site["longitude"], method = "nearest")
    grid_latitude = float(point["latitude"])
    grid_longitude = float(point["longitude"])
    distance = _distance_km(site["latitude"], site["longitude"], grid_latitude, grid_longitude)
    print(f"พิกัดที่ใช้ในการกริด: {grid_latitude:.3f}, {grid_longitude:.3f} (พิกัดห่างจากไซต์ {distance:.1f} กม.)")
    if distance > 20:
        print("จุดกริดห่างเกิน 20 กม.")

    height = site["era5"]["height"]
    u_value = point[f"u{height}"].to_series()
    v_value = point[f"v{height}"].to_series()
    temp = point["t2m"].to_series()
    pressure = point["sp"].to_series()

    output = pd.DataFrame({ # TODO ไปหาข้อมูลมาว่าทำไม่ต้องเขียนแบบนี้สูตรพวกนี้มีผลยังไง
        "era_ws": np.sqrt(u_value**2 + v_value**2),
        "era_wd": (270 - np.degrees(np.arctan2(v_value, u_value))) % 360,
        "era_temp": temp - 273.15,
        "era_pres": pressure / 100,
    })

    output.index = pd.to_datetime(output.index)
    output = output.sort_index()
    output = output[~output.index.duplicated(keep = "first")]

    # ERA5 เป็นเวลา UTC ต้องแปลงเวลา แล้วถอด timezone ออก เพราะถ้าเป็นกันจะ join กันไม่ได้
    if timezone:
        output.index = output.index.tz_localize("UTC").tz_convert(timezone).tz_localize(None)
    output.index.name = "timestamp"

    print(f"ช่วงเวลา: {output.index.min()} ถึงช่วง {output.index.max()} ({'เวลา ' + timezone if timezone else 'UTC'})")
    print(f"จำนวนแถว: {len(output):,}")
    print(f"ลมเฉลี่ยที่ {height} ม.: {output['era_ws'].mean():.2f} m/s")
    print("")
    return output

# ฟังก์ชันแสดงผลข้อมูล เอาไว้ใส่ run_info.json
def era5_info(era5: pd.DataFrame, site_code: str | None = None, timezone: str | None = DEFAULT_TIMEZONE) -> dict:
    """
    OUTPUT: dict โชว์สรุปว่ารอบนี้ใช้ ERA5 ชุดไหนไป
    """
    site = get_site(site_code)
    return {
        "source": "ERA5 (Copernicus CDS)",
        "dataset": DATASET,
        "height_m": site["era5"]["height"],
        "years": list(site["era5"]["years"]),
        "area": era5_area(site_code),
        "timezone": timezone or "UTC",
        "num_rows": len(era5),
        "period": [str(era5.index.min()), str(era5.index.max())],
        "mean_ws": round(float(era5["era_ws"].mean()), 3),
    }