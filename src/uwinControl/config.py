"""
config.py กำหนดค่าคงที่สำหรับ Site

เวลาเพิ่มไซต์ใหม่ก็เพิ่มตรง entry ใน SITES แล้วเปลี่ยน ACTIVE_SITE
"""
from pathlib import Path
import os

# ค่าตั้งต้นการวิเคราะห์คุณภาพข้อมูล
QC_LIMITS = {
    "v_min": 0.0,
    "v_max": 40.0,
    "stuck_steps": 6,
    "spike_ratio": 2.0,
}
MIN_WIND_FOR_ALPHA = 3.0 # ตัดลมอ่อนออกจากการคำนวณทิศทางลม
TRAIN_FRACTION = 0.70 # แบ่งตามเวลา

# ที่อยู่ของโฟลเดอร์ อยู่ใน Docker และส่วนนอก Docker:<project_root>/data
DATA_ROOT = Path(os.environ.get("UWIN_DATA_ROOT", "/home/jovyan/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

RAW_DIR = DATA_ROOT / "raw"             # ไฟล์ CSV จากไซต์
VERSION_DIR = DATA_ROOT / "version"     # snapshot ที่เรา freeze เอาไว้
REFERENCE_DIR = DATA_ROOT / "reference" # ERA5 / MERRA-2
OUTPUT_DIR = DATA_ROOT / "outputs"      # ผลลัพธ์จากการรันโมเดล
MODEL_DIR = DATA_ROOT.parent / "models" # โมเดลที่เทรนแล้ว

for _dir in (RAW_DIR, VERSION_DIR, REFERENCE_DIR, OUTPUT_DIR, MODEL_DIR):
    _dir.mkdir(parents = True, exist_ok = True)

# ข้อมูลของไซต์งาน
SITES = {
    "GWD_10_160": {
        "station_code": "GWD_10_160",
        "province": "county_name",
        "latitude": 15.98565,
        "longitude": 104.2271899,
        "mast_height_m": 160, # เซนเซอร์ตัวบนสุด
        "hub_height_m": 150,  # TODO: ค่าสมมติ ของรุ่นกังหัน
        "data_start": "2025-10-08",
        "records_per_day": 144, # ราย 10 นาที (144 = 24*60/10)

        # เซนเซอร์ที่ใช้เป็น input ของโมเดล (ตั้งใจใช้แค่ตัวล่าง ให้โมเดลต้องขยายขึ้นไปหายอดเสาเอง)
        "feature_sensors": ["WS60", "WS80", "WS100"],
        # ฐานอ้างอิงของ Power Law และ shear_low ต้องอยู่ใน feature_sensors
        "base_sensor": "WS100",

        "column_map": {
            "Ch1_Anem_160m_NW_m/s": "WS160_NW",
            "Ch2_Anem_160m_SE_m/s": "WS160_SE",
            "Ch3_Anem_140m_NW_m/s": "WS140",
            "Ch4_Anem_120m_NW_m/s": "WS120",
            "Ch5_Anem_100m_NW_m/s": "WS100",
            "Ch6_Anem_80m_NW_m/s":  "WS80",
            "Ch7_Anem_60m_NW_m/s":  "WS60",

            "ChX_Vane_Deg": "WD",
            "ChX_SD_m/s": "SD",
            "Temperature_C": "Temp",
            "Pressure_hPa": "Pres",
            "Humidity_pct": "RH",
        },

        "sensor_heights": {
            "WS160_NW": 160,
            "WS160_SE": 160,
            "WS140": 140,
            "WS120": 120,
            "WS100": 100,
            "WS80": 80,
            "WS60": 60,
        },

        "boom_bearing_deg": {
            "WS160_NW": 315.0,
            "WS160_SE": 135.0,
        },

        "era5": {
            "years": (2006, 2026), # ปีแรกถึงปีสุดท้าย
            "margin_deg": 0.25, # ระยะขอบรอบพิกัดไซต์ หน่วยองศา
            "height": 100, # ระดับความสูงที่ใช้ทำ MCP
        },
    },

    "GWD_15_125": {
        "station_code": "GWD_15_125",
        "province": "county_name",
        "latitude": 15.98565, # TODO: ก๊อปมาจาก GWD_10_160
        "longitude": 104.2271899,
        "mast_height_m": 125, # เซนเซอร์ตัวบนสุด
        "hub_height_m": 120, # TODO: ค่าสมมติ ตั้งไว้ไม่เกินยอดเสา จะได้เป็น interpolation
        "data_start": "2025-10-08",
        "records_per_day": 96, # ราย 15 นาที (96 = 24*60/15)

        "feature_sensors": ["WS60", "WS80", "WS100"],
        "base_sensor": "WS100",

        "column_map": {
            "Ch1_Anem_125m_NW_m/s": "WS125",
            "Ch2_Anem_100m_NW_m/s": "WS100",
            "Ch3_Anem_80m_NW_m/s": "WS80",
            "Ch4_Anem_60m_NW_m/s": "WS60",

            "ChX_Vane_Deg": "WD",
            "ChX_SD_m/s": "SD",
            "Temperature_C": "Temp",
            "Pressure_hPa": "Pres",
            "Humidity_pct": "RH",
        },

        "sensor_heights": {
            "WS125": 125,
            "WS100": 100,
            "WS80": 80,
            "WS60": 60,
        },

        "boom_bearing_deg": {
            "WS125": 315.0
        },

        "era5": {
            "years": (2006, 2026),
            "margin_deg": 0.25,
            "height": 100,
        },
    }, 

    # เพิ่มไซต์ใหม่
}

ACTIVE_SITE = os.environ.get("UWIN_SITE", "GWD_10_160")

def get_site(code: str | None = None) -> dict:
    """
    INPUT : code รหัสไซต์งาน
    OUTPUT: dict ข้อมูลของไซต์งาน
    """
    code = code or ACTIVE_SITE
    if code not in SITES:
        raise ValueError(f"ไม่พบไซต์งาน '{code}' - ไซต์งานที่มีอยู่: {list(SITES)}")
    return SITES[code]

# หาเซนเซอร์ตัวบนสุดของเสา
def top_sensor(code: str | None = None) -> str:
    """
    OUTPUT: ชื่อเซนเซอร์ที่ยอดเสา
    """
    site = get_site(code)
    mast = site["mast_height_m"]
    at_top = [s for s, h in site["sensor_heights"].items() if h == mast]
    if not at_top:
        raise ValueError(f"ไซต์ {site['station_code']} ไม่มีเซนเซอร์ที่ความสูงยอดเสา {mast} ม.")
    with_boom = [s for s in at_top if s in site["boom_bearing_deg"]]
    return (with_boom or at_top)[0]

# เรียงเซนเซอร์จากสูงลงต่ำ ระดับความสูงละ 1 ตัว (ระดับที่มี 2 ตัวเช่น NW/SE จะเอาตัวแรก)
def sensors_top_down(code: str | None = None) -> list[str]:
    """OUTPUT: list ชื่อเซนเซอร์เรียงจากสูงสุดไปต่ำสุด"""
    site = get_site(code)
    first_at_height = {}
    for sensor, height in site["sensor_heights"].items():
        first_at_height.setdefault(height, sensor)
    return [first_at_height[h] for h in sorted(first_at_height, reverse = True)]

def era5_area(code: str | None = None) -> list[float]:
    """
    OUTPUT: [North, West, South, East] กล่องพื้นที่ใช้สำหรับ EAR5
    """
    site = get_site(code)
    margin = site["era5"]["margin_deg"]
    lat, lon = site["latitude"], site["longitude"]
    return [lat + margin, lon - margin, lat - margin, lon + margin]

# ตรวจ entry ของไซต์ เอาไว้เช็คข้อมูลในไซต์
def validate_site(code: str | None = None) -> list[str]:
    """
    ตรวจว่า entry ของไซต์ในตัวแปร SITES ว่าถูกต้องไหม
    OUTPUT: list ของปัญหาที่เจอ ถ้าว่าง = ผ่าน
    """
    site = get_site(code)
    heights = site["sensor_heights"]
    problems = []

    # ตรวจว่ามีเซนเซอร์อยู่ที่ความสูงยอดเสาไหม
    if not any(h == site["mast_height_m"] for h in heights.values()):
        problems.append(f"ไม่มีเซนเซอร์ตัวไหนอยู่ที่ความสูงยอดเสา {site['mast_height_m']} ม.")

    # ตรวจ feature_sensors
    for sensor in site["feature_sensors"]:
        if sensor not in heights:
            problems.append(f"feature_sensors มี '{sensor}' แต่ไม่มีใน sensor_heights")

    # ตรวจ base_sensor
    if site["base_sensor"] not in site["feature_sensors"]:
        problems.append(f"base_sensor '{site['base_sensor']}' ต้องอยู่ใน feature_sensors ด้วย")

    # ตรวจ boom_bearing_deg
    for sensor in site["boom_bearing_deg"]:
        if sensor not in heights:
            problems.append(f"boom_bearing_deg มี '{sensor}' แต่ไม่มีใน sensor_heights")

    # ตรวจจำนวนข้อมูลต่อวัน
    minutes_per_day = 24 * 60
    if minutes_per_day % site["records_per_day"] != 0:
        problems.append(f"records_per_day = {site['records_per_day']} หารกับ 1440 นาทีไม่ลงตัว")

    # ตรวจว่า sensor ทุกตัวถูก map ใน column_map ไหม
    mapped = set(site["column_map"].values())
    for sensor in heights:
        if sensor not in mapped:
            problems.append(f"sensor_heights มี '{sensor}' แต่ column_map ไม่ได้ map ไปหาชื่อนี้")

    # ตรวจหา ERA5
    era5 = site.get("era5")
    if era5 is None:
        problems.append("ค่า era5 ไม่มีใน entry ของ SITE")
    else:
        first_year, last_year = era5["years"]
        if first_year > last_year:
            problems.append(f"era5.years ผิดลำดับ: {first_year} > {last_year}")

        # ตรวจ overlap ของข้อมูลเสา กัน MCP เพี้ยน
        start_year = int(site["data_start"][:4])
        if last_year < start_year + 1:
            problems.append(f"era5.years ถึงแค่ช่วง {last_year} แต่ข้อมูลเสาเริ่ม {start_year} "
                            f"ต้องโหลด ERA5 ถึงอย่างน้อย {start_year + 1}")

        # ตรวจ height ของเสากับ ERA5
        if era5["height"] not in (10,100):
            problems.append(f"era5.height ต้องเป็น 10 หรือ 100 ไม่ใช่ {era5['height']}")

        # ตรวจ Margin หรือระยะขอบของพิกัดไซต์
        if not 0 < era5["margin_deg"] <= 2:
            problems.append(f"era5.margin_deg = {era5['margin_deg']} ผิดปกติ")

    return problems