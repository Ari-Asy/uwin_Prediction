"""
config.py กำหนดค่าคงที่สำหรับ Site

เวลาเพิ่มไซต์ใหม่ก็เพิ่มตรง entry ใน SITES แล้วเปลี่ยน ACTIVE_SITE
"""
from pathlib import Path
import os

# ค่าตั้งต้นการวิเคราะห์คุณภาพข้อมูล
QC_LIMITS = {
    "v_min": 0.0, "v_max": 40.0,
    "stuck_steps": 6,
    "spike_ratio": 2.0,
    "sd_max": 5.0,

    #TODO รู้ได้ยังไงว่าค่านี้ มีหลักฐานอะไรไหมไปหามา
    "temp_min": 5.0, "temp_max":50.0,
    "pres_min": 950.0, "pres_max": 1050.0,
    "rh_min": 0.0, "rh_max": 100.0,
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
    "013054":{
        "station_code": "013054",
        "site_numeric_code": "013054",
        "layout": "long",
        "timezone": "+07:00",

        "province": "ยโสธร",
        "latitude": 15.98565,
        "longitude": 104.2271899,

        "raw_filename": "wind_raws_202608181213.csv",

        "mast_height_m": 160, #TODO คืออะไร
        "hub_height_m": 150, #TODO ค่าสมมติ
        "data_start": "2024-01-01", 
        "records_per_day": 144, #TODO คืออะไร

        "feature_sensors": ["WS60", "WS80", "WS100"], #TODO คืออะไร
        "base_sensor": "WS100", #TODO คืออะไร

        "channel_map": { #TODO คืออะไร
            1: "WS160_N",
            2: "WS160_S",
            3: "WS140",
            4: "WS120",
            5: "WS100",
            6: "WS80",
            7: "WS60",

            13: "WD152_N",
            14: "WD152_S",
            15: "WD132",
            20: "WD112",
            21: "WD92",

            16: "Temp",
            17: "Pres",
            18: "RH",
        },

        "sensor_heights": { #TODO คืออะไร
            "WS160_N": 160,
            "WS160_S": 160,
            "WS140": 140,
            "WS120": 120,
            "WS100": 100,
            "WS80": 80,
            "WS60": 60,
        },

        "boom_bearing_deg": { #TODO คืออะไร
            "WS160_N": 0.0,
            "WS160_S": 180.0,
        },

        "wind_direction_sensor": "WD152_N", #TODO คืออะไร
        "sd_sensor": "WS160_N", #TODO คืออะไร
        "era5": {
            "years": (2006, 2026),
            "margin_deg": 0.25,
            "height": 100,
        },
    },

    # เพิ่มไซต์ใหม่
}

ACTIVE_SITE = os.environ.get("UWIN_SITE", "013054")

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

    # ตรวจค่าไฟล์ที่เป็น long format
    if site.get("layout") == "long":
        for key in ("site_numeric_code", "channel_map", "wind_direction_sensor", "sd_sensor"):
            if key not in site:
                problems.append(f"layout เป็น long format แต่ไม่มี '{key}'")

        mapped = set(site.get("channel_map", {}).values())
        if site.get("wind_direction_sensor") not in mapped:
            problems.append(f"wind_direction_sensor '{site.get('wind_direction_sensor')}' ไม่มีใน channel_map")
        if site.get("sd_sensor") not in mapped:
            problems.append(f"sd_sensor '{site.get('sd_sensor')}' ไม่มีใน channel_map")

        for sensor in site["sensor_heights"]:
            if sensor not in mapped:
                problems.append(f"sensor_heights มี '{sensor}' แต่ channel_map ไม่ได้ map ไปหา")

    return problems