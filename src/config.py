"""
config.py กำหนดค่าคงที่สำหรับ Site

เวลาเพิ่มไซต์ใหม่ก็เพิ่มตรง entry ใน SITES แล้วเปลี่ยน ACTIVE_SITE
"""
from pathlib import Path
import os

# ที่อยู่ของโฟลเดอร์
# อยู่ใน Docker:/home/jovyan/data และส่วนนอก Docker:<project_root>/data
DATA_ROOT = Path(os.environ.get("UWIN_DATA_ROOT", "/home/jovyan/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parent[2] / "data"

VERSION_DIR = DATA_ROOT / "version"     # snapshot ที่เรา freeze เอาไว้
REFERENCE_DIR = DATA_ROOT / "reference" # ERA5 / MERRA-2
OUTPUT_DIR = DATA_ROOT / "outputs"      # ผลลัพธ์จากการรันโมเดล
MODEL_DIR = DATA_ROOT.parent / "models" # โมเดลที่เราเทรนแล้ว

for _dir in (VERSION_DIR, REFERENCE_DIR, OUTPUT_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ข้อมูลของไซต์งาน
SITES = {
    "GWD54": {
        "station_code": "GWD54",
        "province": "ยโสธร",
        "latitude": 15.98565,
        "longitude": 104.2271899,
        "mast_height_m": 160,
        "data_start": "2025-10-08",
        "records_per_day": 144, # ราย 10 นาที

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
    },

    # เพิ่มไซต์ใหม่
}

ACTIVE_SITE = os.environ.get("UWIN_SITE", "GWD54")

def get_site(code: str | None = None) -> dict:
    """
    INPUT : code รหัสไซต์งาน
    OUTPUT: dict ข้อมูลของไซต์งาน
    """
    code = code or ACTIVE_SITE
    if code not in SITES:
        raise ValueError(f"ไม่พบไซต์งาน '{code}' - ไซต์งานที่มีอยู่: {list(SITES)}")
    return SITES[code]

# ค่าตั้งต้นการวิเคราะห์คุณภาพข้อมูล
QC_LIMITS = {
    "v_min": 0.0, "v_max": 40.0,
    "stuck_steps": 6,
    "spike_ratio": 0.5,
}
MIN_WIND_FOR_ALPHA = 3.0 # ตัดลมอ่อนออกจากการคำนวณทิศทางลม
TRAIN_FRACTION = 0.70 # แบ่งตามเวลา