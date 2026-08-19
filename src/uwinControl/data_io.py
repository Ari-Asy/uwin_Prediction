"""
data_io.py เอาไว้โหลดข้อมูลและทำ Data Versioning

ต้องทำ Data Versioning: เพราะข้อมูลไซต์เพิ่มตลอด ถ้าไม่ทำการ Freeze snapshot ไว้
จะทำให้การเทรนโมเดล แต่ละรอบจะเทียบกันไม่ได้
"""

# import libraries
import hashlib, json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .config import VERSION_DIR, RAW_DIR, OUTPUT_DIR, get_site

# เอาไว้อ่านไฟล์ CSV จากโฟลเดอร์ data/raw/
def load_raw(filename: str, site_code: str | None = None, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    INPUT : filename = ชื่อไฟล์ .CSV ใน data/raw/ และค่า site_code
    OUTPUT: DataFrame index = timestamp, ชื่อคอลัมน์ที่สั้น
    """
    site = get_site(site_code)
    df = pd.read_csv(RAW_DIR / filename)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    # เปลี่ยนชื่อเฉพาะคอลัมน์ที่มีในไฟล์
    present = {k: v for k, v in site["column_map"].items() if k in df.columns}
    missing = [k for k in site["column_map"] if k not in df.columns]
    if missing:
        print(f"ไม่พบคอลัมน์ในไฟล์: {missing}")
    return df.rename(columns = present)

# คำนวณ hash ของข้อมูลเพื่อสร้างแต่ละเวอร์ชัน แล้วบันทึกข้อมูลเป็นไฟล์ไว้
def save_version(df: pd.DataFrame, note: str = "", site_code: str | None = None) -> str:
    """
    การ Freeze snapshot ข้อมูล และ Metadata เพื่อให้การเทรนนของโมเดลสามารถทำซ้ำกับข้อมูลเก่าได้
    INPUT: df = ข้อมูลที่ทำ Freeze ไว้ และ note = คำอธิบายเพิ่มเติม
    OUTPUT: version_id
    """
    site = get_site(site_code)
    content_hash = hashlib.md5(pd.util.hash_pandas_object(df.fillna(-999)).values.tobytes()).hexdigest()[:8]
    version_id = f"{site['station_code']}_{datetime.now():%d%m%Y_%H%M}_{content_hash}"

    folder = VERSION_DIR / version_id
    folder.mkdir(parents = True, exist_ok = True)
    df.to_csv(folder / "data.csv.gz", compression="gzip")

    metadata = {
        "version_id": version_id,
        "content_hash": content_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "site": site["station_code"],
        "num_rows": len(df),
        "columns": list(df.columns),
        "period": [str(df.index.min()), str(df.index.max())],
        "coverage_pct": {c: round(float(df[c].notna().mean() * 100), 2) for c in df.columns},
        "note": note,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent = 2, ensure_ascii = False), encoding = "utf-8")
    return version_id

# โหลดข้อมูลเวอร์ชันที่เคย freeze ไว้กลับมา
def load_version(version_id: str):
    """OUTPUT: (DataFrame, metadata dict)"""
    folder = VERSION_DIR / version_id
    df = pd.read_csv(folder / "data.csv.gz", index_col = 0, parse_dates = True)
    meta = json.loads((folder / "metadata.json").read_text(encoding = "utf-8"))
    return df, meta

# เอาไว้ไล่ดูเวอร์ชันของข้อมูลที่มี
def list_versions(site_code: str | None = None) -> pd.DataFrame:
    """OUTPUT: DataFrame สรุปเวอร์ชันทั้งหมดของไซต์"""
    rows = []
    for folder in sorted(VERSION_DIR.iterdir(), reverse = True):
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue
        m = json.loads(meta_file.read_text(encoding="utf-8"))
        if site_code and m.get("site") != site_code:
            continue
        rows.append({
            "version_id": m["version_id"], 
            "site": m.get("site"),
            "created": m["created_at"], 
            "rows": m.get("num_rows"),
            "period_end": m["period"][1][:10], 
            "note": m["note"]
        })
    return pd.DataFrame(rows)

# คืนตัว version_id ล่าสุดของไซต์
def latest_version(site_code: str | None = None) -> str | None:
    """OUTPUT: version_id ล่าสุดของไซต์"""
    value = list_versions(site_code)
    return None if value.empty else value.iloc[0]["version_id"]


# บันทึกโฟลเดอร์ผลลัพธ์แยกตามรอบการรัน (1 รอบ = 1 โฟลเดอร์ใน data/outputs/)
_CURRENT_RUN: Path | None = None # เรียกใช้เฉพาะในไฟล์

# เปิดรอบการรันใหม่ สร้างโฟลเดอร์ผลลัพธ์
def start_run(note: str = "", site_code: str | None = None, data_version: str | None = None, run_id: str | None = None, era5_info: dict | None = None) -> str:
    """
    สร้างโฟลเดอร์ data/outputs/<run_id>/ แล้วให้ save_output ทุกครั้งหลังจากนี้เขียนลงในนั้นเสร็จ
    INPUT : note = คำอธิบาย, data_version = version_id ที่ใช้เทรนข้อมูล
    OUTPUT: run_id
    """
    global _CURRENT_RUN
    site = get_site(site_code)
    run_id = run_id or f"{site['station_code']}_{datetime.now():%Y%m%d_%H%M%S}"

    folder = OUTPUT_DIR / run_id
    folder.mkdir(parents = True, exist_ok = True)
    _CURRENT_RUN = folder

    # อธิบายว่าโฟลเดอร์ รันจากรอบไหน
    info = {
        "run_id": run_id,
        "site": site["station_code"],
        "mast_height_m": site["mast_height_m"],
        "records_per_day": site["records_per_day"],
        "data_version": data_version,
        "era5": era5_info,
        "created_at": datetime.now().isoformat(timespec = "seconds"),
        "note": note,
        "files": [],
    }
    (folder / "run_info.json").write_text(json.dumps(info, indent = 2, ensure_ascii = False), encoding = "utf-8")
    print(f"Run file: {folder}")
    return run_id

# คืนโฟลเดอร์ของรอบปัจจุบัน ถ้ายังไม่เปิดจะเปิดให้อัตโนมัติ
def run_dir(site_code: str | None = None) -> Path:
    """OUTPUT: Path ของโฟลเดอร์รอบปัจจุบัน"""
    if _CURRENT_RUN is None:
        start_run(note = "เปิดอัตโนมัติ (ไม่ได้เรียก start_run เอง)", site_code = site_code)
    return _CURRENT_RUN

# เพิ่มชื่อไฟล์ที่บันทึกแล้วลงใน run_info.json
def _record_file(folder: Path, filename: str) -> None:
    info_file = folder / "run_info.json"
    if not info_file.exists():
        return
    info = json.loads(info_file.read_text(encoding = "utf-8"))
    if filename not in info.get("files", []):
        info.setdefault("files", []).append(filename)
    info_file.write_text(json.dumps(info, indent = 2, ensure_ascii = False), encoding = "utf-8")

# บันทึกผลลัพธ์ให้เป็นไฟล์ .csv ลงในโฟลเดอร์ปัจจุบัน
def save_output(data, name: str, add_timestamp: bool = False, site_code: str | None = None) -> str:
    """
    INPUT : data = DataFrame หรือ Series, name = ชื่อไฟล์, add_timestamp = ใส่วันเวลาต่อท้ายชื่อไฟล์
    OUTPUT: path ของไฟล์ที่บันทึกไว้
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    folder = run_dir(site_code)
    timestamp = f"_{datetime.now():%H%M%S}" if add_timestamp else ""
    filename = f"{name}{timestamp}.csv"

    data.to_csv(folder / filename, encoding = "utf-8-sig")
    _record_file(folder, filename)
    print(f"Save: {folder.name}/{filename}")
    return str(folder / filename)

# บันทึกกราฟลงโฟลเดอร์รอบเดียวกับ .csv
def save_image(image, name: str, dpi: int = 130, site_code: str | None = None) -> str:
    """
    INPUT : image = matplotlib image, name = ชื่อไฟล์
    OUTPUT: path ของไฟล์ที่บันทึกไว้
    """
    folder = run_dir(site_code)
    filename = f"{name}.png" # ชื่อไฟล์ที่รับมา
    image.savefig(folder / filename, # ตำแหน่งไฟล์
                  dpi = dpi, # กำหนดความละเอียด
                  bbox_inches = "tight" # ตัดขอบว่างรอบรูป
                  )
    _record_file(folder, filename)
    print(f"Save: {folder.name}/{filename}")
    return str(folder / filename)

# ไล่ดูรอบการรันทั้งหมดที่เคยทำไว้
def list_runs(site_code: str | None = None) -> pd.DataFrame:
    """OUTPUT: DataFrame สรุปรอบการรันทั้งหมด เรียงจากใหม่ไปเก่า"""
    rows = []
    for folder in sorted(OUTPUT_DIR.iterdir(), reverse = True):
        info_file = folder / "run_info.json"
        if not folder.is_dir() or not info_file.exists():
            continue
        info = json.loads(info_file.read_text(encoding = "utf-8"))
        if site_code and info.get("site") != site_code:
            continue
        rows.append({
            "run_id": info["run_id"],
            "site": info.get("site"),
            "created": info.get("created_at"),
            "data_version": info.get("data_version"),
            "num_files": len(info.get("files", [])),
            "note": info.get("note"),
        })
    return pd.DataFrame(rows)