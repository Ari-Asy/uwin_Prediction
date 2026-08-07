"""
data_io.py เอาไว้โหลดข้อมูลและทำ Data Versioning

ต้องทำ Data Versioning: เพราะข้อมูลไซต์เพิ่มตลอด ถ้าไม่ทำการ Freeze snapshot ไว้
จะทำให้การเทรนโมเดล แต่ละรอบจะเทียบกันไม่ได้
"""

# import libraries
import hashlib, json
from datetime import datetime
import pandas as pd
from .config import VERSION_DIR, RAW_DIR, get_site

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
    return df.rename(columns=present)

# คำนวณ hash ของข้อมูลเพื่อสร้างแต่ละเวอร์ชัน แล้วบันทึกข้อมูลเป็นไฟล์ไว้
def save_version(df: pd.DataFrame, note: str = "", site_code: str | None = None) -> str:
    """
    การ Freeze snapshot ข้อมูล และ Metadata เพื่อให้การเทรนนของโมเดลสามารถทำซ้ำกับข้อมูลเก่าได้
    INPUT: df = ข้อมูลที่ทำ Freeze ไว้ และ note = คำอธิบายเพิ่มเติม
    OUTPUT: version_id
    """

    site = get_site(site_code)
    content_hash = hashlib.md5(
        pd.util.hash_pandas_object(df.fillna(-999)).values.tobytes()).hexdigest()[:8]
    version_id = f"{site['station_code']}_{datetime.now():%Y %m %d_ %H %M}_{content_hash}"

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