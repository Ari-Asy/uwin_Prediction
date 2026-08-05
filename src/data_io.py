"""
data_io.py เอาไว้โหลดข้อมูลและทำ Data Versioning

ต้องทำ Data Versioning: เพราะข้อมูลไซต์เพิ่มตลอด ถ้าไม่ทำการ Freeze snapshot ไว้
จะทำให้การเทรนโมเดล แต่ละรอบจะเทียบกันไม่ได้
"""

# import libraries
import hashlib, json
from datetime import datetime
import pandas as pd
from .config import VERSION_DIR, get_site

def save_version(df: pd.DataFrame, note: str = "", site_code: str | None = None) -> str:
    """
    การ Freeze snapshot ข้อมูล และ Metadata เพื่อให้ผลการเทรนนของโมเดลสามารถทำซ้ำได้
    INPUT: df = ข้อมูลที่จะทำ Freeze | note = คำอธิบายเพิ่มเติม
    OUTPUT: version_id เช่น 'GWD54_20260803_1420_a3ad7d0c'
    """

    site = get_site(site_code)
    content_hash = hashlib.md5(
        pd.util.hash_pandas_object(df.fillna(-999)).values.tobytes()).hexdigest()[:8]
    version_id = f"{site['station_code']}_{datetime.now():%Y%m%d_%H%M}_{content_hash}"

    folder = VERSION_DIR / version_id
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "data.csv.gz", compression="gzip")

    metadata = {
        "version_id": version_id,
        "content_hash": content_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "site": site["station_code"],
        "n_rows": len(df),
        "columns": list(df.columns),
        "period": [str(df.index.min()), str(df.index.max())],
        "coverage_pct": {c: round(float(df[c].notna().mean()*100), 2) for c in df.columns},
        "note": note,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return version_id


def load_version(version_id: str):
    """OUTPUT: (DataFrame, metadata dict)"""
    folder = VERSION_DIR / version_id
    df = pd.read_csv(folder / "data.csv.gz", index_col=0, parse_dates=True)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return df, meta


def list_versions(site_code: str | None = None) -> pd.DataFrame:
    """OUTPUT: DataFrame สรุปเวอร์ชันทั้งหมดของไซต์"""
    rows = []
    for folder in sorted(VERSION_DIR.iterdir(), reverse=True):
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue
        m = json.loads(meta_file.read_text(encoding="utf-8"))
        if site_code and m.get("site") != site_code:
            continue
        rows.append({"version_id": m["version_id"], "site": m.get("site"),
                     "created": m["created_at"], "rows": m["n_rows"],
                     "period_end": m["period"][1][:10], "note": m["note"]})
    return pd.DataFrame(rows)


def latest_version(site_code: str | None = None) -> str | None:
    """OUTPUT: version_id ล่าสุดของไซต์"""
    v = list_versions(site_code)
    return None if v.empty else v.iloc[0]["version_id"]