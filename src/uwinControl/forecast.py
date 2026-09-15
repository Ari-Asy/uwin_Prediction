"""
forecast.py โมดูลเอาไว้ทำนายลมล่วงหน้า

ค่าลมที่ทำนาย = บล็อกข้อมูลวัดจริง 7 วัน * (รูปร่าง Fourier ที่เวลาปลายทาง / รูปร่างที่เวลาต้นทาง) * ความผันผวนรายปี
ผลลัพธ์ถูกต้องเชิงสถิติ (ค่าเฉลี่ย ค่าแกว่ง การแจกแจง พลังงานรวม) แต่ไม่ถูกต้องราย timestamp
"""

# import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .config import FORECAST, RAW_DIR, get_site
from .energy import DEFAULT_LOSSES, DEFAULT_TURBINE, power_curve
from .models import evaluate

# สร้าง feature เวลาสำหรับ Fourier
def _fourier(index, n_year, n_day):
    """
    INPUT: DatetimeIndex
    OUTPUT: DataFrame sin/cos เวลารอบปีและรอบวัน
    """
    day_of_year = index.dayofyear + index.hour / 24
    hour_of_day = index.hour + index.minute / 60
    output = {}
    for hour in range(1, n_year + 1):
        output[f"y{hour}_sin"] = np.sin(2 * np.pi * hour * day_of_year / 365.25)
        output[f"y{hour}_cos"] = np.cos(2 * np.pi * hour * day_of_year / 365.25)
    for hour in range(1, n_day + 1):
        output[f"d{hour}_sin"] = np.sin(2 * np.pi * hour * hour_of_day / 24)
        output[f"d{hour}_cos"] = np.cos(2 * np.pi * hour * hour_of_day / 24)
    return pd.DataFrame(output, index = index)

# ความผันผวนรายปีจาก ERA5
def annual_iav(era5: pd.DataFrame, min_hours: int = 8000) -> float:
    """
    IAV = sd / mean ของลมเฉลี่ยรายปี โดยจะคิดจาก Raw ERA5 เท่านั้น
    INPUT: era5 = DataFrame จาก reference.load_era5(), min_hours = ปีที่มีข้อมูลน้อยกว่านี้ถูกตัดทิ้ง
    OUTPUT: IAV เป็นสัดส่วน เช่น 0.04
    """
    yearly = era5["era_ws"].resample("YS")
    annual = yearly.mean()[yearly.size() > min_hours]
    return float(annual.std() / annual.mean())

# เรียนรู้ลมจากข้อมูลที่วัดได้
def fit(wind: pd.Series, era5: pd.DataFrame) -> dict:
    """
    INPUT: wind = Series ลมราย 10 นาทีที่วัดได้ (จะเป็นความสูงตรงไหนก็ได้เช่น WS160 หรือลมที่ hub height)
           era5 = DataFrame จาก reference.load_era5()
    OUTPUT: dict status Model ส่งต่อให้ expected() / simulate() / ensemble_stats()
    """
    site = get_site()
    step = pd.Timedelta(minutes = 24 * 60 // site["records_per_day"])
    y = wind.dropna()

    # M1 รูปร่างรอบปีและรอบวัน
    x = _fourier(y.index, FORECAST["harmonics_year"], FORECAST["harmonics_day"])
    shape = LinearRegression().fit(x, y)

    # ตัดข้อมูล ERA5 ที่อยู่หลังข้อมูลที่ใช้เทรนทิ้ง กันข้อมูลอนาคตรั่วเข้ามาใน climatology
    era5 = era5.loc[:y.index.max()]

    # M2 ข้อมูลระดับรายเดือน
    # "site" = ใช้ฤดูกาลของเสาเอง (Fourier ใน M1 มีอยู่แล้ว ไม่ต้องปรับ)
    # "era5" = บังคับค่าเฉลี่ยรายเดือนตาม ERA5 ระยะยาว * อัตราส่วนไซต์ ใช้ได้เฉพาะไซต์ที่เสาสอดคล้องกับ ERA5
    level = None
    if FORECAST["level_source"] == "era5":
        site_hourly = y.resample("1h").mean().dropna()
        era_overlap = era5["era_ws"].reindex(site_hourly.index).dropna()
        scale = site_hourly.loc[era_overlap.index].mean() / era_overlap.mean()
        level = era5["era_ws"].groupby(era5.index.month).mean() * scale

    # M3 ความแกว่งของข้อมูล = ยกบล็อกข้อมูลวัดจริงทั้งก้อนมาวาง (analog) เซนเซอร์ทุกตัวใช้บล็อกเดียวกัน
    full_index = pd.date_range(y.index.min(), y.index.max(), freq = step)
    measured = y.reindex(full_index)
    block = FORECAST["block_days"] * site["records_per_day"]

    # สุ่มได้เฉพาะบล็อกที่เริ่มเที่ยงคืน (รอบวันตรงกับปลายทาง) และมีข้อมูลจริง >= 90%
    valid = np.concatenate([[0], np.cumsum(measured.notna().to_numpy())])
    good = (valid[block:] - valid[:-block]) >= 0.9 * block
    midnight = (full_index.hour == 0) & (full_index.minute == 0)
    starts = np.flatnonzero(good & midnight[:len(good)])
    if len(starts) == 0:
        raise ValueError(f"ไม่มีช่วง {FORECAST['block_days']} วันที่ข้อมูลลมครบ 90% ให้ทำการสุ่ม เพื่อลดค่า block_days ใน config")

    # เดือนที่มีบล็อกน้อยกว่า min_blocks ยืมเดือนข้างเคียงมาเพิ่ม กันทุกรอบ ensemble ได้บล็อกเดิมซ้ำ
    by_month = {m: starts[full_index.month[starts] == m] for m in range(1, 13)}
    pools = {}
    for m in range(1, 13):
        pool = by_month[m]
        if len(pool) < FORECAST["min_blocks"]:
            pool = np.concatenate([by_month[(m - 2) % 12 + 1], pool, by_month[m % 12 + 1]])
        pools[m] = pool if len(pool) else starts

    return {
        "step": step,
        "shape": shape,
        "level": level,
        "iav": annual_iav(era5),
        "measured": measured.to_numpy(),
        "measured_index": full_index,
        # ponytail: กันหารด้วยค่าใกล้ 0 ถ้าไซต์ไหนลมเฉลี่ยรายชั่วโมงต่ำกว่า 0.5 m/s ต้องเปลี่ยนเป็นแบบบวก
        "shape_measured": np.maximum(shape.predict(_fourier(full_index, FORECAST["harmonics_year"], FORECAST["harmonics_day"])), 0.5),
        "block": block,
        "pools": pools,
        "trained_on": [str(y.index.min()), str(y.index.max())],
    }

# Expectation ของลม (ไม่มี noise) ใช้เป็น point forecast
def expected(state: dict, start, end) -> pd.Series:
    """
    INPUT: start, end = ช่วงเวลาที่ต้องการทำนาย เช่น "2025-01-01", "2025-12-31 23:50"
    OUTPUT: Series ลมคาดหวังที่ความละเอียดเดียวกับข้อมูลเสา
    """
    index = pd.date_range(start, end, freq = state["step"])
    base = pd.Series(state["shape"].predict(_fourier(index, FORECAST["harmonics_year"], FORECAST["harmonics_day"])), index = index)
    if state["level"] is None:
        return base.clip(lower = 0.0)
    # ดันค่าเฉลี่ยรายเดือนให้ตรงกับระดับค่าข้อมูลระยะยาว
    target = state["level"].reindex(index.month).to_numpy()
    return (base * target / base.groupby(index.month).transform("mean")).clip(lower = 0.0)

# สร้างลม 1 เส้น ที่มีความแกว่งของข้อมูลเหมือนของจริง
def simulate(state: dict, start, end, seed: int = 0, base: pd.Series | None = None) -> pd.Series:
    """
    สุ่มบล็อกข้อมูลวัดจริงยาว block_days จากเดือนเดียวกัน มาต่อกัน ปรับระดับด้วย _ratio แล้วคูณความผันผวนรายปี
    INPUT: base = ผลลัพธ์จาก expected() ส่งมาเพื่อให้ไม่ต้องคำนวณซ้ำตอนรัน ensemble
    OUTPUT: Series ลม (m/s) เอาข้อมูลไป resample ต่อได้ทุกค่าความละเอียด
    """
    base = expected(state, start, end) if base is None else base
    picks, year_factor = _blocks(state, base.index, seed)
    values = np.empty(len(base))
    for i, s, n in picks:
        values[i:i + n] = state["measured"][s:s + n] * _ratio(state, base, i, s, n)
    return pd.Series(values * year_factor, index = base.index)  # ช่วงที่บล็อกต้นทางไม่มีข้อมูลจะเป็น NaN เหมือนของจริง

# เลือกบล็อกต้นทาง ใช้ร่วมกันระหว่าง simulate() และ to_raw() ให้ seed เดียวกันได้ผลตรงกัน
def _blocks(state: dict, index, seed: int):
    """
    OUTPUT: (list ของ (ตำแหน่งปลายทาง, ตำแหน่งต้นทาง, ความยาว), ตัวคูณความผันผวนรายปี)
    """
    if index[0].hour or index[0].minute:
        raise ValueError(f"start ต้องเป็นเวลา 00:00 ให้รอบวันตรงกับบล็อกต้นทาง (ได้ {index[0]})")
    rng = np.random.default_rng(seed)
    block = state["block"]
    picks = [(i, int(rng.choice(state["pools"][index[i].month])), min(block, len(index) - i)) for i in range(0, len(index), block)]
    return picks, 1 + rng.normal(0, state["iav"])

# ตัวคูณปรับระดับจากเวลาต้นทางไปเวลาปลายทาง (ฤดูกาล/รอบวัน/ระดับระยะยาว)
def _ratio(state: dict, base: pd.Series, i: int, s: int, n: int):
    return base.to_numpy()[i:i + n] / state["shape_measured"][s:s + n]

# สร้างข้อมูลพยากรณ์ในรูปแบบเดียวกับไฟล์ Raw Data
def to_raw(state: dict, raw_file: str, start, end, seed: int = 0, site_code: str | None = None) -> pd.DataFrame:
    """
    ยก Raw Data ทุก channel ของบล็อกต้นทางมาวางที่เวลาปลายทาง ใช้บล็อกและตัวคูณชุดเดียวกับ simulate(seed)
    ค่าลมทุกความสูง (unit m/s) คูณตัวคูณเดียวกันทั้ง avg/gust/max/min/sd จึงรักษา wind shear ไว้
    ทิศทาง/อุณหภูมิ/ความดัน/ความชื้น คัดลอกตามเดิม
    INPUT: raw_file = ไฟล์ใน data/raw/ ที่ใช้ Train
    OUTPUT: DataFrame คอลัมน์เดียวกันกับ Raw Data เอาไป to_csv แล้วเข้า data_io.load_raw_data() ได้ทันที
    """
    site = get_site(site_code)
    raw_data = pd.read_csv(RAW_DIR / raw_file, dtype = {"site_code": str})
    raw_data = raw_data[raw_data["site_code"] == site["site_numeric_code"]]
    local = pd.to_datetime(raw_data["timestamp"], utc = True).dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    is_wind = (raw_data["unit"] == "m/s").to_numpy()
    wind_cols = ["avg", "gust", "max", "min", "sd"]
    offset = site["timezone"].replace(":", "")

    base = expected(state, start, end)
    picks, year_factor = _blocks(state, base.index, seed)
    source_index, step = state["measured_index"], state["step"]
    parts = []
    for i, s, n in picks:
        in_block = ((local >= source_index[s]) & (local <= source_index[s] + (n - 1) * step)).to_numpy()
        rows = raw_data[in_block].copy()
        target_time = local[in_block] + (base.index[i] - source_index[s])
        factor = pd.Series(_ratio(state, base, i, s, n), index = base.index[i:i + n]).reindex(target_time).to_numpy() * year_factor
        wind = is_wind[in_block]
        rows.loc[wind, wind_cols] = np.round(rows.loc[wind, wind_cols].to_numpy() * factor[wind, None], 6) # ทศนิยม 6 หลักเท่า Raw Data
        rows["timestamp"] = target_time.dt.strftime(f"%Y-%m-%d %H:%M:%S.000 {offset}").to_numpy()
        parts.append(rows)
    return pd.concat(parts).sort_values(["timestamp", "channel"], ascending = [False, True]).reset_index(drop = True)

# สรุปค่าลมต่อคาบเวลา
def wind_stats(ws: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """
    INPUT: ws = ลมราย 10 นาที : freq = "1h" , "D" , "MS" , "YS"
    OUTPUT: DataFrame สถิติลมต่อคาบ
    """
    group = ws.resample(freq)
    output = pd.DataFrame({
        "ws_mean": group.mean(),
        "ws_min": group.min(),
        "ws_max": group.max(),
        "ws_sd": group.std(),
        "ws_p10": group.quantile(0.10),
        "ws_p50": group.quantile(0.50),
        "ws_p90": group.quantile(0.90),
        "coverage_pct": group.count() / group.size() * 100,
    })
    output["ws_cv"] = output["ws_sd"] / output["ws_mean"]
    return output

# แปลงลมเป็นพลังงานต่อคาบเวลา
def energy_stats(ws: pd.Series, freq: str = "MS", rho: float = 1.225, turbine = None, losses = None) -> pd.DataFrame:
    """
    แปลงลมเป็นกำลังไฟฟ้าทีละ timestamp (ไม่ผ่าน Weibull) หัก losses แล้วรวมเป็นพลังงาน กรณีที่ช่วงที่ข้อมูลหาย เติมด้วยกำลังไฟเฉลี่ยของคาบเวลานั้น
    INPUT: rho = ความหนาแน่นของอากาศ ใช้ค่าเดียวกับ energy.calculate_aep เพื่อให้สามารถเทียบ AEP ได้
    OUTPUT: DataFrame {energy_mwh, capacity_factor_pct}
    """
    t = turbine or DEFAULT_TURBINE
    losses_t = losses or DEFAULT_LOSSES
    hours = 24 / get_site()["records_per_day"]
    corrected = ws.to_numpy() * (rho / 1.225) ** (1 / 3)
    kw = pd.Series(power_curve(corrected, t), index = ws.index).where(ws.notna())
    net = np.prod([1 - loss for loss in losses_t.values()]) * t["num_turbines"]

    group = kw.resample(freq)
    mean_kw = group.mean() * net
    return pd.DataFrame({
        "energy_mwh": mean_kw * group.size() * hours / 1000,
        "capacity_factor_pct": mean_kw / (t["rated_kw"] * t["num_turbines"]) * 100,
    })

# Run หลายรอบ แล้วสรุปเป็นข้อมูล P10/P50/P90
def ensemble_stats(state: dict, start, end, freq: str = "MS", n: int | None = None, rho: float = 1.225) -> pd.DataFrame:
    """
    OUTPUT: DataFrame คอลัมน์ <ค่า>_p10 / _p50 / _p90 เช่น ws_mean_p50, energy_mwh_p90
    ไม่เก็บลมทุกเส้นไว้ในหน่วยความจำ แต่จะเก็บแค่ค่าสถิติของแต่ละรอบ
    """
    num = n or FORECAST["ensemble"]
    base = expected(state, start, end)
    runs = []
    for seed in range(num):
        ws = simulate(state, start, end, seed, base)
        runs.append(wind_stats(ws, freq).join(energy_stats(ws, freq, rho)).drop(columns = "coverage_pct"))
    output = pd.concat(runs).groupby(level = 0).quantile(list(FORECAST["quantiles"])).unstack()
    output.columns = [f"{name}_p{round(q * 100)}" for name, q in output.columns]
    return output

# วัดผลที่หลายความละเอียด
def score_by_freq(actual: pd.Series, predicted: pd.Series, freqs = ("10min", "1h", "D", "MS"), min_coverage: float = 0.8) -> pd.DataFrame:
    """
    INPUT: actual = ลมวัดจริง (ตารางเวลาเต็ม มีค่า NaN ได้), predicted = ลมที่ทำนาย, min_coverage = คาบเวลาที่ Raw Data น้อยกว่านี้จะไม่นำมาคิด
    OUTPUT: DataFrame MAE / RMSE / MAPE / R2 / Hit10 / bias ต่อความละเอียด
    """
    rows = []
    for freq in freqs:
        group = actual.resample(freq)
        real = group.mean()[group.count() / group.size() >= min_coverage]
        pair = pd.concat({"real": real, "pred": predicted.resample(freq).mean()}, axis = 1, sort = True).dropna()
        if len(pair) < 2: # คาบเวลาที่มี Data ไม่พอ (เช่น YS ที่ coverage ต่ำ) จะข้ามไป
            continue
        score = evaluate(pair["real"], pair["pred"], freq)
        score["bias"] = float((pair["pred"] - pair["real"]).mean())
        score["n"] = len(pair)
        rows.append(score)
    return pd.DataFrame(rows).set_index("model")

# ตรวจความถูกต้องเบื้องต้น: python -m uwinControl.forecast
def _demo():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", "2024-12-31 23:50", freq = "10min")
    daily = pd.Series(rng.normal(0, 1.0, 366), index = pd.date_range("2024-01-01", periods = 366, freq = "D"))
    truth = pd.Series(5 + np.sin(2 * np.pi * idx.dayofyear / 365) + 0.8 * np.sin(2 * np.pi * idx.hour / 24) + daily.rolling(3, min_periods = 1).mean().reindex(idx, method = "ffill").to_numpy() + rng.normal(0, 0.5, len(idx)), index = idx)
    truth.iloc[1000:1500] = np.nan  # จำลองช่องว่างข้อมูล
    era_idx = pd.date_range("2010-01-01", "2025-12-31 23:00", freq = "1h")
    era5 = pd.DataFrame({"era_ws": 4 + np.sin(2 * np.pi * era_idx.dayofyear / 365) + rng.normal(0, 0.3, len(era_idx))}, index = era_idx)
    era5.loc["2025", "era_ws"] = 100.0  # ปีอนาคตค่าผิดชัด ๆ ถ้ารั่วเข้ามา level จะพุ่ง

    state = fit(truth, era5)
    assert 0 < state["iav"] < 0.05, f"IAV ผิดปกติ {state['iav']} (ERA5 ปี 2025 รั่วเข้ามาหรือเปล่า)"

    sim = simulate(state, "2025-01-01", "2025-12-31 23:50", seed = 1)
    assert len(sim) == 365 * 144 and (sim.dropna() >= 0).all() and sim.notna().mean() > 0.9
    assert abs(sim.mean() - truth.mean()) < 0.5, "ระดับลมเพี้ยน"
    assert 0.8 < sim.std() / truth.std() < 1.2, "ความแกว่งหาย"
    assert sim.groupby(sim.index.month).mean().idxmax() == truth.groupby(idx.month).mean().idxmax(), "ฤดูกาลเพี้ยน"

    monthly = wind_stats(sim, "MS")
    assert len(monthly) == 12 and (monthly["ws_min"] <= monthly["ws_mean"]).all() and (monthly["ws_max"] >= monthly["ws_mean"]).all()
    assert len(wind_stats(sim, "YS")) == 1

    # พลังงานรายปีต้องเท่ากับผลรวมรายเดือน และช่องว่างต้องไม่ทำให้พลังงานหาย
    e_month, e_year = energy_stats(sim, "MS"), energy_stats(sim, "YS")
    assert abs(e_month["energy_mwh"].sum() / e_year["energy_mwh"].iloc[0] - 1) < 0.01
    gappy = sim.copy()
    gappy.iloc[::2] = np.nan
    assert abs(energy_stats(gappy, "YS")["energy_mwh"].iloc[0] / e_year["energy_mwh"].iloc[0] - 1) < 0.03

    ens = ensemble_stats(state, "2025-01-01", "2025-12-31 23:50", "MS", n = 30)
    assert (ens["ws_mean_p10"] <= ens["ws_mean_p50"]).all() and (ens["ws_mean_p50"] <= ens["ws_mean_p90"]).all()

    scores = score_by_freq(truth, expected(state, "2024-01-01", "2024-12-31 23:50"))
    assert scores.loc["MS", "R2"] > scores.loc["10min", "R2"], "รวบข้อมูลแล้ว R2 ต้องดีขึ้น"
    print(scores.round(3))

    # to_raw: สร้างไฟล์ Raw จำลอง 3 channel (ลม 160 m, ลม 100 m, อุณหภูมิ) แล้วเช็คว่าตรงกับ simulate()
    import tempfile
    from pathlib import Path
    site = get_site()
    ok = truth.dropna()
    stamp = ok.index.strftime("%Y-%m-%d %H:%M:%S.000 +0700")
    channels = [(1, 107, 160.0, "m/s", ok.to_numpy()), (5, 111, 100.0, "m/s", ok.to_numpy() * 0.9), (16, 122, 155.0, "C", np.full(len(ok), 27.0))]
    fake = pd.concat([pd.DataFrame({"site_code": site["site_numeric_code"], "sensor_id": sid, "timestamp": stamp, "channel": ch, "height": h,
                                    "direction": "N", "unit": unit, "avg": v, "gust": v * 1.1, "gust_dir": np.nan, "max": v * 1.2,
                                    "min": v * 0.8, "sd": v * 0.1}) for ch, sid, h, unit, v in channels])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fake_raw.csv"
        fake.to_csv(path, index = False)
        out = to_raw(state, str(path), "2025-01-01", "2025-01-21 23:50", seed = 1)
    assert list(out.columns) == list(fake.columns), "คอลัมน์ไม่ตรงกับ Raw Data"
    out_time = pd.to_datetime(out["timestamp"], utc = True).dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    assert out_time.min() >= pd.Timestamp("2025-01-01") and out_time.max() <= pd.Timestamp("2025-01-21 23:50")
    top = pd.Series(out.loc[out["channel"] == 1, "avg"].to_numpy(), index = out_time[out["channel"] == 1]).sort_index()
    expect = simulate(state, "2025-01-01", "2025-01-21 23:50", seed = 1).dropna()
    assert np.allclose(top.to_numpy(), expect.loc[top.index].to_numpy()), "ลมยอดเสาใน to_raw ไม่ตรงกับ simulate"
    low = pd.Series(out.loc[out["channel"] == 5, "avg"].to_numpy(), index = out_time[out["channel"] == 5]).sort_index()
    assert np.allclose(low.to_numpy() / top.to_numpy(), 0.9), "wind shear ไม่ถูกรักษาไว้"
    assert (out.loc[out["channel"] == 16, "avg"] == 27.0).all(), "channel ที่ไม่ใช่ลมต้องคัดลอกตามเดิม"
    print("forecast self-check ผ่าน")


if __name__ == "__main__":
    _demo()