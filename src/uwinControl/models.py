"""
models.py คือโมเดลที่เอาไว้เทรนโมเดล และวัดผลความแม่นยำของโมเดล
"""

# import libraries
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODEL_DIR

# วัดความแม่นยำของการ Prediction
def evaluate(y_true, y_pred, label: str, collector: list | None = None) -> dict:
    """
    INPUT : y_true, y_pred = ค่าจริง/ค่าทำนาย | label = ชื่อโมเดล
    OUTPUT: dict {model, MAE, RMSE, MAPE, R2}
    """
    above = y_true > 3 # ตัดลมอ่อนออกตอนคิด MAPE
    score = {
        "model": label,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true[above] - y_pred[above]) / y_true[above])) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if collector is not None:
        collector.append(score)
    return score

# สูตรใช้เป็นเกณฑ์เปรียบเทียบ
def power_law_baseline(wind_lower, height_lower, height_upper, alpha):
    """Baseline ที่ใช้ประเมิน ML : v2 = v1 * (h2/h1)**α"""
    return wind_lower * (height_upper / height_lower) ** alpha

# สร้างโมเดล Random Forest
def make_random_forest(**kwargs) -> RandomForestRegressor:
    """สร้างโมเดล Random Forest ด้วยค่าตั้งต้น"""
    params = dict(n_estimators = 300, min_samples_leaf = 5, max_features = "sqrt", random_state = 42, n_jobs = -1)
    params.update(kwargs)
    return RandomForestRegressor(**params)

# เทรนโมเดลทั้งหมดทั้ง 4 โมเดล
def train_all(train, test, feature_cols, target, alpha_site, height_lower=100, height_upper=160) -> tuple[pd.DataFrame, dict]:
    """
    เทรนโมเดลทั้งชุด: Power Law → Linear → RandomForest → RandomForest ผ่าน α
    OUTPUT: (ตารางเปรียบเทียบผล, dict ของโมเดลที่เทรนแล้ว)
    """
    X_train = train[feature_cols]
    y_train = train[target]

    X_test = test[feature_cols]
    y_test = test[target]

    scores = []
    models = {}

    evaluate(y_test, power_law_baseline(X_test[f"WS{height_lower}"], height_lower, height_upper, alpha_site), "Power Law (baseline)", scores)

    models["linear"] = LinearRegression().fit(X_train, y_train)
    evaluate(y_test, models["linear"].predict(X_test), "Linear Regression", scores)

    models["random_forest"] = make_random_forest().fit(X_train, y_train)
    evaluate(y_test, models["random_forest"].predict(X_test), "Random Forest", scores)

    models["random_forest_alpha"] = make_random_forest().fit(X_train, train["alpha_observed"])
    alpha_pred = models["random_forest_alpha"].predict(X_test)
    evaluate(y_test, X_test[f"WS{height_lower}"] * (height_upper / height_lower)**alpha_pred, "Random Forest ผ่าน α", scores)

    return pd.DataFrame(scores).set_index("model").round(4), models

# บันทึกโมเดลที่ได้ทำการเทรนไปแล้ว
def save_model(model, name: str, data_version: str, feature_cols: list, scores: dict | None = None, note: str = "") -> str:
    """
    บันทึกโมเดล พร้อมทำ model card เพื่อบอกว่าโมเดลเทรนจากข้อมูลเวอร์ชันไหนมา
    OUTPUT: ชื่อไฟล์ที่บันทึกไว้
    """
    stamp = f"{name}_{data_version}"
    joblib.dump(model, MODEL_DIR / f"{stamp}.pkl")
    card = {
        "model_name": name, 
        "data_version": data_version,
        "features": feature_cols, 
        "scores": scores,
        "trained_at": datetime.now().isoformat(timespec="seconds"), 
        "note": note
        }
    (MODEL_DIR / f"{stamp}.json").write_text(json.dumps(card, indent = 2, ensure_ascii = False), encoding = "utf-8")
    return stamp