"""
models.py คือโมเดลที่เอาไว้เทรนโมเดล และวัดผลความแม่นยำของโมเดล
"""

# import libraries
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor ,HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODEL_DIR

# โมเดลเสริมที่ติดตั้งเพิ่ม
try:
    from xgboost import XGBRegressor
    _HAS_XGR = True
except ImportError:
    _HAS_XGR = False
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

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
    params = dict(n_estimators = 300, 
                  min_samples_leaf = 5, 
                  max_features = "sqrt", 
                  random_state = 42, 
                  n_jobs = -1)
    params.update(kwargs)
    return RandomForestRegressor(**params)

# โมเดลที่ใช้ในการ Prediction
def build_models() -> dict:
    """
    OUTPUT: dict {key: (label, estimator)} ของโมเดลที่ทำนาย
    """
    models = {
        "linear": ("Linear Regression", LinearRegression()),
        "random_forest": ("Random Forest", make_random_forest()),
        "hist_gb": ("HistgradientBoosting", HistGradientBoostingRegressor(max_iter = 300, random_state = 42)),
        "svr": ("SVR", SVR(C = 10, epsilon = 0.1)),
    }
    if _HAS_XGR:
        models["xgboost"] = ("XGBoost", XGBRegressor(n_estimators = 300,
                                                     max_depth = 6,
                                                     learning_rate = 0.05,
                                                     subsample = 0.9,
                                                     random_state = 42,
                                                     n_jobs = 1))
    if _HAS_LGBM:
        models["lightgbm"] = ("LightGBM", LGBMRegressor(n_estimators = 300, 
                                                        learning_rate = 0.05,
                                                        random_state = 42,
                                                        n_jobs = 1,
                                                        verbose = -1))
    return models

# เทรนโมเดลทั้งหมดทั้ง 4 โมเดล
def train_all(train, test, feature_cols, target, alpha_site, height_lower = 100, height_upper = 160) -> tuple[pd.DataFrame, dict]:
    """
    เทรนโมเดลทั้งชุด: Power Law → Linear → RandomForest → RandomForest ผ่าน α
    OUTPUT: (ตารางเปรียบเทียบผล, dict ของโมเดลที่เทรนแล้ว)
    """
    x_train = train[feature_cols]
    y_train = train[target]

    x_test = test[feature_cols]
    y_test = test[target]

    scores = []
    models = {}

    # Baseline
    evaluate(y_test, power_law_baseline(x_test[f"WS{height_lower}"], height_lower, height_upper, alpha_site), "Power Law (baseline)", scores)

    # Model ทุกตัวใน def build_model
    for key, (label, estimator) in build_models().items():
        estimator.fit(x_train, y_train)
        models[key] = estimator
        evaluate(y_test, estimator.predict(x_test), label, scores)

    # RF ที่ทำนาย alpha
    models["random_forest_alpha"] = make_random_forest().fit(x_train, train["alpha_observed"])
    alpha_pred = models["random_forest_alpha"].predict(x_test)
    evaluate(y_test, x_test[f"WS{height_lower}"] * (height_upper / height_lower)**alpha_pred, "Random Forest pass α", scores)

    #เรียงผลโมเดล
    table_model = pd.DataFrame(scores).set_index("model").sort_values("RMSE").round(4)
    return table_model, models

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