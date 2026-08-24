"""
models.py คือโมเดลที่เอาไว้เทรนโมเดล และวัดผลความแม่นยำของโมเดล
"""

# import libraries
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor ,HistGradientBoostingRegressor ,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODEL_DIR, get_site

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

# สร้างโมเดล MLP
def make_mlp(**kwargs):
    """MLP สำหรับทำนายค่าลม โดยใช้ StandardScaler ก่อนเข้าโมเดล"""
    params = dict(
        hidden_layer_sizes = (64, 32),
        activation = "relu",
        solver = "adam",
        learning_rate_init = 0.001,
        max_iter = 1000,
        random_state = 42,
        early_stopping = False,
    )
    params.update(kwargs)
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(**params)
    )

def make_qgb(**kwargs):
    """
    Quantile Gradient Boosting
    ค่า alpha = 0.50 หมายถึงการทำนายค่า P50
    """
    params = dict(
        loss = "quantile",
        alpha = 0.50,
        n_estimators = 300,
        learning_rate = 0.05,
        max_depth = 3,
        min_samples_leaf = 5,
        random_state = 42,
    )
    params.update(kwargs)
    return GradientBoostingRegressor(**params)


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
        "mlp": ("MLP", make_mlp()),
        "qgb": ("Quantile GB (median)", make_qgb()),
        
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

# เทรนโมเดลทั้งหมด
def train_all(train, test, feature_cols, target, alpha_site, height_lower = None, height_upper = None, site_code = None) -> tuple[pd.DataFrame, dict]:
    """
    เทรนโมเดลทั้งชุด: Power Law → Linear → RandomForest → RandomForest ผ่าน Alpha
    height_lower/height_upper ถ้าไม่ส่งมา จะอ่านจาก config ของไซต์ (base_sensor และยอดของเสา)
    OUTPUT: (ตารางเปรียบเทียบผล, dict ของโมเดลที่เทรนแล้ว)
    """
    site = get_site(site_code)
    height_lower = height_lower or site["sensor_heights"][site["base_sensor"]]
    height_upper = height_upper or site["mast_height_m"]

    base_col = f"WS{height_lower}"
    if base_col not in feature_cols:
        raise ValueError(f"Power Law baseline ต้องใช้ '{base_col}' แต่ไม่มีใน feature_cols: {feature_cols}")

    x_train = train[feature_cols]
    y_train = train[target]

    x_test = test[feature_cols]
    y_test = test[target]

    scores = []
    models = {}

    # Baseline
    evaluate(y_test, power_law_baseline(x_test[base_col], height_lower, height_upper, alpha_site), "Power Law (baseline)", scores)

    # Model ทุกตัวใน def build_model
    for key, (label, estimator) in build_models().items():
        estimator.fit(x_train, y_train)
        models[key] = estimator
        evaluate(y_test, estimator.predict(x_test), label, scores)

    # Model ที่ทำนาย alpha
    alpha_models = {
        "random_forest_alpha": (
            "Random Forest pass α",
            make_random_forest()
        ),
        "mlp_alpha": (
            "MLP pass α",
            make_mlp()
        ),
        "qgb_alpha": (
            "QGB pass α",
            make_qgb()
        ),
    }
    for key, (label, estimator) in alpha_models.items():
        estimator.fit(x_train, train["alpha_observed"])
        models[key] = estimator
        alpha_predic = estimator.predict(x_test)

        wind_predic = (
            x_test[base_col]
            * (height_upper / height_lower) ** alpha_predic
        )

        evaluate(
            y_test,
            wind_predic,
            label,
            scores
        )

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

def predict_hub_wind(model, features, mode, base_col, height_lower, height_upper):
    """
    mode = direct: โมเดลที่เอาไว้ทำนายลมที่ความสูง hub height
    mode = alpha: โมเดลทำนายค่า alpha แล้วใช้ Power Law
    """
    prediction = np.asarray(model.predict(features)).ravel()

    if mode == "direct":
        return prediction

    if mode == "alpha":
        return (features[base_col].to_numpy() * (height_upper / height_lower) ** prediction)

    raise ValueError(f"not found: {mode}")