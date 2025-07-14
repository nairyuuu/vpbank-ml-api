import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, precision_recall_curve
)

# ─── Load dữ liệu ───────────────────────────────
df = pd.read_csv("models/qr/features_qr_boosted.csv")
df = df.sort_values("event_timestamp")
df_meta = df.copy()

X = df.drop(columns=["target", "entity_id_hash", "event_timestamp"], errors="ignore")
X = X.select_dtypes(include=["number"]).fillna(0)
y = df["target"]

# ─── Time-based split (80/20) ───────────────────
split_index = int(len(df) * 0.8)
X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]
df_meta_valid = df_meta.iloc[split_index:].reset_index(drop=True)
valid_idx = X_valid.index
stack_y = y_valid.reset_index(drop=True)

# ─── Base models ────────────────────────────────
xgb_model = XGBClassifier(
    n_estimators=173,
    max_depth=9,
    learning_rate=0.0119,
    scale_pos_weight=2.535,
    subsample=0.7295,
    colsample_bytree=0.8835,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
score_xgb = xgb_model.predict_proba(X_valid)[:, 1]

model_lgb = lgb.LGBMClassifier(
    learning_rate=0.01, num_leaves=72, max_depth=10,
    min_child_samples=98, scale_pos_weight=5.5, n_estimators=100,
    random_state=42, n_jobs=-1
)
model_lgb.fit(X_train, y_train)
score_lgb = model_lgb.predict_proba(X_valid)[:, 1]

model_rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)
model_rf.fit(X_train, y_train)
score_rf = model_rf.predict_proba(X_valid)[:, 1]

# ─── Rule logic ────────────────────────────────
score_rule = (
    (df_meta_valid["amount_usd"] > 10_000_000) &
    (df_meta_valid["is_night"] == 1) &
    (df_meta_valid["geo_distance_from_last_txn"] > 50)
).astype(int)

# ─── Meta model input ──────────────────────────
stack_X = pd.DataFrame({
    "score_lgb": score_lgb,
    "score_rf": score_rf,
    "score_rule": score_rule,
    "amount_log": X.loc[valid_idx, "amount_log"].reset_index(drop=True),
    "geo_distance": X.loc[valid_idx, "geo_distance_from_last_txn"].reset_index(drop=True),
    "velocity_1h": X.loc[valid_idx, "velocity_1h"].reset_index(drop=True),
    "rank_amount_per_day": X.loc[valid_idx, "rank_amount_per_day"].reset_index(drop=True),
    "geo_speed_km_per_min": X.loc[valid_idx, "geo_speed_km_per_min"].reset_index(drop=True),
    "same_device_txn_1h": X.loc[valid_idx, "same_device_txn_1h"].reset_index(drop=True),
})

# ─── Meta model = LightGBM ─────────────────────
meta_model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.05)
meta_model.fit(stack_X, stack_y)
score_stack = meta_model.predict_proba(stack_X)[:, 1]

# ─── Rule mềm: boost score nếu nghi ngờ ────────
rule_soft = (
    (df_meta_valid["amount_usd"] > 356) &
    (df_meta_valid["same_device_txn_1h"] > 1)
)
score_stack[rule_soft] = np.clip(score_stack[rule_soft] + 0.25, 0, 1)

# ─── Optimize blending weights via Optuna ──────
def objective(trial):
    w = trial.suggest_float("w", 0.2, 0.8)
    blended = w * score_xgb + (1 - w) * score_stack
    prec, rec, thresh = precision_recall_curve(stack_y, blended)
    f1s = 2 * prec * rec / (prec + rec + 1e-6)
    return np.max(f1s)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
w_opt = study.best_params["w"]

# ─── Dự đoán với trọng số tốt nhất ─────────────
score_ensemble = w_opt * score_xgb + (1 - w_opt) * score_stack

# ─── Tìm threshold tối ưu lại ──────────────────
prec, rec, thresh = precision_recall_curve(stack_y, score_ensemble)
f1s = 2 * prec * rec / (prec + rec + 1e-6)
best_idx = np.argmax(f1s)
best_thresh = thresh[best_idx]

final_preds = (score_ensemble > best_thresh).astype(int)

# ─── Rule strict: override = 1 nếu chắc chắn ───
rule_strict = (
    (df_meta_valid["amount_usd"] > 918) &
    (df_meta_valid["geo_distance_from_last_txn"] > 0) &
    (df_meta_valid["velocity_1h"] > 1)
)
final_preds[rule_strict] = 1

# ─── Evaluation ────────────────────────────────
results = {
    "f1": f1_score(stack_y, final_preds),
    "precision": precision_score(stack_y, final_preds),
    "recall": recall_score(stack_y, final_preds),
    "auc": roc_auc_score(stack_y, score_ensemble),
    "threshold": round(float(best_thresh), 4),
    "blend_weight_xgb": round(float(w_opt), 3),
}
print("\n✅ Test11 – MetaModel = LightGBM + Optuna tuned weight:")
print(results)
print("\n📊 Classification report:")
print(classification_report(stack_y, final_preds, digits=4))
