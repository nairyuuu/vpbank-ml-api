import pandas as pd
import optuna
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from lightgbm import LGBMClassifier
import joblib

# ─── Load và sort theo thời gian ───────────────
df = pd.read_csv(r"F:\hackathon\models\qr\features_qr_boosted.csv")
df = df.sort_values("event_timestamp")

drop_cols = ["entity_id_hash", "event_timestamp", "email_domain", "ip_address", "user_agent"]
drop_cols = [col for col in drop_cols if col in df.columns]
X_full = df.drop(columns=["target"] + drop_cols)
y_full = df["target"]

# ─── Time-based split ──────────────────────────
split_index = int(len(df) * 0.8)
X_train = X_full.iloc[:split_index]
y_train = y_full.iloc[:split_index]
X_valid = X_full.iloc[split_index:]
y_valid = y_full.iloc[split_index:]

# ─── Optuna objective ──────────────────────────
def objective(trial):
    clf = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        num_leaves=trial.suggest_int("num_leaves", 15, 128),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 100),
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    preds = clf.predict(X_valid)
    return f1_score(y_valid, preds)

# ─── Tối ưu ─────────────────────────────────────
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("✅ Best trial:")
print(study.best_trial)

# ─── Train lại với best params ─────────────────
best_params = study.best_params
final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# ─── Evaluate ──────────────────────────────────
y_pred = final_model.predict(X_valid)
print("📊 Boosted QR + Optuna (time split):")
print(classification_report(y_valid, y_pred, digits=4))
print("F1:", f1_score(y_valid, y_pred))
print("Precision:", precision_score(y_valid, y_pred))
print("Recall:", recall_score(y_valid, y_pred))

joblib.dump(final_model, "models/qr_boosted_lgbm_optuna_timesplit.pkl")
print("✅ Model saved to models/qr_boosted_lgbm_optuna_timesplit.pkl")
