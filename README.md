# VPBank ML API - Fraud Detection System

A real-time fraud detection system that processes transactions from Kafka and provides ML-based fraud scoring using ONNX models.

## Features

- **Real-time Processing**: Consumes transactions from Kafka topic `vpbank-enriched-features`
- **ML Models**: Supports QR, IBFT, and Top-up transaction types using ONNX models
- **Alerting**: Sends fraud alerts via AWS SNS
- **Decision Integration**: Sends decisions to external decision system
- **Health Monitoring**: Comprehensive health checks and monitoring
- **Docker Support**: Fully containerized with Docker Compose

## Architecture

```
Kafka Topic → ML API → ONNX Models → Fraud Score → Alerts (SNS) + Decision System
```

## Sample Input Data from Kafka Topic

The system processes enriched transaction data from the `vpbank-enriched-features` Kafka topic:
```python

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

```

```python
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

```

This is the first row of the csv that used to train the model:

```
entity_id_hash,event_timestamp,is_weekend,is_night,txn_type_idx,is_ibft,is_topup,is_qr,amount_usd,amount_log,is_high_amount,name_is_ascii,name_has_digit,name_has_symbol,name_repeated_char,email_domain,disposable_email_flag,billing_lat,billing_long,ip_address,user_agent,seconds_since_last_txn,is_new_device,geo_distance_from_last_txn,suspicious_agent,is_business_hours,target,sum_amount_1h,avg_amount_1h,txn_count_1h,sum_amount_24h,avg_amount_24h,txn_count_24h,velocity_1h,rank_amount_per_day,change_in_user_agent,geo_speed_km_per_min,same_device_txn_1h
```

Don't try to make up or make any change, just make the system do prediction, alert (using AWS SNS), send the output to the decision system so that it can decide what to do with the transaction

The tech stack of the system gonna use ExpressJS, ONNXjs and docker.

```json
{
    "transaction_id": "65a6cbd5-e478-41be-ad54-26f62249bf42",
    "user_id": "44ad0106-aa4f-4112-b92c-aac8ad5806a4",
    "merchant_id": "1c0c2de5-d426-4efd-aaef-07218cc33827",
    "timestamp": "2025-05-05T05:59:35",
    "transaction": {
        "amount_raw": 944.26,
        "amount_log": 6.850401551952162,
        "amount_range": "micro",
        "is_round_amount": false,
        "amount_digits": 3,
        "txn_type": "QR",
        "is_payment": true,
        "is_transfer": false,
        "is_withdrawal": false,
        "is_deposit": false,
        "is_topup": false,
        "channel": "mobile",
        "is_online": false,
        "is_atm": false,
        "is_pos": false,
        "is_mobile": true,
        "is_branch": false,
        "currency": "JPY",
        "is_foreign_currency": false,
        "description_length": 15,
        "description_words": 3,
        "has_description": true
    },
    "temporal": {
        "hour": 5,
        "minute": 59,
        "day_of_week": 0,
        "day_of_month": 5,
        "month": 5,
        "year": 2025,
        "quarter": 2,
        "week_of_year": 19,
        "time_of_day": "morning",
        "is_business_hours": false,
        "is_banking_hours": false,
        "is_weekend": false,
        "is_weekday": true,
        "is_morning_peak": false,
        "is_evening_peak": false,
        "is_lunch_time": false,
        "is_late_night": false,
        "is_very_early": true,
        "hour_sin": 0.9659,
        "hour_cos": 0.2588,
        "dow_sin": 0.0,
        "dow_cos": 1.0,
        "month_sin": 0.866,
        "month_cos": -0.5,
        "minutes_since_midnight": 359,
        "days_since_epoch": 20213
    },
    "location": {
        "billing_lat": 1.0567995,
        "billing_long": -35.044979,
        "geo_distance_from_last_txn": 8664.03
    },
    "aggregation": {
        "user_txn_count_1h": 1,
        "user_amount_sum_1h": 2863.05,
        "user_amount_avg_1h": 2863.05,
        "user_unique_merchants_1h": 0,
        "user_txn_count_24h": 1,
        "user_amount_sum_24h": 2863.05,
        "user_amount_avg_24h": 2863.05,
        "user_unique_merchants_24h": 0,
        "user_unique_channels_24h": 1,
        "merchant_txn_count_1h": 0,
        "merchant_amount_sum_1h": 0.0,
        "merchant_unique_users_1h": 0,
        "device_txn_count_1h": 0,
        "ip_txn_count_1h": 0,
        "seconds_since_last_txn": 7081851,
        "is_velocity_anomaly": false,
        "is_amount_anomaly": false,
        "is_new_merchant": false,
        "is_new_location": false,
        "is_rapid_fire": false
    },
    "behavioral": {
        "entity_id_hash": "9b205471fe709b7ad6a37146aa6c396a",
        "user_id_hash": "9b205471fe709b7ad6a37146aa6c396a",
        "merchant_id_hash": "560570a6fdf821c5206784f1dcbfc225",
        "device_id_hash": null,
        "ip_address_hash": "8e895554326c83cd69d3000d72c25693",
        "is_new_user": false,
        "is_new_device": false,
        "is_new_ip": true,
        "email_domain": null,
        "is_disposable_email": false,
        "has_common_email_domain": false,
        "name_is_ascii": true,
        "name_has_digit": false,
        "name_has_symbol": false,
        "name_repeated_char": false
    },
    "feature_extraction_timestamp": "2025-07-13T13:32:37.817854",
    "pipeline_version": "1.0.0"
}
```

## Model Training Information

The system maps Kafka enriched features to the ONNX models that were trained using the approach outlined in the provided training scripts. The models were trained using:

1. **Base Model Training** - LightGBM with Optuna hyperparameter optimization
2. **Meta Model Ensemble** - Combination of XGBoost, LightGBM, Random Forest with meta-learning
3. **Rule-based Enhancements** - Additional business logic for edge cases

Expected CSV training format headers:
```
entity_id_hash,event_timestamp,is_weekend,is_night,txn_type_idx,is_ibft,is_topup,is_qr,amount_usd,amount_log,is_high_amount,name_is_ascii,name_has_digit,name_has_symbol,name_repeated_char,email_domain,disposable_email_flag,billing_lat,billing_long,ip_address,user_agent,seconds_since_last_txn,is_new_device,geo_distance_from_last_txn,suspicious_agent,is_business_hours,target,sum_amount_1h,avg_amount_1h,txn_count_1h,sum_amount_24h,avg_amount_24h,txn_count_24h,velocity_1h,rank_amount_per_day,change_in_user_agent,geo_speed_km_per_min,same_device_txn_1h
```

## Quick Start

### Prerequisites
- Node.js 18+
- Docker & Docker Compose
- ONNX model files (see `models/README.md`)

### Local Development

1. **Clone and Install**
```bash
git clone <repository>
cd vpbank-ml-api
npm install
```

2. **Environment Setup**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start Services**
```bash
# Start Kafka and dependencies
docker-compose up -d kafka zookeeper

# Start the application
npm run dev
```

### Production Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f vpbank-ml-api
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BROKERS` | Kafka broker addresses | `localhost:9092` |
| `KAFKA_TOPIC` | Input topic name | `vpbank-enriched-features` |
| `AWS_SNS_TOPIC_ARN` | SNS topic for alerts | - |
| `DECISION_SYSTEM_URL` | Decision system endpoint | - |
| `QR_FRAUD_THRESHOLD` | QR model threshold | `0.5` |
| `IBFT_FRAUD_THRESHOLD` | IBFT model threshold | `0.5` |
| `TOPUP_FRAUD_THRESHOLD` | Top-up model threshold | `0.5` |

### Model Configuration

Place ONNX model files in the `models/` directory:
- `qr_model.onnx`
- `ibft_model.onnx` 
- `topup_model.onnx`

See `models/README.md` for model conversion instructions.

## API Endpoints

### Health & Status
- `GET /health` - Comprehensive health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

### Fraud Detection
- `POST /api/predict` - Manual fraud prediction
- `GET /api/models/status` - Model status
- `POST /api/models/reload` - Reload models

### Monitoring
- `GET /api/services/status` - All services status
- `POST /api/test/alert` - Send test alert

## Data Flow

### Output Format
```json
{
  "transactionId": "uuid",
  "fraudScore": 0.85,
  "isFraud": true,
  "decision": {
    "action": "BLOCK",
    "reason": "High fraud risk detected",
    "riskLevel": "HIGH"
  }
}
```

## Feature Engineering

The system maps Kafka enriched features to model features:

- **Temporal**: Weekend, night time, business hours
- **Transaction**: Amount, type, currency conversion
- **Behavioral**: Device, user, email patterns
- **Location**: Geographic distance, speed calculations
- **Aggregation**: Historical transaction patterns

See `src/utils/featureMapper.js` for complete mapping logic.

## Fraud Detection Logic

### Model Selection
- QR transactions → `qr_model.onnx`
- IBFT transactions → `ibft_model.onnx`  
- Top-up transactions → `topup_model.onnx`

### Decision Rules
- Score ≥ 0.9 → **BLOCK** (Critical risk)
- Score ≥ 0.7 → **BLOCK** + Review (High risk)
- Score ≥ 0.5 → **CHALLENGE** (Medium risk)
- Score ≥ 0.3 → **MONITOR** (Low risk)
- Score < 0.3 → **ALLOW** (Minimal risk)

## Alerting

### AWS SNS Integration
Fraud alerts are sent to configured SNS topic with:
- Transaction details
- Risk assessment
- Risk factors
- Severity level

### Alert Severity Levels
- **CRITICAL**: Score ≥ 0.9
- **HIGH**: Score ≥ 0.7
- **MEDIUM**: Score ≥ 0.5
- **LOW**: Score < 0.5

## Monitoring & Observability

### Logging
- Structured JSON logs
- Request/response tracking
- Error handling and stack traces
- Performance metrics

### Health Checks
- Model availability
- Kafka connectivity
- Service dependencies
- System resources

### Metrics
- Processing time per transaction
- Fraud detection rates
- Model performance
- System resource usage

## Development

### Project Structure
```
src/
├── services/           # Core business services
│   ├── fraudDetectionService.js
│   ├── kafkaConsumer.js
│   ├── alertService.js
│   └── decisionService.js
├── routes/            # API routes
├── utils/             # Utilities
└── server.js          # Main application
```

### Testing
```bash
# Install dependencies
npm install

# Test Docker build
npm run docker:build

# Start development server
npm run dev
```

### Adding New Models
1. Train model and convert to ONNX
2. Place in `models/` directory
3. Update `fraudDetectionService.js` model paths
4. Update feature mapping if needed
5. Configure threshold in environment

## Security Considerations

- API key authentication for decision system
- Input validation and sanitization
- Rate limiting (recommended)
- Network security in production
- Secure model file storage
- Audit logging for compliance

## Performance

### Optimizations
- ONNX runtime for fast inference
- Async processing pipeline
- Connection pooling
- Graceful error handling
- Memory management

### Scaling
- Horizontal scaling with multiple instances
- Kafka consumer group for load distribution
- Health checks for load balancer integration
- Resource limits in Docker

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Check file paths in environment variables
   - Verify ONNX file format and compatibility
   - Check logs for specific errors

2. **Kafka connection issues**
   - Verify broker addresses and ports
   - Check network connectivity
   - Validate topic existence

3. **High memory usage**
   - Monitor model sizes
   - Check for memory leaks in logs
   - Restart service if needed

4. **AWS SNS failures**
   - Verify AWS credentials and permissions
   - Check topic ARN configuration
   - Review AWS service limits

### Debug Mode
Set `LOG_LEVEL=debug` for detailed logging.

## Tech Stack

- **ExpressJS** - Web framework
- **ONNX Runtime** - ML model inference
- **KafkaJS** - Kafka client
- **AWS SDK** - SNS integration
- **Docker** - Containerization
- **Winston** - Logging
- **Joi** - Input validation

## Support

- Check logs in `/logs` directory
- Use health endpoints for diagnostics
- Monitor system resources
- Review Kafka consumer lag

## License

MIT License - see LICENSE file for details.

---

**Note**: This system performs fraud detection exactly as the models were trained, without modifications to the prediction logic. The system focuses on real-time processing, alerting via AWS SNS, and sending decisions to the decision system for transaction processing decisions.