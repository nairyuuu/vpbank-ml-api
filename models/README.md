# VPBank ML API Models

This directory contains the ONNX models for fraud detection.

## Model Files Expected:
- `qr_model.onnx` - QR transaction fraud detection model
- `ibft_model.onnx` - IBFT transaction fraud detection model  
- `topup_model.onnx` - Top-up transaction fraud detection model

## Converting Models to ONNX

To convert your trained LightGBM models to ONNX format, use the following Python script:

```python
import joblib
import lightgbm as lgb
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

# Load your trained model
model = joblib.load('path/to/your/model.pkl')

# Define input shape (number of features)
# Update this based on your actual feature count
n_features = 32  # Adjust this number

initial_type = [('float_input', FloatTensorType([None, n_features]))]

# Convert to ONNX
onnx_model = convert_lightgbm(model, initial_types=initial_type)

# Save the ONNX model
onnx.save_model(onnx_model, 'qr_model.onnx')
```

## Model Requirements:
- Input: Float32 tensor with shape [1, n_features]
- Output: Float32 tensor with shape [1, 2] (probabilities for [non-fraud, fraud])
- All models should be trained on the same feature set for consistency

## Feature Order:
The models expect features in this exact order:
1. is_weekend
2. is_night  
3. txn_type_idx
4. is_ibft
5. is_topup
6. is_qr
7. amount_usd
8. amount_log
9. is_high_amount
10. name_is_ascii
11. name_has_digit
12. name_has_symbol
13. name_repeated_char
14. disposable_email_flag
15. billing_lat
16. billing_long
17. seconds_since_last_txn
18. is_new_device
19. geo_distance_from_last_txn
20. suspicious_agent
21. is_business_hours
22. sum_amount_1h
23. avg_amount_1h
24. txn_count_1h
25. sum_amount_24h
26. avg_amount_24h
27. txn_count_24h
28. velocity_1h
29. rank_amount_per_day
30. change_in_user_agent
31. geo_speed_km_per_min
32. same_device_txn_1h
