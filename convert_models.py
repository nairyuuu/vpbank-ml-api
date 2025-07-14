#!/usr/bin/env python3
"""
Model Converter Script for VPBank ML API
Converts trained LightGBM models to ONNX format for use in the fraud detection system.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
except ImportError as e:
    print(f"Error: Missing required packages. Please install:")
    print("pip install lightgbm onnxmltools onnx skl2onnx")
    print(f"Missing package: {e}")
    sys.exit(1)

# Feature order based on training CSV
FEATURE_NAMES = [
    'is_weekend', 'is_night', 'txn_type_idx', 'is_ibft', 'is_topup', 'is_qr',
    'amount_usd', 'amount_log', 'is_high_amount', 'name_is_ascii', 'name_has_digit',
    'name_has_symbol', 'name_repeated_char', 'disposable_email_flag', 'billing_lat',
    'billing_long', 'seconds_since_last_txn', 'is_new_device', 'geo_distance_from_last_txn',
    'suspicious_agent', 'is_business_hours', 'sum_amount_1h', 'avg_amount_1h',
    'txn_count_1h', 'sum_amount_24h', 'avg_amount_24h', 'txn_count_24h',
    'velocity_1h', 'rank_amount_per_day', 'change_in_user_agent',
    'geo_speed_km_per_min', 'same_device_txn_1h'
]

def convert_lightgbm_to_onnx(model_path, output_path, model_name="model"):
    """
    Convert LightGBM model to ONNX format
    """
    print(f"Converting LightGBM model: {model_path}")
    
    # Load the model
    if model_path.endswith('.pkl'):
        model = joblib.load(model_path)
    else:
        model = lgb.Booster(model_file=model_path)
    
    # Number of features
    n_features = len(FEATURE_NAMES)
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    
    # Save the model
    onnx.save_model(onnx_model, output_path)
    print(f"✅ Saved ONNX model to: {output_path}")
    
    return True

def convert_sklearn_to_onnx(model_path, output_path, model_name="model"):
    """
    Convert sklearn model to ONNX format
    """
    print(f"Converting sklearn model: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Number of features
    n_features = len(FEATURE_NAMES)
    
    # Define input type
    initial_type = [('float_input', SklearnFloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save the model
    onnx.save_model(onnx_model, output_path)
    print(f"✅ Saved ONNX model to: {output_path}")
    
    return True

def validate_onnx_model(onnx_path):
    """
    Validate the converted ONNX model
    """
    try:
        # Load and check the model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Create sample input
        n_features = len(FEATURE_NAMES)
        sample_input = np.random.random((1, n_features)).astype(np.float32)
        
        print(f"✅ ONNX model validation successful: {onnx_path}")
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Features: {n_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        return False

def main():
    """
    Main conversion script
    """
    print("VPBank ML API - Model Converter")
    print("=" * 40)
    
    # Define model paths (adjust these paths as needed)
    model_configs = [
        {
            'name': 'QR Model',
            'input_path': 'models/qr_boosted_lgbm_optuna_timesplit.pkl',
            'output_path': 'models/qr_model.onnx',
            'type': 'lightgbm'
        },
        {
            'name': 'IBFT Model', 
            'input_path': 'models/ibft_boosted_lgbm_optuna_timesplit.pkl',
            'output_path': 'models/ibft_model.onnx',
            'type': 'lightgbm'
        },
        {
            'name': 'Top-up Model',
            'input_path': 'models/topup_boosted_lgbm_optuna_timesplit.pkl', 
            'output_path': 'models/topup_model.onnx',
            'type': 'lightgbm'
        }
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    success_count = 0
    
    for config in model_configs:
        print(f"\n🔄 Processing {config['name']}...")
        
        # Check if input model exists
        if not os.path.exists(config['input_path']):
            print(f"⚠️  Input model not found: {config['input_path']}")
            print(f"   Skipping {config['name']}")
            continue
        
        try:
            # Convert based on model type
            if config['type'] == 'lightgbm':
                success = convert_lightgbm_to_onnx(
                    config['input_path'], 
                    config['output_path'], 
                    config['name']
                )
            elif config['type'] == 'sklearn':
                success = convert_sklearn_to_onnx(
                    config['input_path'], 
                    config['output_path'], 
                    config['name']
                )
            else:
                print(f"❌ Unknown model type: {config['type']}")
                continue
            
            if success:
                # Validate the converted model
                validate_onnx_model(config['output_path'])
                success_count += 1
            
        except Exception as e:
            print(f"❌ Error converting {config['name']}: {e}")
            continue
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Successfully converted: {success_count}/{len(model_configs)} models")
    
    if success_count > 0:
        print(f"\n✅ Models ready for VPBank ML API!")
        print(f"   Place the .onnx files in the API's models/ directory")
        print(f"   Update environment variables as needed")
    else:
        print(f"\n⚠️  No models were converted. Please check input paths and model files.")

def create_sample_data():
    """
    Create sample data for testing
    """
    print("\n🔄 Creating sample test data...")
    
    # Create sample features
    sample_data = {
        'is_weekend': 0,
        'is_night': 1,
        'txn_type_idx': 2,  # QR
        'is_ibft': 0,
        'is_topup': 0,
        'is_qr': 1,
        'amount_usd': 1500.0,
        'amount_log': 7.313,
        'is_high_amount': 0,
        'name_is_ascii': 1,
        'name_has_digit': 0,
        'name_has_symbol': 0,
        'name_repeated_char': 0,
        'disposable_email_flag': 0,
        'billing_lat': 10.8231,
        'billing_long': 106.6297,
        'seconds_since_last_txn': 3600,
        'is_new_device': 1,
        'geo_distance_from_last_txn': 500.0,
        'suspicious_agent': 0,
        'is_business_hours': 0,
        'sum_amount_1h': 1500.0,
        'avg_amount_1h': 1500.0,
        'txn_count_1h': 1,
        'sum_amount_24h': 3000.0,
        'avg_amount_24h': 1500.0,
        'txn_count_24h': 2,
        'velocity_1h': 1,
        'rank_amount_per_day': 1.0,
        'change_in_user_agent': 1,
        'geo_speed_km_per_min': 8.33,
        'same_device_txn_1h': 0
    }
    
    # Create DataFrame
    df = pd.DataFrame([sample_data])
    
    # Save sample data
    df.to_csv('sample_transaction_features.csv', index=False)
    print(f"✅ Sample data saved to: sample_transaction_features.csv")
    
    # Display the data
    print(f"\nSample transaction features:")
    for feature, value in sample_data.items():
        print(f"  {feature}: {value}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        create_sample_data()
    else:
        main()
