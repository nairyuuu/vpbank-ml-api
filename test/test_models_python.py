#!/usr/bin/env python3
"""
Python script to test ONNX models that don't work with onnxruntime-node
This will help us understand if the models work with the Python ONNX runtime
"""

import numpy as np
import onnxruntime as ort
import sys
import json
import os

def test_model(model_path, features, model_type):
    """Test an ONNX model with given features"""
    try:
        # Load the model
        session = ort.InferenceSession(model_path)
        
        print(f"✓ Model loaded successfully: {model_path}")
        print(f"  Input names: {session.get_inputs()[0].name}")
        print(f"  Input shape: {session.get_inputs()[0].shape}")
        print(f"  Input type: {session.get_inputs()[0].type}")
        print(f"  Output names: {[output.name for output in session.get_outputs()]}")
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        
        # Convert features to numpy array
        if model_type == 'qr':
            # XGBoost QR model expects int64
            input_data = np.array([features], dtype=np.int64)
        else:
            # IBFT and TopUp models expect float32
            input_data = np.array([features], dtype=np.float32)
        
        print(f"  Input data shape: {input_data.shape}")
        print(f"  Input data type: {input_data.dtype}")
        
        # Run inference
        result = session.run(None, {input_name: input_data})
        
        print(f"✓ Inference successful!")
        for i, output in enumerate(session.get_outputs()):
            print(f"  Output {i} ({output.name}): {result[i]}")
        
        return {
            'success': True,
            'predictions': [r.tolist() if hasattr(r, 'tolist') else r for r in result],
            'input_shape': input_data.shape,
            'input_type': str(input_data.dtype)
        }
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Test features for each model type
    test_cases = {
        'qr': {
            'model_path': 'models/xgb_qr.onnx',
            'features': [0, 0, 2, 0, 0, 1, 40, 14, 1, 1, 0, 0, 0, 0, 11, 107, 3600, 0, 3, 0, 1, 1000000, 1000000, 1, 2000000, 1000000, 2, 1, 1, 0, 28, 198, 0, 1]
        },
        'ibft': {
            'model_path': 'models/model_ibft.onnx',
            'features': [0, 0, 2, 0, 0, 1, 40.0, 13.8155, 1, 1, 0, 0, 0, 0, 10.8231, 106.6297, 3600.0, 0, 2.5, 0, 1, 1000000.0, 1000000.0, 1, 2000000.0, 1000000.0, 2]
        },
        'topup': {
            'model_path': 'models/topup_model.onnx',
            'features': [0, 0, 2, 0, 0, 1, 40.0, 13.8155, 1, 1, 0, 0, 0, 0, 10.8231, 106.6297, 3600.0, 0, 2.5, 0, 1, 1000000.0, 1000000.0, 1, 2000000.0, 1000000.0, 2]
        }
    }
    
    print("Testing ONNX Models with Python Runtime...")
    print("=" * 50)
    
    results = {}
    
    for model_type, config in test_cases.items():
        print(f"\n=== Testing {model_type.upper()} Model ===")
        
        model_path = config['model_path']
        features = config['features']
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            results[model_type] = {'success': False, 'error': 'Model file not found'}
            continue
        
        results[model_type] = test_model(model_path, features, model_type)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for model_type, result in results.items():
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        print(f"{model_type.upper()} Model: {status}")
        if not result['success']:
            print(f"  Error: {result['error']}")
    
    # Output results as JSON for Node.js consumption
    print("\n" + "=" * 50)
    print("JSON Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
