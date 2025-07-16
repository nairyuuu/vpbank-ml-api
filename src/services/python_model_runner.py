#!/usr/bin/env python3
"""
Python ONNX model runner for Node.js service
This script runs ONNX models that don't work with onnxruntime-node
"""

import numpy as np
import onnxruntime as ort
import sys
import json
import os

def run_model(model_type, features):
    """Run inference on specified model with features"""
    
    # Model paths
    model_paths = {
        'qr': 'models/xgb_qr.onnx',
        'ibft': 'models/model_ibft.onnx',
        'topup': 'models/topup_model.onnx'
    }
    
    if model_type not in model_paths:
        return {
            'success': False,
            'error': f'Unknown model type: {model_type}'
        }
    
    model_path = model_paths[model_type]
    
    if not os.path.exists(model_path):
        return {
            'success': False,
            'error': f'Model file not found: {model_path}'
        }
    
    try:
        # Load the model
        session = ort.InferenceSession(model_path)
        
        # Get input info
        input_name = session.get_inputs()[0].name
        
        # Prepare input data
        if model_type == 'qr':
            # XGBoost QR model expects int64
            input_data = np.array([features], dtype=np.int64)
        else:
            # IBFT and TopUp models expect float32
            input_data = np.array([features], dtype=np.float32)
        
        # Run inference
        result = session.run(None, {input_name: input_data})
        
        # Process results
        predictions = []
        for r in result:
            if hasattr(r, 'tolist'):
                predictions.append(r.tolist())
            else:
                predictions.append(r)
        
        return {
            'success': True,
            'predictions': predictions,
            'model_type': model_type,
            'input_shape': input_data.shape,
            'input_type': str(input_data.dtype)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python python_model_runner.py <model_type> <features_json>'
        }))
        sys.exit(1)
    
    model_type = sys.argv[1]
    features_json = sys.argv[2]
    
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError as e:
        print(json.dumps({
            'success': False,
            'error': f'Invalid JSON features: {str(e)}'
        }))
        sys.exit(1)
    
    result = run_model(model_type, features)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
