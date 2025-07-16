# Cross-Platform Setup Guide

## VPBank ML API - Python ONNX Service Setup

This guide covers setting up the Python ONNX service for both Windows and Linux/WSL environments.

### Prerequisites

- Node.js (v16 or higher)
- Python 3.8+ 
- Git

### Setup Instructions

#### 1. Clone and Install Node.js Dependencies

```bash
git clone <repository-url>
cd vpbank-ml-api
npm install
```

#### 2. Set Up Python Environment

**For Linux/WSL:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install numpy onnxruntime
```

**For Windows:**
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install Python dependencies
pip install numpy onnxruntime
```

#### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# Model paths
QR_XGB_MODEL_PATH=./models/xgb_qr.onnx
IBFT_MODEL_PATH=./models/model_ibft.onnx
TOPUP_MODEL_PATH=./models/topup_model.onnx

# Fraud thresholds
QR_FRAUD_THRESHOLD=0.5
IBFT_FRAUD_THRESHOLD=0.5
TOPUP_FRAUD_THRESHOLD=0.5

# Decision system (optional)
LAMBDA_DECISION_SYSTEM_URL=https://your-lambda-url.amazonaws.com
LAMBDA_DECISION_SYSTEM_API_KEY=your-api-key
```

#### 4. Place Model Files

Ensure your ONNX model files are in the `models/` directory:
- `models/xgb_qr.onnx` (QR payments model)
- `models/model_ibft.onnx` (IBFT model)
- `models/topup_model.onnx` (TopUp model)

#### 5. Test the Setup

**Test cross-platform Python detection:**
```bash
node test/test-python-cross-platform.js
```

**Test feature mapping:**
```bash
node test/test-feature-mapper.js
```

**Test complete fraud detection service:**
```bash
node test/test-fraud-service-python.js
```

**Test Python ONNX service directly:**
```bash
node test/test-python-onnx-service.js
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection Service                 │
├─────────────────────────────────────────────────────────────┤
│  QR Model (XGBoost)     │  IBFT Model (RF)  │  TopUp Model  │
│  ✅ Node.js ONNX        │  ✅ Python ONNX   │  ✅ Python ONNX│
│  34 features, int64     │  27 features, f32  │  27 features,f32│
│  Windows/Linux          │  Cross-platform    │  Cross-platform │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Platform Features

- **Automatic Python Detection**: Detects Python executable across Windows/Linux/WSL
- **Virtual Environment Support**: Uses `.venv` when available, falls back to system Python
- **Path Resolution**: Handles Windows (`Scripts/python.exe`) and Linux (`bin/python`) paths
- **Robust Error Handling**: Graceful fallbacks for missing dependencies

### API Usage

```javascript
const fraudService = require('./src/services/fraudDetectionService');

// Initialize service
await fraudService.initialize();

// Detect fraud
const result = await fraudService.detectFraud(transactionData);
console.log('Fraud score:', result.fraudScore);
```

### Troubleshooting

**Python not found:**
- Ensure Python 3.8+ is installed
- Create virtual environment: `python3 -m venv .venv`
- Activate and install dependencies

**Model loading errors:**
- Check model files exist in `models/` directory
- Verify ONNX model compatibility with onnxruntime

**Permission errors (Linux/WSL):**
- Make sure virtual environment has correct permissions
- Use `chmod +x .venv/bin/python` if needed

### Performance

- **QR Model**: ~5-10ms (Node.js ONNX)
- **IBFT Model**: ~500-800ms (Python ONNX)
- **TopUp Model**: ~500-800ms (Python ONNX)

The hybrid approach provides the best balance of performance and compatibility.
