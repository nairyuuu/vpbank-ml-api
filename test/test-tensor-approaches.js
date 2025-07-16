const FeatureMapper = require('../src/utils/featureMapper');
const onnx = require('onnxruntime-node');
const path = require('path');

// Test sample transaction data
const sampleKafkaData = {
  transaction: {
    txn_type: 'QR',
    amount_raw: 1000000,
    amount_log: 13.8155,
    currency: 'VND'
  },
  temporal: {
    is_weekend: false,
    hour: 14,
    is_business_hours: true
  },
  location: {
    billing_lat: 10.8231,
    billing_long: 106.6297,
    geo_distance_from_last_txn: 2.5
  },
  aggregation: {
    user_amount_sum_1h: 1000000,
    user_amount_avg_1h: 1000000,
    user_txn_count_1h: 1,
    user_amount_sum_24h: 2000000,
    user_amount_avg_24h: 1000000,
    user_txn_count_24h: 2,
    seconds_since_last_txn: 3600,
    device_txn_count_1h: 1,
    user_unique_merchants_1h: 1,
    user_unique_merchants_24h: 2,
    user_unique_channels_24h: 1,
    merchant_txn_count_1h: 10,
    merchant_amount_sum_1h: 5000000,
    merchant_unique_users_1h: 8,
    ip_txn_count_1h: 1,
    is_velocity_anomaly: false,
    is_amount_anomaly: false,
    is_new_merchant: false,
    is_new_location: false,
    is_rapid_fire: false
  },
  behavioral: {
    name_is_ascii: true,
    name_has_digit: false,
    name_has_symbol: false,
    name_repeated_char: false,
    is_disposable_email: false,
    is_new_device: false,
    email_domain: 'gmail.com',
    user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    device_fingerprint: 'abc123'
  }
};

async function testModelWithDifferentTensorTypes(modelPath, modelType, features) {
  console.log(`\n=== Testing ${modelType.toUpperCase()} Model with Different Tensor Types ===`);
  
  try {
    // Load the model
    const session = await onnx.InferenceSession.create(modelPath);
    console.log(`✓ Model loaded successfully: ${modelPath}`);
    
    // Print model info
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    
    // Prepare features
    const inputFeatures = FeatureMapper.prepareONNXInput(features, modelType);
    console.log(`✓ Features prepared: ${inputFeatures.length} features, type: ${inputFeatures.constructor.name}`);
    
    const inputName = session.inputNames[0];
    
    // Try different tensor creation approaches
    const approaches = [
      {
        name: 'Original approach',
        createTensor: () => {
          const tensorType = inputFeatures.constructor.name === 'BigInt64Array' ? 'int64' : 'float32';
          return new onnx.Tensor(tensorType, inputFeatures, [1, inputFeatures.length]);
        }
      },
      {
        name: 'Force float32',
        createTensor: () => {
          const floatArray = new Float32Array(inputFeatures.length);
          for (let i = 0; i < inputFeatures.length; i++) {
            floatArray[i] = Number(inputFeatures[i]);
          }
          return new onnx.Tensor('float32', floatArray, [1, inputFeatures.length]);
        }
      },
      {
        name: 'Force double',
        createTensor: () => {
          const doubleArray = new Float64Array(inputFeatures.length);
          for (let i = 0; i < inputFeatures.length; i++) {
            doubleArray[i] = Number(inputFeatures[i]);
          }
          return new onnx.Tensor('float64', doubleArray, [1, inputFeatures.length]);
        }
      },
      {
        name: 'Regular Array to float32',
        createTensor: () => {
          const regularArray = Array.from(inputFeatures).map(v => Number(v));
          return new onnx.Tensor('float32', regularArray, [1, inputFeatures.length]);
        }
      }
    ];
    
    for (const approach of approaches) {
      try {
        console.log(`\n  Trying: ${approach.name}...`);
        const tensor = approach.createTensor();
        console.log(`    Tensor created: ${tensor.type}, shape: [${tensor.dims.join(', ')}]`);
        
        const feeds = { [inputName]: tensor };
        const results = await session.run(feeds);
        
        const outputName = session.outputNames[0];
        const prediction = results[outputName];
        
        console.log(`    ✓ SUCCESS with ${approach.name}!`);
        console.log('    Prediction shape:', prediction.dims);
        console.log('    Prediction data:', prediction.data);
        
        // Clean up
        await session.release();
        
        return {
          success: true,
          approach: approach.name,
          prediction: prediction.data,
          features: inputFeatures.length
        };
        
      } catch (error) {
        console.log(`    ✗ Failed with ${approach.name}:`, error.message);
      }
    }
    
    // If we get here, all approaches failed
    await session.release();
    return {
      success: false,
      error: 'All tensor creation approaches failed',
      features: inputFeatures.length
    };
    
  } catch (error) {
    console.log(`✗ Error loading ${modelType} model:`, error.message);
    return {
      success: false,
      error: error.message,
      features: 0
    };
  }
}

async function runAdvancedTests() {
  console.log('Testing Models with Different Tensor Type Approaches...\n');
  
  // Map features from Kafka data
  const features = FeatureMapper.mapKafkaToModelFeatures(sampleKafkaData);
  
  const modelPaths = {
    qr: path.join(__dirname, '..', 'models', 'xgb_qr.onnx'),
    ibft: path.join(__dirname, '..', 'models', 'model_ibft.onnx'),
    topup: path.join(__dirname, '..', 'models', 'topup_model.onnx')
  };
  
  const results = {};
  
  // Test each model with different approaches
  for (const [modelType, modelPath] of Object.entries(modelPaths)) {
    results[modelType] = await testModelWithDifferentTensorTypes(modelPath, modelType, features);
  }
  
  // Summary
  console.log('\n=== Final Test Summary ===');
  console.log('QR Model (XGBoost):', results.qr.success ? `✓ PASSED (${results.qr.approach})` : '✗ FAILED');
  console.log('IBFT Model (RF):', results.ibft.success ? `✓ PASSED (${results.ibft.approach})` : '✗ FAILED');
  console.log('TopUp Model:', results.topup.success ? `✓ PASSED (${results.topup.approach})` : '✗ FAILED');
  
  if (results.qr.success) {
    console.log('\n=== QR Model Results ===');
    console.log('Working approach:', results.qr.approach);
    console.log('Prediction:', results.qr.prediction);
  }
  
  if (results.ibft.success) {
    console.log('\n=== IBFT Model Results ===');
    console.log('Working approach:', results.ibft.approach);
    console.log('Prediction:', results.ibft.prediction);
  }
  
  if (results.topup.success) {
    console.log('\n=== TopUp Model Results ===');
    console.log('Working approach:', results.topup.approach);
    console.log('Prediction:', results.topup.prediction);
  }
}

runAdvancedTests().catch(console.error);
