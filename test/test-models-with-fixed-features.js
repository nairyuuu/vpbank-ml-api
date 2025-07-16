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

async function testModel(modelPath, modelType, features) {
  console.log(`\n=== Testing ${modelType.toUpperCase()} Model ===`);
  
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
    
    // Create tensor
    const inputName = session.inputNames[0];
    const tensor = new onnx.Tensor(
      inputFeatures.constructor.name === 'BigInt64Array' ? 'int64' : 'float32',
      inputFeatures,
      [1, inputFeatures.length]
    );
    
    // Run inference
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);
    
    const outputName = session.outputNames[0];
    const prediction = results[outputName];
    
    console.log('✓ Inference successful');
    console.log('Prediction shape:', prediction.dims);
    console.log('Prediction data:', prediction.data);
    
    // Clean up
    await session.release();
    
    return {
      success: true,
      prediction: prediction.data,
      features: inputFeatures.length
    };
    
  } catch (error) {
    console.log(`✗ Error testing ${modelType} model:`, error.message);
    return {
      success: false,
      error: error.message,
      features: 0
    };
  }
}

async function runTests() {
  console.log('Testing Updated Feature Mapper with All Models...\n');
  
  // Map features from Kafka data
  const features = FeatureMapper.mapKafkaToModelFeatures(sampleKafkaData);
  
  const modelPaths = {
    qr: path.join(__dirname, '..', 'models', 'xgb_qr.onnx'),
    ibft: path.join(__dirname, '..', 'models', 'model_ibft.onnx'),
    topup: path.join(__dirname, '..', 'models', 'topup_model.onnx')
  };
  
  const results = {};
  
  // Test each model
  for (const [modelType, modelPath] of Object.entries(modelPaths)) {
    results[modelType] = await testModel(modelPath, modelType, features);
  }
  
  // Summary
  console.log('\n=== Test Summary ===');
  console.log('QR Model (XGBoost):', results.qr.success ? '✓ PASSED' : '✗ FAILED');
  console.log('IBFT Model (RF):', results.ibft.success ? '✓ PASSED' : '✗ FAILED');
  console.log('TopUp Model:', results.topup.success ? '✓ PASSED' : '✗ FAILED');
  
  console.log('\n=== Feature Counts ===');
  console.log('QR Model features:', results.qr.features, '(expected: 34)');
  console.log('IBFT Model features:', results.ibft.features, '(expected: 21)');
  console.log('TopUp Model features:', results.topup.features, '(expected: 27)');
  
  if (results.qr.success) {
    console.log('\n=== QR Model Working ===');
    console.log('XGBoost QR prediction:', results.qr.prediction);
  }
  
  if (!results.ibft.success) {
    console.log('\n=== IBFT Model Error ===');
    console.log('Error:', results.ibft.error);
  }
  
  if (!results.topup.success) {
    console.log('\n=== TopUp Model Error ===');
    console.log('Error:', results.topup.error);
  }
}

runTests().catch(console.error);
