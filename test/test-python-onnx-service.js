const PythonONNXService = require('../src/services/pythonONNXService');
const FeatureMapper = require('../src/utils/featureMapper');

// Test data
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

async function testPythonONNXService() {
  console.log('Testing Python ONNX Service Directly...\n');
  
  const pythonService = new PythonONNXService();
  const features = FeatureMapper.mapKafkaToModelFeatures(sampleKafkaData);
  
  const models = ['qr', 'ibft', 'topup'];
  
  for (const modelType of models) {
    console.log(`=== Testing ${modelType.toUpperCase()} Model ===`);
    
    try {
      const inputArray = FeatureMapper.prepareONNXInput(features, modelType);
      const inputFeatures = Array.from(inputArray);
      
      console.log(`Input features length: ${inputFeatures.length}`);
      console.log(`Input features type: ${inputArray.constructor.name}`);
      
      const result = await pythonService.runInference(modelType, inputFeatures);
      
      if (result.success) {
        console.log('✓ Success!');
        console.log('Predictions:', result.predictions);
        console.log('Input shape:', result.input_shape);
        console.log('Input type:', result.input_type);
        
        // Extract fraud score
        if (result.predictions && result.predictions.length > 1) {
          const probabilities = result.predictions[1];
          if (Array.isArray(probabilities) && probabilities.length > 1) {
            console.log('Fraud Score (array):', probabilities[1]);
          } else if (typeof probabilities === 'object' && probabilities['1']) {
            console.log('Fraud Score (object):', probabilities['1']);
          }
        }
      } else {
        console.log('✗ Failed:', result.error);
      }
      
    } catch (error) {
      console.log('✗ Error:', error.message);
    }
    
    console.log('');
  }
}

testPythonONNXService().catch(console.error);
