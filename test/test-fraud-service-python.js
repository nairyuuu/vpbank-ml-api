const fraudService = require('../src/services/fraudDetectionService');
const FeatureMapper = require('../src/utils/featureMapper');

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

async function testFraudDetectionService() {
  console.log('Testing Fraud Detection Service with Python ONNX Support...\n');
  
  try {
    // Initialize the service
    console.log('=== Initializing Fraud Detection Service ===');
    await fraudService.initialize();
    console.log('✓ Service initialized successfully\n');
    
    // Test different transaction types
    const testCases = [
      { ...sampleKafkaData, transaction: { ...sampleKafkaData.transaction, txn_type: 'QR' } },
      { ...sampleKafkaData, transaction: { ...sampleKafkaData.transaction, txn_type: 'IBFT' } },
      { ...sampleKafkaData, transaction: { ...sampleKafkaData.transaction, txn_type: 'TOPUP' } }
    ];
    
    for (const testCase of testCases) {
      const txnType = testCase.transaction.txn_type;
      console.log(`=== Testing ${txnType} Transaction ===`);
      
      try {
        const result = await fraudService.detectFraud(testCase);
        
        console.log(`✓ ${txnType} prediction successful:`);
        console.log(`  Fraud Score: ${result.fraudScore}`);
        console.log(`  Is Fraud: ${result.isFraud}`);
        console.log(`  Confidence: ${result.confidence}`);
        console.log(`  Model Version: ${result.modelVersion}`);
        console.log(`  Processing Time: ${result.processingTime}ms`);
        
        if (result.recommendation) {
          console.log(`  Recommendation: ${result.recommendation.action}`);
        }
        
      } catch (error) {
        console.log(`✗ ${txnType} prediction failed:`, error.message);
      }
      
      console.log('');
    }
    
    console.log('=== Testing Individual Model Predictions ===');
    
    // Test individual model predictions
    const features = FeatureMapper.mapKafkaToModelFeatures(sampleKafkaData);
    
    // Test QR model (Node.js ONNX)
    try {
      console.log('Testing QR Model (Node.js ONNX)...');
      const qrModelConfig = fraudService.models.get('qr');
      const qrResult = await fraudService.predictQR(features, qrModelConfig);
      console.log('✓ QR Model Result:', qrResult);
    } catch (error) {
      console.log('✗ QR Model Error:', error.message);
    }
    
    // Test IBFT model (Python ONNX)
    try {
      console.log('Testing IBFT Model (Python ONNX)...');
      const ibftModelConfig = fraudService.models.get('ibft');
      const ibftResult = await fraudService.predictIBFT(features, ibftModelConfig);
      console.log('✓ IBFT Model Result:', ibftResult);
    } catch (error) {
      console.log('✗ IBFT Model Error:', error.message);
    }
    
    // Test TopUp model (Python ONNX)
    try {
      console.log('Testing TopUp Model (Python ONNX)...');
      const topupModelConfig = fraudService.models.get('topup');
      const topupResult = await fraudService.predictTopup(features, topupModelConfig);
      console.log('✓ TopUp Model Result:', topupResult);
    } catch (error) {
      console.log('✗ TopUp Model Error:', error.message);
    }
    
  } catch (error) {
    console.error('Test failed:', error);
  }
}

// Run the test
testFraudDetectionService().catch(console.error);
