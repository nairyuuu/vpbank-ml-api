const axios = require('axios');

const API_BASE_URL = 'http://localhost:3000';

// Sample transaction data that matches the Kafka topic format
const sampleTransaction = {
  "transaction_id": "test-65a6cbd5-e478-41be-ad54-26f62249bf42",
  "user_id": "test-44ad0106-aa4f-4112-b92c-aac8ad5806a4",
  "merchant_id": "test-1c0c2de5-d426-4efd-aaef-07218cc33827",
  "timestamp": "2025-07-14T10:30:00Z",
  "transaction": {
    "amount_raw": 1500.50,
    "amount_log": 7.313,
    "amount_range": "medium",
    "is_round_amount": false,
    "amount_digits": 4,
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
    "currency": "USD",
    "is_foreign_currency": false,
    "description_length": 20,
    "description_words": 4,
    "has_description": true
  },
  "temporal": {
    "hour": 22,
    "minute": 30,
    "day_of_week": 0,
    "day_of_month": 14,
    "month": 7,
    "year": 2025,
    "quarter": 3,
    "week_of_year": 29,
    "time_of_day": "night",
    "is_business_hours": false,
    "is_banking_hours": false,
    "is_weekend": false,
    "is_weekday": true,
    "is_morning_peak": false,
    "is_evening_peak": false,
    "is_lunch_time": false,
    "is_late_night": true,
    "is_very_early": false,
    "hour_sin": -0.5,
    "hour_cos": 0.866,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "month_sin": 0.5,
    "month_cos": 0.866,
    "minutes_since_midnight": 1350,
    "days_since_epoch": 20213
  },
  "location": {
    "billing_lat": 10.8231,
    "billing_long": 106.6297,
    "geo_distance_from_last_txn": 500.0
  },
  "aggregation": {
    "user_txn_count_1h": 3,
    "user_amount_sum_1h": 4500.0,
    "user_amount_avg_1h": 1500.0,
    "user_unique_merchants_1h": 2,
    "user_txn_count_24h": 5,
    "user_amount_sum_24h": 7500.0,
    "user_amount_avg_24h": 1500.0,
    "user_unique_merchants_24h": 3,
    "user_unique_channels_24h": 1,
    "merchant_txn_count_1h": 10,
    "merchant_amount_sum_1h": 15000.0,
    "merchant_unique_users_1h": 8,
    "device_txn_count_1h": 3,
    "ip_txn_count_1h": 3,
    "seconds_since_last_txn": 1800,
    "is_velocity_anomaly": true,
    "is_amount_anomaly": false,
    "is_new_merchant": false,
    "is_new_location": true,
    "is_rapid_fire": false
  },
  "behavioral": {
    "entity_id_hash": "test-hash-entity",
    "user_id_hash": "test-hash-user",
    "merchant_id_hash": "test-hash-merchant",
    "device_id_hash": "test-hash-device",
    "ip_address_hash": "test-hash-ip",
    "is_new_user": false,
    "is_new_device": true,
    "is_new_ip": false,
    "email_domain": "gmail.com",
    "is_disposable_email": false,
    "has_common_email_domain": true,
    "name_is_ascii": true,
    "name_has_digit": false,
    "name_has_symbol": false,
    "name_repeated_char": false
  },
  "feature_extraction_timestamp": "2025-07-14T10:30:00Z",
  "pipeline_version": "1.0.0"
};

async function testHealthEndpoints() {
  console.log('🔍 Testing Health Endpoints...');
  
  try {
    // Test basic health
    const healthResponse = await axios.get(`${API_BASE_URL}/health`);
    console.log('✅ Health check:', healthResponse.data.status);
    
    // Test readiness
    const readyResponse = await axios.get(`${API_BASE_URL}/health/ready`);
    console.log('✅ Readiness check:', readyResponse.data.status);
    
    // Test liveness
    const liveResponse = await axios.get(`${API_BASE_URL}/health/live`);
    console.log('✅ Liveness check:', liveResponse.data.status);
    
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
  }
}

async function testModelStatus() {
  console.log('\n🔍 Testing Model Status...');
  
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/status`);
    console.log('✅ Model status retrieved');
    console.log('   Initialized:', response.data.models.initialized);
    
    if (response.data.models.models) {
      Object.entries(response.data.models.models).forEach(([modelType, status]) => {
        console.log(`   ${modelType.toUpperCase()}: ${status.loaded ? 'Loaded' : 'Not loaded'} (threshold: ${status.threshold})`);
      });
    }
    
  } catch (error) {
    console.error('❌ Model status check failed:', error.message);
  }
}

async function testFraudPrediction() {
  console.log('\n🔍 Testing Fraud Prediction...');
  
  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict`, sampleTransaction, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    console.log('✅ Fraud prediction successful');
    console.log('   Transaction ID:', response.data.result.transactionId);
    console.log('   Fraud Score:', response.data.result.fraudScore);
    console.log('   Is Fraud:', response.data.result.isFraud);
    console.log('   Model Type:', response.data.result.modelType);
    console.log('   Processing Time:', response.data.result.processingTimeMs, 'ms');
    
  } catch (error) {
    console.error('❌ Fraud prediction failed:', error.response?.data || error.message);
  }
}

async function testServicesStatus() {
  console.log('\n🔍 Testing Services Status...');
  
  try {
    const response = await axios.get(`${API_BASE_URL}/api/services/status`);
    console.log('✅ Services status retrieved');
    
    const services = response.data.services;
    console.log('   Fraud Detection:', services.fraudDetection.initialized ? 'Ready' : 'Not ready');
    console.log('   Alerts:', services.alerts.enabled ? 'Enabled' : 'Disabled');
    console.log('   Decision System:', services.decisionSystem.enabled ? 'Enabled' : 'Disabled');
    
  } catch (error) {
    console.error('❌ Services status check failed:', error.message);
  }
}

async function testAlert() {
  console.log('\n🔍 Testing Alert System...');
  
  try {
    const response = await axios.post(`${API_BASE_URL}/api/test/alert`);
    console.log('✅ Test alert sent successfully');
    console.log('   Test Alert ID:', response.data.testAlert.transactionId);
    
  } catch (error) {
    console.error('❌ Alert test failed:', error.response?.data || error.message);
  }
}

async function runAllTests() {
  console.log('🚀 VPBank ML API - Test Suite');
  console.log('=' * 50);
  
  const startTime = Date.now();
  
  await testHealthEndpoints();
  await testModelStatus();
  await testFraudPrediction();
  await testServicesStatus();
  await testAlert();
  
  const totalTime = Date.now() - startTime;
  console.log(`\n⏱️  Total test time: ${totalTime}ms`);
  console.log('✅ Test suite completed');
}

// Handle command line arguments
const args = process.argv.slice(2);

if (args.length === 0) {
  runAllTests();
} else {
  const testType = args[0];
  
  switch (testType) {
    case 'health':
      testHealthEndpoints();
      break;
    case 'models':
      testModelStatus();
      break;
    case 'predict':
      testFraudPrediction();
      break;
    case 'services':
      testServicesStatus();
      break;
    case 'alert':
      testAlert();
      break;
    default:
      console.log('Usage: node test.js [health|models|predict|services|alert]');
      console.log('Run without arguments to execute all tests');
  }
}

module.exports = {
  testHealthEndpoints,
  testModelStatus,
  testFraudPrediction,
  testServicesStatus,
  testAlert,
  sampleTransaction
};
