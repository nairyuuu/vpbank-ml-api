const fraudService = require('../src/services/fraudDetectionService');

// Test the simplified QR system with only XGBoost
async function testSimplifiedQRSystem() {
  console.log('=== Testing Simplified QR System (XGBoost Only) ===');
  
  try {
    console.log('1. Initializing fraud detection service...');
    await fraudService.initialize();
    
    console.log('2. Checking model status...');
    const status = fraudService.getModelStatus();
    console.log('Model status:', JSON.stringify(status, null, 2));
    
    // Test high-risk transaction
    console.log('\n3. Testing high-risk transaction...');
    const highRiskFeatures = {
      amount_usd: 25000,
      is_night: 1,
      is_weekend: 0,
      geo_distance_from_last_txn: 250,
      velocity_1h: 7,
      velocity_6h: 15,
      txn_count_1h: 3,
      txn_count_6h: 8,
      is_new_device: 1,
      suspicious_agent: 0,
      avg_amount_last_10_txns: 5000,
      
      // Additional required features
      txn_type_idx: 2, // QR
      is_ibft: 0,
      is_topup: 0,
      is_qr: 1,
      amount_log: Math.log(25000),
      is_high_amount: 1,
      name_is_ascii: 1,
      name_has_digit: 0,
      name_has_symbol: 0,
      name_repeated_char: 0,
      disposable_email_flag: 0,
      billing_lat: 21.0285,
      billing_long: 105.8542,
      seconds_since_last_txn: 300,
      is_business_hours: 0,
      sum_amount_1h: 25000,
      avg_amount_1h: 25000,
      sum_amount_24h: 30000,
      avg_amount_24h: 5000,
      txn_count_24h: 6,
      rank_amount_per_day: 5.0,
      change_in_user_agent: 1,
      lat_shift: 0.1,
      lng_shift: 0.1,
      geo_speed_km_per_min: 50,
      same_device_txn_1h: 1
    };
    
    const result = await fraudService.predict(highRiskFeatures, 'qr');
    console.log('\n=== High-Risk QR Prediction Results ===');
    console.log('Fraud score:', result.fraudScore);
    console.log('Confidence:', result.confidence);
    console.log('Model version:', result.modelVersion);
    console.log('Details:', JSON.stringify(result.details, null, 2));
    
    // Test low-risk transaction
    console.log('\n4. Testing low-risk transaction...');
    const lowRiskFeatures = {
      ...highRiskFeatures,
      amount_usd: 100,
      amount_log: Math.log(100),
      is_night: 0,
      is_high_amount: 0,
      geo_distance_from_last_txn: 5,
      velocity_1h: 1,
      velocity_6h: 2,
      txn_count_1h: 1,
      txn_count_6h: 2,
      is_new_device: 0,
      suspicious_agent: 0,
      sum_amount_1h: 100,
      avg_amount_1h: 100,
      rank_amount_per_day: 0.5,
      change_in_user_agent: 0,
      geo_speed_km_per_min: 1,
      same_device_txn_1h: 5
    };
    
    const lowResult = await fraudService.predict(lowRiskFeatures, 'qr');
    console.log('\n=== Low-Risk QR Prediction Results ===');
    console.log('Fraud score:', lowResult.fraudScore);
    console.log('Confidence:', lowResult.confidence);
    console.log('Model version:', lowResult.modelVersion);
    console.log('Details:', JSON.stringify(lowResult.details, null, 2));
    
    // Test prediction consistency
    console.log('\n5. Testing prediction consistency...');
    const predictions = [];
    for (let i = 0; i < 3; i++) {
      const pred = await fraudService.predict(highRiskFeatures, 'qr');
      predictions.push(pred.fraudScore);
      console.log(`Prediction ${i+1}: ${pred.fraudScore.toFixed(4)}`);
    }
    
    const avgScore = predictions.reduce((a, b) => a + b, 0) / predictions.length;
    const variance = predictions.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) / predictions.length;
    console.log(`Average score: ${avgScore.toFixed(4)}, Variance: ${variance.toFixed(6)}`);
    
    console.log('\n✓ Simplified QR system test completed successfully!');
    
  } catch (error) {
    console.error('Error testing simplified QR system:', error);
    console.error('Stack trace:', error.stack);
  }
}

testSimplifiedQRSystem().catch(console.error);
