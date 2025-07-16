const fraudService = require('../src/services/fraudDetectionService');

// Test all models: QR, IBFT, and TopUp
async function testAllModels() {
  console.log('=== Testing All Models (QR, IBFT, TopUp) ===');
  
  try {
    console.log('1. Initializing fraud detection service...');
    await fraudService.initialize();
    
    console.log('2. Checking model status...');
    const status = fraudService.getModelStatus();
    console.log('Model status:', JSON.stringify(status, null, 2));
    
    // Common feature set for testing
    const baseFeatures = {
      amount_usd: 5000,
      is_night: 0,
      is_weekend: 0,
      geo_distance_from_last_txn: 50,
      velocity_1h: 2,
      velocity_6h: 4,
      txn_count_1h: 2,
      txn_count_6h: 5,
      is_new_device: 0,
      suspicious_agent: 0,
      
      // Base transaction features
      amount_log: Math.log(5000),
      is_high_amount: 0,
      name_is_ascii: 1,
      name_has_digit: 0,
      name_has_symbol: 0,
      name_repeated_char: 0,
      disposable_email_flag: 0,
      billing_lat: 21.0285,
      billing_long: 105.8542,
      seconds_since_last_txn: 300,
      is_business_hours: 1,
      sum_amount_1h: 5000,
      avg_amount_1h: 5000,
      sum_amount_24h: 10000,
      avg_amount_24h: 2000,
      txn_count_24h: 5,
      rank_amount_per_day: 2.5,
      change_in_user_agent: 0,
      lat_shift: 0.1,
      lng_shift: 0.1,
      geo_speed_km_per_min: 10,
      same_device_txn_1h: 2
    };
    
    // Test QR model
    console.log('\n3. Testing QR model...');
    const qrFeatures = {
      ...baseFeatures,
      txn_type_idx: 2, // QR
      is_ibft: 0,
      is_topup: 0,
      is_qr: 1
    };
    
    const qrResult = await fraudService.predict(qrFeatures, 'qr');
    console.log('QR Result:', {
      fraudScore: qrResult.fraudScore,
      confidence: qrResult.confidence,
      modelVersion: qrResult.modelVersion
    });
    
    // Test IBFT model
    console.log('\n4. Testing IBFT model...');
    const ibftFeatures = {
      ...baseFeatures,
      txn_type_idx: 1, // IBFT
      is_ibft: 1,
      is_topup: 0,
      is_qr: 0
    };
    
    const ibftResult = await fraudService.predict(ibftFeatures, 'ibft');
    console.log('IBFT Result:', {
      fraudScore: ibftResult.fraudScore,
      confidence: ibftResult.confidence,
      modelVersion: ibftResult.modelVersion
    });
    
    // Test TopUp model
    console.log('\n5. Testing TopUp model...');
    const topupFeatures = {
      ...baseFeatures,
      txn_type_idx: 3, // TopUp
      is_ibft: 0,
      is_topup: 1,
      is_qr: 0
    };
    
    const topupResult = await fraudService.predict(topupFeatures, 'topup');
    console.log('TopUp Result:', {
      fraudScore: topupResult.fraudScore,
      confidence: topupResult.confidence,
      modelVersion: topupResult.modelVersion
    });
    
    console.log('\n✓ All models tested successfully!');
    
  } catch (error) {
    console.error('Error testing models:', error);
    console.error('Stack trace:', error.stack);
  }
}

testAllModels().catch(console.error);
