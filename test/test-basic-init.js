const fraudService = require('../src/services/fraudDetectionService');

async function testBasicInitialization() {
  try {
    console.log('Testing basic initialization...');
    await fraudService.initialize();
    console.log('✓ Initialization successful');
    
    // Test a simple prediction
    const features = {
      amount_usd: 1000,
      is_night: 0,
      is_weekend: 0,
      geo_distance_from_last_txn: 10,
      velocity_1h: 1,
      velocity_6h: 2,
      txn_count_1h: 1,
      txn_count_6h: 2,
      is_new_device: 0,
      suspicious_agent: 0,
      
      // Additional features
      txn_type_idx: 2,
      is_ibft: 0,
      is_topup: 0,
      is_qr: 1,
      amount_log: Math.log(1000),
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
      sum_amount_1h: 1000,
      avg_amount_1h: 1000,
      sum_amount_24h: 2000,
      avg_amount_24h: 500,
      txn_count_24h: 4,
      rank_amount_per_day: 2.0,
      change_in_user_agent: 0,
      lat_shift: 0.01,
      lng_shift: 0.01,
      geo_speed_km_per_min: 2,
      same_device_txn_1h: 3
    };
    
    console.log('Testing QR prediction...');
    const result = await fraudService.predict(features, 'qr');
    console.log('✓ QR prediction successful:', result.fraudScore);
    
  } catch (error) {
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
  }
}

testBasicInitialization().catch(console.error);
