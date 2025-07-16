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

console.log('Testing Feature Mapper...\n');

// Test feature mapping
const features = FeatureMapper.mapKafkaToModelFeatures(sampleKafkaData);
console.log('Mapped features:', features);

// Test QR model preparation (34 features, int64)
console.log('\n=== QR Model (XGBoost) ===');
const qrFeatures = FeatureMapper.prepareONNXInput(features, 'qr', 'xgb');
console.log('QR Features length:', qrFeatures.length, '(expected: 34)');
console.log('QR Features type:', qrFeatures.constructor.name, '(expected: BigInt64Array)');
console.log('QR Features sample:', Array.from(qrFeatures.slice(0, 10)));

// Test IBFT model preparation (21 features, float32)
console.log('\n=== IBFT Model (Random Forest) ===');
const ibftFeatures = FeatureMapper.prepareONNXInput(features, 'ibft');
console.log('IBFT Features length:', ibftFeatures.length, '(expected: 21)');
console.log('IBFT Features type:', ibftFeatures.constructor.name, '(expected: Float32Array)');
console.log('IBFT Features sample:', Array.from(ibftFeatures.slice(0, 10)));

// Test TopUp model preparation (27 features, float32)
console.log('\n=== TopUp Model ===');
const topupFeatures = FeatureMapper.prepareONNXInput(features, 'topup');
console.log('TopUp Features length:', topupFeatures.length, '(expected: 27)');
console.log('TopUp Features type:', topupFeatures.constructor.name, '(expected: Float32Array)');
console.log('TopUp Features sample:', Array.from(topupFeatures.slice(0, 10)));

// Test feature validation
console.log('\n=== Feature Validation ===');
const validation = FeatureMapper.validateFeatures(features);
console.log('Validation result:', validation);

// Print detailed feature breakdown for QR model
console.log('\n=== QR Model Feature Details ===');
const qrFeatureOrder = [
  'is_weekend', 'is_night', 'txn_type_idx', 'is_ibft', 'is_topup', 'is_qr',
  'amount_usd', 'amount_log', 'is_high_amount',
  'name_is_ascii', 'name_has_digit', 'name_has_symbol', 'name_repeated_char',
  'disposable_email_flag', 'billing_lat', 'billing_long',
  'seconds_since_last_txn', 'is_new_device', 'geo_distance_from_last_txn',
  'suspicious_agent', 'is_business_hours', 'sum_amount_1h', 'avg_amount_1h',
  'txn_count_1h', 'sum_amount_24h', 'avg_amount_24h', 'txn_count_24h',
  'velocity_1h', 'rank_amount_per_day', 'change_in_user_agent',
  'lat_shift', 'lng_shift', 'geo_speed_km_per_min', 'same_device_txn_1h'
];

qrFeatureOrder.forEach((featureName, index) => {
  const value = features[featureName];
  const tensorValue = qrFeatures[index];
  console.log(`${index.toString().padStart(2)}: ${featureName.padEnd(25)} = ${value} -> ${tensorValue}`);
});
