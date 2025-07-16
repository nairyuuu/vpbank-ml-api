/**
 * Feature mapper to convert Kafka enriched features to model input format
 * Maps from the Kafka topic structure to the training CSV format
 */

class FeatureMapper {
  /**
   * Maps Kafka enriched features to model input format
   * @param {Object} kafkaData - Data from vpbank-enriched-features topic
   * @returns {Object} - Features in model training format
   */
  static mapKafkaToModelFeatures(kafkaData) {
    const {
      transaction,
      temporal,
      location,
      aggregation,
      behavioral
    } = kafkaData;

    // Base transaction features
    const features = {
      // Weekend and night detection
      is_weekend: temporal.is_weekend ? 1 : 0,
      is_night: this.isNightTime(temporal.hour),
      
      // Transaction type mapping
      txn_type_idx: this.getTxnTypeIndex(transaction.txn_type),
      is_ibft: transaction.txn_type === 'IBFT' ? 1 : 0,
      is_topup: transaction.txn_type === 'TOPUP' ? 1 : 0,
      is_qr: transaction.txn_type === 'QR' ? 1 : 0,
      
      // Amount features
      amount_usd: this.convertToUSD(transaction.amount_raw, transaction.currency),
      amount_log: transaction.amount_log,
      is_high_amount: this.isHighAmount(transaction.amount_raw),
      
      // Behavioral features
      name_is_ascii: behavioral.name_is_ascii ? 1 : 0,
      name_has_digit: behavioral.name_has_digit ? 1 : 0,
      name_has_symbol: behavioral.name_has_symbol ? 1 : 0,
      name_repeated_char: behavioral.name_repeated_char ? 1 : 0,
      
      // Email features
      email_domain: behavioral.email_domain || '',
      disposable_email_flag: behavioral.is_disposable_email ? 1 : 0,
      
      // Location features
      billing_lat: location.billing_lat,
      billing_long: location.billing_long,
      
      // Device and IP features
      is_new_device: behavioral.is_new_device ? 1 : 0,
      geo_distance_from_last_txn: location.geo_distance_from_last_txn,
      
      // Time-based features
      seconds_since_last_txn: aggregation.seconds_since_last_txn,
      is_business_hours: temporal.is_business_hours ? 1 : 0,
      
      // Aggregation features
      sum_amount_1h: aggregation.user_amount_sum_1h,
      avg_amount_1h: aggregation.user_amount_avg_1h,
      txn_count_1h: aggregation.user_txn_count_1h,
      sum_amount_24h: aggregation.user_amount_sum_24h,
      avg_amount_24h: aggregation.user_amount_avg_24h,
      txn_count_24h: aggregation.user_txn_count_24h,
      
      // Velocity features
      velocity_1h: this.calculateVelocity(aggregation.user_txn_count_1h),
      
      // Derived features
      rank_amount_per_day: this.calculateAmountRank(
        transaction.amount_raw,
        aggregation.user_amount_avg_24h
      ),
      
      // Additional features for ensemble model
      geo_speed_km_per_min: this.calculateGeoSpeed(
        location.geo_distance_from_last_txn,
        aggregation.seconds_since_last_txn
      ),
      same_device_txn_1h: aggregation.device_txn_count_1h,
      
      // New features for QR model
      lat_shift: this.calculateLatShift(location.billing_lat, 38.9), // Assume center lat
      lng_shift: this.calculateLngShift(location.billing_long, -91.2), // Assume center lng
      change_in_user_agent: behavioral.is_new_device ? 1 : 0,
      
      // Suspicious indicators
      suspicious_agent: this.isSuspiciousUserAgent(behavioral.user_agent || behavioral.device_fingerprint)
    };

    // Set default values for missing fields
    features.user_agent = behavioral.user_agent || behavioral.device_fingerprint || '';
    features.device_fingerprint = behavioral.device_fingerprint || '';
    features.email_domain = behavioral.email_domain || features.email_domain || '';
    
    // Ensure all aggregation fields have defaults
    features.user_txn_count_1h = features.user_txn_count_1h || 0;
    features.user_amount_sum_1h = features.user_amount_sum_1h || 0;
    features.user_amount_avg_1h = features.user_amount_avg_1h || 0;
    features.user_unique_merchants_1h = aggregation.user_unique_merchants_1h || 0;
    features.user_txn_count_24h = features.user_txn_count_24h || 0;
    features.user_amount_sum_24h = features.user_amount_sum_24h || 0;
    features.user_amount_avg_24h = features.user_amount_avg_24h || 0;
    features.user_unique_merchants_24h = aggregation.user_unique_merchants_24h || 0;
    features.user_unique_channels_24h = aggregation.user_unique_channels_24h || 0;
    features.merchant_txn_count_1h = aggregation.merchant_txn_count_1h || 0;
    features.merchant_amount_sum_1h = aggregation.merchant_amount_sum_1h || 0;
    features.merchant_unique_users_1h = aggregation.merchant_unique_users_1h || 0;
    features.device_txn_count_1h = aggregation.device_txn_count_1h || 0;
    features.ip_txn_count_1h = aggregation.ip_txn_count_1h || 0;
    features.seconds_since_last_txn = aggregation.seconds_since_last_txn || 0;
    features.is_velocity_anomaly = aggregation.is_velocity_anomaly ? 1 : 0;
    features.is_amount_anomaly = aggregation.is_amount_anomaly ? 1 : 0;
    features.is_new_merchant = aggregation.is_new_merchant ? 1 : 0;
    features.is_new_location = aggregation.is_new_location ? 1 : 0;
    features.is_rapid_fire = aggregation.is_rapid_fire ? 1 : 0;

    // Calculate derived features
    features.amount_x_velocity = features.amount_log * features.velocity_1h;
    features.geo_rank_ratio = features.geo_distance_from_last_txn / Math.max(features.rank_amount_per_day, 1);

    return features;
  }

  /**
   * Determines if the given hour is night time
   * @param {number} hour - Hour of the day (0-23)
   * @returns {number} - 1 if night time, 0 otherwise
   */
  static isNightTime(hour) {
    return (hour >= 22 || hour <= 6) ? 1 : 0;
  }

  /**
   * Maps transaction type to index
   * @param {string} txnType - Transaction type
   * @returns {number} - Transaction type index
   */
  static getTxnTypeIndex(txnType) {
    const typeMap = {
      'QR': 2,
      'IBFT': 1,
      'TOPUP': 3
    };
    return typeMap[txnType] || 0;
  }

  /**
   * Converts amount to USD (simplified conversion)
   * @param {number} amount - Original amount
   * @param {string} currency - Currency code
   * @returns {number} - Amount in USD
   */
  static convertToUSD(amount, currency) {
    // Simplified conversion rates - in production, use real-time rates
    const rates = {
      'USD': 1,
      'VND': 0.00004,
      'JPY': 0.0067,
      'EUR': 1.08,
      'GBP': 1.27
    };
    
    return amount * (rates[currency] || 1);
  }

  /**
   * Determines if amount is considered high
   * @param {number} amount - Transaction amount
   * @returns {number} - 1 if high amount, 0 otherwise
   */
  static isHighAmount(amount) {
    // Define high amount threshold (this should be configurable)
    const HIGH_AMOUNT_THRESHOLD = 10000;
    return amount > HIGH_AMOUNT_THRESHOLD ? 1 : 0;
  }

  /**
   * Calculates velocity metric
   * @param {number} txnCount1h - Transaction count in last hour
   * @returns {number} - Velocity metric
   */
  static calculateVelocity(txnCount1h) {
    return txnCount1h || 0;
  }

  /**
   * Calculates amount rank compared to user's historical pattern
   * @param {number} currentAmount - Current transaction amount
   * @param {number} avgAmount24h - Average amount in last 24h
   * @returns {number} - Rank score
   */
  static calculateAmountRank(currentAmount, avgAmount24h) {
    if (!avgAmount24h || avgAmount24h === 0) {
      return 1;
    }
    return Math.min(currentAmount / avgAmount24h, 10); // Cap at 10x
  }

  /**
   * Calculates geographical speed
   * @param {number} distance - Distance in km
   * @param {number} timeDiff - Time difference in seconds
   * @returns {number} - Speed in km per minute
   */
  static calculateGeoSpeed(distance, timeDiff) {
    if (!timeDiff || timeDiff <= 0) {
      return 0;
    }
    const timeInMinutes = timeDiff / 60;
    return distance / timeInMinutes;
  }

  /**
   * Calculates latitude shift from reference point
   * @param {number} lat - Current latitude
   * @param {number} refLat - Reference latitude
   * @returns {number} - Latitude shift
   */
  static calculateLatShift(lat, refLat) {
    return Math.abs(lat - refLat);
  }

  /**
   * Calculates longitude shift from reference point
   * @param {number} lng - Current longitude
   * @param {number} refLng - Reference longitude
   * @returns {number} - Longitude shift
   */
  static calculateLngShift(lng, refLng) {
    return Math.abs(lng - refLng);
  }

  /**
   * Checks if user agent is suspicious
   * @param {string} userAgent - User agent string
   * @returns {number} - 1 if suspicious, 0 otherwise
   */
  static isSuspiciousUserAgent(userAgent) {
    if (!userAgent) return 1;
    
    const suspiciousPatterns = [
      /bot/i,
      /crawler/i,
      /spider/i,
      /automated/i,
      /script/i
    ];
    
    return suspiciousPatterns.some(pattern => pattern.test(userAgent)) ? 1 : 0;
  }

  /**
   * Calculates latitude shift from center
   * @param {number} lat - Current latitude
   * @param {number} centerLat - Center latitude
   * @returns {number} - Latitude shift
   */
  static calculateLatShift(lat, centerLat) {
    return Math.abs(lat - centerLat);
  }

  /**
   * Calculates longitude shift from center
   * @param {number} lng - Current longitude
   * @param {number} centerLng - Center longitude
   * @returns {number} - Longitude shift
   */
  static calculateLngShift(lng, centerLng) {
    return Math.abs(lng - centerLng);
  }

  /**
   * Prepares features array for ONNX model input
   * @param {Object} features - Feature object
   * @param {string} modelType - Model type (qr, ibft, topup)
   * @param {string} stage - Model stage ('xgb' for QR, ignored for others)
   * @returns {Float32Array|BigInt64Array} - Features array for ONNX
   */
  static prepareONNXInput(features, modelType, stage = 'xgb') {
    let featureOrder = [];
    let dataType = 'float32';

    if (modelType === 'qr') {
      // QR XGBoost model requires int64 input with 34 features
      dataType = 'int64';
      featureOrder = [
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
    } else if (modelType === 'ibft') {
      // IBFT model requires float32 input with 27 features (same as TopUp but without last feature)
      dataType = 'float32';
      featureOrder = [
        'is_weekend', 'is_night', 'txn_type_idx', 'is_ibft', 'is_topup', 'is_qr',
        'amount_usd', 'amount_log', 'is_high_amount', 
        'name_is_ascii', 'name_has_digit', 'name_has_symbol', 'name_repeated_char', 
        'disposable_email_flag', 'billing_lat', 'billing_long', 'seconds_since_last_txn',
        'is_new_device', 'geo_distance_from_last_txn', 'suspicious_agent', 'is_business_hours', 
        'sum_amount_1h', 'avg_amount_1h', 'txn_count_1h', 
        'sum_amount_24h', 'avg_amount_24h', 'txn_count_24h'
      ];
    } else if (modelType === 'topup') {
      // TopUp model requires float32 input with 27 features (note: spec shows 26 features but indices 0-26 = 27)
      dataType = 'float32';
      featureOrder = [
        'is_weekend', 'is_night', 'txn_type_idx', 'is_ibft', 'is_topup', 'is_qr',
        'amount_usd', 'amount_log', 'is_high_amount', 
        'name_is_ascii', 'name_has_digit', 'name_has_symbol', 'name_repeated_char', 
        'disposable_email_flag', 'billing_lat', 'billing_long', 'seconds_since_last_txn',
        'is_new_device', 'geo_distance_from_last_txn', 'suspicious_agent', 'is_business_hours', 
        'sum_amount_1h', 'avg_amount_1h', 'txn_count_1h', 
        'sum_amount_24h', 'avg_amount_24h', 'txn_count_24h'
      ];
    } else {
      throw new Error(`Unknown model type: ${modelType}`);
    }

    // Extract features in the correct order
    const featureArray = featureOrder.map(key => {
      const value = features[key];
      return typeof value === 'number' ? value : 0;
    });

    // Return correct data type based on model requirements
    if (dataType === 'int64') {
      return new BigInt64Array(featureArray.map(v => BigInt(Math.round(v))));
    } else {
      return new Float32Array(featureArray);
    }
  }

  /**
   * Validates that all required features are present
   * @param {Object} features - Feature object
   * @returns {Object} - Validation result
   */
  static validateFeatures(features) {
    const requiredFeatures = [
      'amount_usd', 'amount_log', 'txn_type_idx', 'billing_lat', 'billing_long'
    ];

    const missing = requiredFeatures.filter(key => 
      features[key] === undefined || features[key] === null
    );

    return {
      isValid: missing.length === 0,
      missingFeatures: missing
    };
  }
}

module.exports = FeatureMapper;
