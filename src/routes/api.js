const express = require('express');
const Joi = require('joi');
const router = express.Router();
const fraudDetectionService = require('../services/fraudDetectionService');
const alertService = require('../services/alertService');
const decisionService = require('../services/decisionService');
const logger = require('../utils/logger');

// Validation schema for transaction data
const transactionSchema = Joi.object({
  transaction_id: Joi.string().required(),
  user_id: Joi.string().required(),
  merchant_id: Joi.string().optional(),
  timestamp: Joi.string().isoDate().required(),
  transaction: Joi.object({
    amount_raw: Joi.number().positive().required(),
    amount_log: Joi.number().optional(),
    amount_range: Joi.string().optional(),
    is_round_amount: Joi.boolean().optional(),
    amount_digits: Joi.number().integer().optional(),
    txn_type: Joi.string().valid('QR', 'IBFT', 'TOPUP').required(),
    is_payment: Joi.boolean().optional(),
    is_transfer: Joi.boolean().optional(),
    is_withdrawal: Joi.boolean().optional(),
    is_deposit: Joi.boolean().optional(),
    is_topup: Joi.boolean().optional(),
    channel: Joi.string().optional(),
    is_online: Joi.boolean().optional(),
    is_atm: Joi.boolean().optional(),
    is_pos: Joi.boolean().optional(),
    is_mobile: Joi.boolean().optional(),
    is_branch: Joi.boolean().optional(),
    currency: Joi.string().optional(),
    is_foreign_currency: Joi.boolean().optional(),
    description_length: Joi.number().integer().optional(),
    description_words: Joi.number().integer().optional(),
    has_description: Joi.boolean().optional()
  }).required(),
  temporal: Joi.object({
    hour: Joi.number().integer().min(0).max(23).required(),
    minute: Joi.number().integer().min(0).max(59).required(),
    day_of_week: Joi.number().integer().min(0).max(6).required(),
    day_of_month: Joi.number().integer().min(1).max(31).required(),
    month: Joi.number().integer().min(1).max(12).required(),
    year: Joi.number().integer().required(),
    quarter: Joi.number().integer().min(1).max(4).optional(),
    week_of_year: Joi.number().integer().min(1).max(53).optional(),
    time_of_day: Joi.string().optional(),
    is_business_hours: Joi.boolean().optional(),
    is_banking_hours: Joi.boolean().optional(),
    is_weekend: Joi.boolean().optional(),
    is_weekday: Joi.boolean().optional(),
    is_morning_peak: Joi.boolean().optional(),
    is_evening_peak: Joi.boolean().optional(),
    is_lunch_time: Joi.boolean().optional(),
    is_late_night: Joi.boolean().optional(),
    is_very_early: Joi.boolean().optional(),
    hour_sin: Joi.number().optional(),
    hour_cos: Joi.number().optional(),
    dow_sin: Joi.number().optional(),
    dow_cos: Joi.number().optional(),
    month_sin: Joi.number().optional(),
    month_cos: Joi.number().optional(),
    minutes_since_midnight: Joi.number().integer().optional(),
    days_since_epoch: Joi.number().integer().optional()
  }).required(),
  location: Joi.object({
    billing_lat: Joi.number().required(),
    billing_long: Joi.number().required(),
    geo_distance_from_last_txn: Joi.number().min(0).optional()
  }).required(),
  aggregation: Joi.object({
    user_txn_count_1h: Joi.number().integer().min(0).optional(),
    user_amount_sum_1h: Joi.number().min(0).optional(),
    user_amount_avg_1h: Joi.number().min(0).optional(),
    user_unique_merchants_1h: Joi.number().integer().min(0).optional(),
    user_txn_count_24h: Joi.number().integer().min(0).optional(),
    user_amount_sum_24h: Joi.number().min(0).optional(),
    user_amount_avg_24h: Joi.number().min(0).optional(),
    user_unique_merchants_24h: Joi.number().integer().min(0).optional(),
    user_unique_channels_24h: Joi.number().integer().min(0).optional(),
    merchant_txn_count_1h: Joi.number().integer().min(0).optional(),
    merchant_amount_sum_1h: Joi.number().min(0).optional(),
    merchant_unique_users_1h: Joi.number().integer().min(0).optional(),
    device_txn_count_1h: Joi.number().integer().min(0).optional(),
    ip_txn_count_1h: Joi.number().integer().min(0).optional(),
    seconds_since_last_txn: Joi.number().min(0).optional(),
    is_velocity_anomaly: Joi.boolean().optional(),
    is_amount_anomaly: Joi.boolean().optional(),
    is_new_merchant: Joi.boolean().optional(),
    is_new_location: Joi.boolean().optional(),
    is_rapid_fire: Joi.boolean().optional()
  }).required(),
  behavioral: Joi.object({
    entity_id_hash: Joi.string().optional(),
    user_id_hash: Joi.string().optional(),
    merchant_id_hash: Joi.string().optional(),
    device_id_hash: Joi.string().allow(null).optional(),
    ip_address_hash: Joi.string().optional(),
    is_new_user: Joi.boolean().optional(),
    is_new_device: Joi.boolean().optional(),
    is_new_ip: Joi.boolean().optional(),
    email_domain: Joi.string().allow(null).optional(),
    is_disposable_email: Joi.boolean().optional(),
    has_common_email_domain: Joi.boolean().optional(),
    name_is_ascii: Joi.boolean().optional(),
    name_has_digit: Joi.boolean().optional(),
    name_has_symbol: Joi.boolean().optional(),
    name_repeated_char: Joi.boolean().optional()
  }).required(),
  feature_extraction_timestamp: Joi.string().isoDate().optional(),
  pipeline_version: Joi.string().optional()
});

// Middleware for request validation
const validateTransaction = (req, res, next) => {
  const { error, value } = transactionSchema.validate(req.body);
  
  if (error) {
    logger.warn('Invalid transaction data:', {
      requestId: req.id,
      error: error.details.map(d => d.message).join(', ')
    });
    
    return res.status(400).json({
      error: 'Invalid transaction data',
      details: error.details.map(d => ({
        field: d.path.join('.'),
        message: d.message
      })),
      requestId: req.id
    });
  }
  
  req.validatedTransaction = value;
  next();
};

// POST /api/predict - Manual fraud prediction
router.post('/predict', validateTransaction, async (req, res) => {
  const startTime = Date.now();
  
  try {
    logger.info('Manual fraud prediction requested', {
      requestId: req.id,
      transactionId: req.validatedTransaction.transaction_id,
      txnType: req.validatedTransaction.transaction.txn_type
    });

    const result = await fraudDetectionService.predictTransaction(req.validatedTransaction);
    
    res.json({
      success: true,
      requestId: req.id,
      result,
      processingTimeMs: Date.now() - startTime
    });

  } catch (error) {
    logger.error('Error in manual prediction:', {
      requestId: req.id,
      error: error.message,
      stack: error.stack
    });

    res.status(500).json({
      success: false,
      error: 'Prediction failed',
      message: error.message,
      requestId: req.id,
      processingTimeMs: Date.now() - startTime
    });
  }
});

// GET /api/models/status - Get model status
router.get('/models/status', (req, res) => {
  try {
    const status = fraudDetectionService.getModelStatus();
    
    res.json({
      success: true,
      requestId: req.id,
      models: status
    });

  } catch (error) {
    logger.error('Error getting model status:', {
      requestId: req.id,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to get model status',
      requestId: req.id
    });
  }
});

// POST /api/models/reload - Reload models
router.post('/models/reload', async (req, res) => {
  try {
    logger.info('Model reload requested', { requestId: req.id });
    
    await fraudDetectionService.reloadModels();
    
    res.json({
      success: true,
      message: 'Models reloaded successfully',
      requestId: req.id
    });

  } catch (error) {
    logger.error('Error reloading models:', {
      requestId: req.id,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to reload models',
      message: error.message,
      requestId: req.id
    });
  }
});

// GET /api/services/status - Get all services status
router.get('/services/status', async (req, res) => {
  try {
    const status = {
      fraudDetection: fraudDetectionService.getModelStatus(),
      alerts: alertService.getStatus(),
      decisionSystem: decisionService.getStatus()
    };

    // Test decision system connection
    if (status.decisionSystem.enabled) {
      status.decisionSystem.connectivity = await decisionService.testConnection();
    }

    res.json({
      success: true,
      requestId: req.id,
      services: status
    });

  } catch (error) {
    logger.error('Error getting services status:', {
      requestId: req.id,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to get services status',
      requestId: req.id
    });
  }
});

// POST /api/test/alert - Send test alert
router.post('/test/alert', async (req, res) => {
  try {
    const testAlert = {
      transactionId: `test-${Date.now()}`,
      userId: 'test-user-123',
      amount: 5000,
      currency: 'USD',
      txnType: 'QR',
      fraudScore: 0.85,
      timestamp: new Date().toISOString(),
      features: {
        amount: 5000,
        location: { billing_lat: 10.8231, billing_long: 106.6297 },
        isNewDevice: true,
        geoDistance: 500
      }
    };

    await alertService.sendFraudAlert(testAlert);

    res.json({
      success: true,
      message: 'Test alert sent successfully',
      requestId: req.id,
      testAlert
    });

  } catch (error) {
    logger.error('Error sending test alert:', {
      requestId: req.id,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to send test alert',
      message: error.message,
      requestId: req.id
    });
  }
});

// Error handling for this router
router.use((error, req, res, next) => {
  logger.error('API route error:', {
    requestId: req.id,
    path: req.path,
    method: req.method,
    error: error.message,
    stack: error.stack
  });

  res.status(500).json({
    success: false,
    error: 'Internal server error',
    requestId: req.id
  });
});

module.exports = router;
