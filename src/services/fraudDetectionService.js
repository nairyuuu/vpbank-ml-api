const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs').promises;
const logger = require('../utils/logger');
const FeatureMapper = require('../utils/featureMapper');
const alertService = require('./alertService');
const decisionService = require('./decisionService');

class FraudDetectionService {
  constructor() {
    this.models = new Map();
    this.isInitialized = false;
    this.modelPaths = {
      qr: process.env.QR_MODEL_PATH || './models/qr_model.onnx',
      ibft: process.env.IBFT_MODEL_PATH || './models/ibft_model.onnx',
      topup: process.env.TOPUP_MODEL_PATH || './models/topup_model.onnx'
    };
    this.thresholds = {
      qr: parseFloat(process.env.QR_FRAUD_THRESHOLD) || 0.5,
      ibft: parseFloat(process.env.IBFT_FRAUD_THRESHOLD) || 0.5,
      topup: parseFloat(process.env.TOPUP_FRAUD_THRESHOLD) || 0.5
    };
  }

  async initialize() {
    try {
      logger.info('Initializing fraud detection models...');

      // Check if models directory exists
      const modelsDir = path.dirname(this.modelPaths.qr);
      try {
        await fs.access(modelsDir);
      } catch (error) {
        logger.warn(`Models directory ${modelsDir} does not exist. Creating placeholder models.`);
        await this.createPlaceholderModels(modelsDir);
      }

      // Load available models
      for (const [modelType, modelPath] of Object.entries(this.modelPaths)) {
        try {
          await fs.access(modelPath);
          const session = await ort.InferenceSession.create(modelPath);
          this.models.set(modelType, session);
          logger.info(`Loaded ${modelType.toUpperCase()} model from ${modelPath}`);
        } catch (error) {
          logger.warn(`Failed to load ${modelType.toUpperCase()} model from ${modelPath}: ${error.message}`);
          // Create a mock model for testing
          this.models.set(modelType, null);
        }
      }

      this.isInitialized = true;
      logger.info('Fraud detection service initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize fraud detection service:', error);
      throw error;
    }
  }

  async createPlaceholderModels(modelsDir) {
    try {
      await fs.mkdir(modelsDir, { recursive: true });
      logger.info(`Created models directory: ${modelsDir}`);
      
      // Note: In production, you would copy actual ONNX model files here
      // For now, we'll use mock predictions when models are not available
      
    } catch (error) {
      logger.error('Error creating models directory:', error);
    }
  }

  async detectFraud(transactionData) {
    const startTime = Date.now();
    
    try {
      if (!this.isInitialized) {
        throw new Error('Fraud detection service not initialized');
      }

      // Extract transaction type
      const txnType = transactionData.transaction?.txn_type?.toLowerCase();
      if (!txnType || !['qr', 'ibft', 'topup'].includes(txnType)) {
        throw new Error(`Unsupported transaction type: ${txnType}`);
      }

      // Map Kafka features to model features
      const features = FeatureMapper.mapKafkaToModelFeatures(transactionData);
      
      // Validate features
      const validation = FeatureMapper.validateFeatures(features);
      if (!validation.isValid) {
        throw new Error(`Missing required features: ${validation.missingFeatures.join(', ')}`);
      }

      // Get model prediction
      const prediction = await this.predict(features, txnType);
      
      // Create result object
      const result = {
        transactionId: transactionData.transaction_id,
        userId: transactionData.user_id,
        modelType: txnType,
        fraudScore: prediction.fraudScore,
        isFraud: prediction.fraudScore > this.thresholds[txnType],
        threshold: this.thresholds[txnType],
        confidence: prediction.confidence,
        processedFeatures: features,
        processingTimeMs: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

      // Send alerts if fraud detected
      if (result.isFraud) {
        await this.handleFraudDetection(transactionData, result);
      }

      // Send result to decision system
      await decisionService.sendDecision(transactionData, result);

      return result;

    } catch (error) {
      logger.error('Error in fraud detection:', {
        transactionId: transactionData.transaction_id,
        error: error.message,
        processingTimeMs: Date.now() - startTime
      });
      throw error;
    }
  }

  async predict(features, modelType) {
    try {
      const model = this.models.get(modelType);
      
      if (!model) {
        // Mock prediction when model is not available
        logger.warn(`Using mock prediction for ${modelType} model`);
        return this.mockPrediction(features, modelType);
      }

      // Prepare input for ONNX model
      const inputArray = FeatureMapper.prepareONNXInput(features, modelType);
      const inputTensor = new ort.Tensor('float32', inputArray, [1, inputArray.length]);

      // Run inference
      const feeds = { [model.inputNames[0]]: inputTensor };
      const output = await model.run(feeds);
      
      // Extract prediction (assuming binary classification with probability output)
      const outputTensor = output[model.outputNames[0]];
      const fraudScore = outputTensor.data[1] || outputTensor.data[0]; // Probability of fraud class
      
      return {
        fraudScore: Math.min(Math.max(fraudScore, 0), 1), // Clamp between 0 and 1
        confidence: Math.abs(fraudScore - 0.5) * 2 // Distance from 0.5, scaled to 0-1
      };

    } catch (error) {
      logger.error(`Error in ${modelType} model prediction:`, error);
      // Fall back to mock prediction
      return this.mockPrediction(features, modelType);
    }
  }

  mockPrediction(features, modelType) {
    // Simplified rule-based mock prediction for testing
    let fraudScore = 0.1; // Base low risk
    
    // Increase score based on suspicious features
    if (features.amount_usd > 10000) fraudScore += 0.3;
    if (features.is_night) fraudScore += 0.2;
    if (features.geo_distance_from_last_txn > 1000) fraudScore += 0.25;
    if (features.velocity_1h > 5) fraudScore += 0.2;
    if (features.is_new_device) fraudScore += 0.15;
    if (features.suspicious_agent) fraudScore += 0.1;
    
    fraudScore = Math.min(fraudScore, 1.0);
    
    return {
      fraudScore,
      confidence: 0.7 // Mock confidence
    };
  }

  async handleFraudDetection(transactionData, result) {
    try {
      // Send alert
      await alertService.sendFraudAlert({
        transactionId: transactionData.transaction_id,
        userId: transactionData.user_id,
        amount: transactionData.transaction?.amount_raw,
        currency: transactionData.transaction?.currency,
        txnType: transactionData.transaction?.txn_type,
        fraudScore: result.fraudScore,
        timestamp: transactionData.timestamp,
        features: {
          amount: transactionData.transaction?.amount_raw,
          location: transactionData.location,
          isNewDevice: transactionData.behavioral?.is_new_device,
          geoDistance: transactionData.location?.geo_distance_from_last_txn
        }
      });

      logger.info('Fraud alert sent', {
        transactionId: transactionData.transaction_id,
        fraudScore: result.fraudScore
      });

    } catch (error) {
      logger.error('Error handling fraud detection:', {
        transactionId: transactionData.transaction_id,
        error: error.message
      });
    }
  }

  async predictTransaction(transactionData) {
    // Public method for manual prediction (API endpoint)
    return await this.detectFraud(transactionData);
  }

  getModelStatus() {
    const status = {};
    for (const [modelType, model] of this.models.entries()) {
      status[modelType] = {
        loaded: model !== null,
        threshold: this.thresholds[modelType],
        path: this.modelPaths[modelType]
      };
    }
    
    return {
      initialized: this.isInitialized,
      models: status
    };
  }

  async reloadModels() {
    logger.info('Reloading fraud detection models...');
    this.isInitialized = false;
    this.models.clear();
    await this.initialize();
  }
}

module.exports = new FraudDetectionService();
