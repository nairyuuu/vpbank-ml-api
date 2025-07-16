const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs').promises;
const logger = require('../utils/logger');
const FeatureMapper = require('../utils/featureMapper');
const alertService = require('./alertService');
const decisionService = require('./decisionService');
const PythonONNXService = require('./pythonONNXService');

class FraudDetectionService {
  constructor() {
    this.models = new Map();
    this.isInitialized = false;
    this.pythonService = new PythonONNXService();
    this.modelPaths = {
      qr: process.env.QR_XGB_MODEL_PATH || './models/xgb_qr.onnx',
      ibft: process.env.IBFT_MODEL_PATH || './models/model_ibft.onnx',
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

      // Load XGBoost QR model (main working model) with Node.js ONNX runtime
      await this.loadSingleModel('qr', this.modelPaths.qr);
      
      // For IBFT and TopUp models, use Python ONNX runtime
      // Mark them as available for Python inference
      this.models.set('ibft', 'python');
      this.models.set('topup', 'python');
      
      logger.info('IBFT and TopUp models configured for Python ONNX runtime');

      this.isInitialized = true;
      logger.info('Fraud detection service initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize fraud detection service:', error);
      throw error;
    }
  }

  async loadModelWithTimeout(modelType, modelPath, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Model loading timeout after ${timeout}ms`));
      }, timeout);

      this.loadSingleModel(modelType, modelPath)
        .then(() => {
          clearTimeout(timeoutId);
          resolve();
        })
        .catch((error) => {
          clearTimeout(timeoutId);
          reject(error);
        });
    });
  }

  async loadSingleModel(modelType, modelPath) {
    try {
      logger.info(`Loading ${modelType.toUpperCase()} model from ${modelPath}...`);
      const session = await ort.InferenceSession.create(modelPath);
      this.models.set(modelType, {
        model: session,
        type: 'single'
      });
      logger.info(`✓ Loaded ${modelType.toUpperCase()} model successfully`);
    } catch (error) {
      logger.error(`✗ Failed to load ${modelType.toUpperCase()} model: ${error.message}`);
      this.models.set(modelType, null);
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
      
      // Fill in missing optional fields with defaults
      features.amount_log = features.amount_log || Math.log10(features.amount_usd || 1);
      features.geo_distance_from_last_txn = features.geo_distance_from_last_txn || 0;
      features.seconds_since_last_txn = features.seconds_since_last_txn || 0;
      features.sum_amount_1h = features.sum_amount_1h || features.amount_usd || 0;
      features.avg_amount_1h = features.avg_amount_1h || features.amount_usd || 0;
      features.txn_count_1h = features.txn_count_1h || 1;
      features.sum_amount_24h = features.sum_amount_24h || features.amount_usd || 0;
      features.avg_amount_24h = features.avg_amount_24h || features.amount_usd || 0;
      features.txn_count_24h = features.txn_count_24h || 1;
      features.velocity_1h = features.velocity_1h || 1;
      features.rank_amount_per_day = features.rank_amount_per_day || 1;
      features.same_device_txn_1h = features.same_device_txn_1h || 1;
      features.lat_shift = features.lat_shift || 0;
      features.lng_shift = features.lng_shift || 0;
      features.geo_speed_km_per_min = features.geo_speed_km_per_min || 0;
      features.change_in_user_agent = features.change_in_user_agent || 0;
      
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
        modelVersion: prediction.modelVersion,
        riskLevel: this.classifyRisk(prediction.fraudScore, txnType),
        processedFeatures: features,
        processingTimeMs: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

      // Add model-specific information
      if (prediction.components) {
        result.components = prediction.components;
      }

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
      const modelConfig = this.models.get(modelType);
      
      if (!modelConfig) {
        // Mock prediction when model is not available
        logger.warn(`Using mock prediction for ${modelType} model`);
        return this.mockPrediction(features, modelType);
      }

      if (modelType === 'qr') {
        return await this.predictQR(features, modelConfig);
      } else if (modelType === 'ibft') {
        return await this.predictIBFT(features, modelConfig);
      } else if (modelType === 'topup') {
        return await this.predictTopup(features, modelConfig);
      }

      throw new Error(`Unsupported model type: ${modelType}`);

    } catch (error) {
      logger.error(`Error in ${modelType} model prediction:`, error);
      // Fall back to mock prediction
      return this.mockPrediction(features, modelType);
    }
  }

  async predictQR(features, modelConfig) {
    try {
      // Use XGBoost model with 34 features for QR prediction
      const input = FeatureMapper.prepareONNXInput(features, 'qr', 'xgb');
      const tensor = new ort.Tensor('int64', input, [1, input.length]);
      
      // Handle both direct session and modelConfig structure
      const session = modelConfig.model || modelConfig;
      
      const feeds = { [session.inputNames[0]]: tensor };
      const output = await session.run(feeds);
      
      // Extract fraud probability from output
      let fraudScore = 0;
      
      // Try to get probabilities output first (usually at index 1 for fraud class)
      if (output.probabilities && output.probabilities.data) {
        const probData = output.probabilities.data;
        fraudScore = probData.length > 1 ? probData[1] : probData[0];
      } else if (output.label && output.label.data) {
        // Fallback to label if probabilities not available
        fraudScore = output.label.data[0];
      } else {
        // Try first output if named outputs don't work
        const outputData = output[session.outputNames[0]].data;
        fraudScore = outputData.length > 1 ? outputData[1] : outputData[0];
      }
      
      // Ensure score is in [0,1] range
      fraudScore = Math.min(Math.max(fraudScore, 0), 1);
      
      logger.info(`QR XGBoost prediction successful: ${fraudScore}`);
      
      return {
        fraudScore: fraudScore,
        confidence: 0.90,
        modelVersion: 'qr_xgb_v1',
        details: {
          xgb_score: fraudScore,
          model_type: 'xgboost',
          feature_count: input.length
        }
      };
      
    } catch (error) {
      logger.error('Error in QR XGBoost prediction:', error);
      return this.mockPrediction(features, 'qr');
    }
  }

  async predictIBFT(features, modelConfig) {
    try {
      // Use Python ONNX service for IBFT model
      const inputArray = FeatureMapper.prepareONNXInput(features, 'ibft');
      const inputFeatures = Array.from(inputArray);
      
      const result = await this.pythonService.runInference('ibft', inputFeatures);
      
      if (!result.success) {
        throw new Error(result.error);
      }

      // Extract fraud probability from Python result
      let fraudScore = 0;
      if (result.predictions && result.predictions.length > 1) {
        const probabilities = result.predictions[1];
        if (Array.isArray(probabilities) && probabilities.length > 1) {
          fraudScore = probabilities[1]; // Index 1 is fraud probability
        } else if (typeof probabilities === 'object' && probabilities['1']) {
          fraudScore = probabilities['1']; // Key '1' is fraud probability
        }
      }

      return {
        fraudScore: Math.min(Math.max(fraudScore, 0), 1),
        confidence: Math.abs(fraudScore - 0.5) * 2,
        modelVersion: 'ibft_model_v1_python'
      };
    } catch (error) {
      logger.error('Error in IBFT model prediction:', error);
      return this.mockPrediction(features, 'ibft');
    }
  }

  async predictTopup(features, modelConfig) {
    try {
      // Use Python ONNX service for TopUp model
      const inputArray = FeatureMapper.prepareONNXInput(features, 'topup');
      const inputFeatures = Array.from(inputArray);
      
      const result = await this.pythonService.runInference('topup', inputFeatures);
      
      if (!result.success) {
        throw new Error(result.error);
      }

      // Extract fraud probability from Python result
      let fraudScore = 0;
      if (result.predictions && result.predictions.length > 1) {
        const probabilities = result.predictions[1];
        if (Array.isArray(probabilities) && probabilities.length > 1) {
          fraudScore = probabilities[1]; // Index 1 is fraud probability
        } else if (typeof probabilities === 'object' && probabilities['1']) {
          fraudScore = probabilities['1']; // Key '1' is fraud probability
        }
      }

      return {
        fraudScore: Math.min(Math.max(fraudScore, 0), 1),
        confidence: Math.abs(fraudScore - 0.5) * 2,
        modelVersion: 'topup_model_v1_python'
      };
    } catch (error) {
      logger.error('Error in TopUp model prediction:', error);
      return this.mockPrediction(features, 'topup');
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

  classifyRisk(score, modelType) {
    // Risk classification matching Python inference APIs
    if (modelType === 'qr') {
      if (score >= 0.85) return 'high';
      if (score >= this.thresholds.qr) return 'medium';
      return 'low';
    } else if (modelType === 'ibft') {
      if (score >= 0.9) return 'high';
      if (score >= this.thresholds.ibft) return 'medium';
      return 'low';
    } else if (modelType === 'topup') {
      if (score >= 0.85) return 'high';
      if (score >= this.thresholds.topup) return 'medium';
      return 'low';
    }
    return 'unknown';
  }

  getModelStatus() {
    const status = {};
    for (const [modelType, model] of this.models.entries()) {
      status[modelType] = {
        loaded: model !== null,
        threshold: this.thresholds[modelType],
        paths: this.modelPaths[modelType],
        type: model?.type || 'unknown'
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
