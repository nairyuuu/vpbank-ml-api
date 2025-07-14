const axios = require('axios');
const logger = require('../utils/logger');

class DecisionService {
  constructor() {
    this.decisionSystemUrl = process.env.DECISION_SYSTEM_URL;
    this.apiKey = process.env.DECISION_SYSTEM_API_KEY;
    this.timeout = 5000; // 5 seconds timeout
    this.isEnabled = !!this.decisionSystemUrl;
    
    if (!this.isEnabled) {
      logger.warn('Decision system not configured. Decisions will be logged only.');
    }
  }

  async sendDecision(transactionData, fraudResult) {
    const decision = this.createDecision(transactionData, fraudResult);

    try {
      if (this.isEnabled) {
        await this.sendToDecisionSystem(decision);
      } else {
        this.logDecision(decision);
      }

      logger.info('Decision sent', {
        transactionId: transactionData.transaction_id,
        action: decision.action,
        fraudScore: fraudResult.fraudScore
      });

    } catch (error) {
      logger.error('Failed to send decision:', {
        transactionId: transactionData.transaction_id,
        error: error.message
      });
      
      // Fallback to logging
      this.logDecision(decision);
    }
  }

  createDecision(transactionData, fraudResult) {
    const action = this.determineAction(fraudResult);
    
    return {
      transactionId: transactionData.transaction_id,
      userId: transactionData.user_id,
      merchantId: transactionData.merchant_id,
      timestamp: new Date().toISOString(),
      originalTimestamp: transactionData.timestamp,
      
      // Transaction details
      transaction: {
        amount: transactionData.transaction?.amount_raw,
        currency: transactionData.transaction?.currency,
        type: transactionData.transaction?.txn_type,
        channel: transactionData.transaction?.channel,
        description: transactionData.transaction?.description
      },
      
      // Risk assessment
      riskAssessment: {
        fraudScore: fraudResult.fraudScore,
        isFraud: fraudResult.isFraud,
        modelType: fraudResult.modelType,
        confidence: fraudResult.confidence,
        threshold: fraudResult.threshold,
        processingTimeMs: fraudResult.processingTimeMs
      },
      
      // Decision
      decision: {
        action: action.type,
        reason: action.reason,
        confidence: action.confidence,
        requiresReview: action.requiresReview,
        riskLevel: this.getRiskLevel(fraudResult.fraudScore)
      },
      
      // Additional context
      context: {
        location: transactionData.location,
        deviceInfo: {
          isNewDevice: transactionData.behavioral?.is_new_device,
          deviceIdHash: transactionData.behavioral?.device_id_hash
        },
        behavioral: {
          isNewUser: transactionData.behavioral?.is_new_user,
          isNewIp: transactionData.behavioral?.is_new_ip
        },
        aggregation: {
          txnCount1h: transactionData.aggregation?.user_txn_count_1h,
          txnCount24h: transactionData.aggregation?.user_txn_count_24h,
          amountSum1h: transactionData.aggregation?.user_amount_sum_1h
        }
      },
      
      // Metadata
      metadata: {
        apiVersion: '1.0.0',
        systemId: 'vpbank-ml-api',
        environment: process.env.NODE_ENV || 'development',
        pipelineVersion: transactionData.pipeline_version
      }
    };
  }

  determineAction(fraudResult) {
    const { fraudScore, isFraud, confidence } = fraudResult;

    // Critical risk - Block immediately
    if (fraudScore >= 0.9) {
      return {
        type: 'BLOCK',
        reason: 'Critical fraud risk detected',
        confidence: 'HIGH',
        requiresReview: false
      };
    }

    // High risk - Block but allow manual review
    if (fraudScore >= 0.7) {
      return {
        type: 'BLOCK',
        reason: 'High fraud risk detected',
        confidence: 'HIGH',
        requiresReview: true
      };
    }

    // Medium-high risk - Require additional verification
    if (fraudScore >= 0.5) {
      return {
        type: 'CHALLENGE',
        reason: 'Moderate fraud risk - additional verification required',
        confidence: 'MEDIUM',
        requiresReview: true
      };
    }

    // Low-medium risk - Monitor but allow
    if (fraudScore >= 0.3) {
      return {
        type: 'MONITOR',
        reason: 'Low fraud risk - monitoring recommended',
        confidence: 'MEDIUM',
        requiresReview: false
      };
    }

    // Low risk - Allow
    return {
      type: 'ALLOW',
      reason: 'Low fraud risk',
      confidence: confidence >= 0.7 ? 'HIGH' : 'MEDIUM',
      requiresReview: false
    };
  }

  getRiskLevel(fraudScore) {
    if (fraudScore >= 0.8) return 'CRITICAL';
    if (fraudScore >= 0.6) return 'HIGH';
    if (fraudScore >= 0.4) return 'MEDIUM';
    if (fraudScore >= 0.2) return 'LOW';
    return 'MINIMAL';
  }

  async sendToDecisionSystem(decision) {
    const config = {
      method: 'POST',
      url: this.decisionSystemUrl,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'vpbank-ml-api/1.0.0'
      },
      data: decision
    };

    // Add API key if configured
    if (this.apiKey) {
      config.headers['Authorization'] = `Bearer ${this.apiKey}`;
      // Or use API key header if that's the expected format
      // config.headers['X-API-Key'] = this.apiKey;
    }

    const response = await axios(config);
    
    logger.info('Decision sent to decision system', {
      transactionId: decision.transactionId,
      status: response.status,
      responseTime: response.headers['x-response-time']
    });

    return response.data;
  }

  logDecision(decision) {
    logger.info('DECISION (Decision System Disabled)', {
      transactionId: decision.transactionId,
      action: decision.decision.action,
      reason: decision.decision.reason,
      fraudScore: decision.riskAssessment.fraudScore,
      riskLevel: decision.decision.riskLevel
    });
  }

  async testConnection() {
    if (!this.isEnabled) {
      return { status: 'disabled', message: 'Decision system not configured' };
    }

    try {
      const testData = {
        test: true,
        timestamp: new Date().toISOString(),
        systemId: 'vpbank-ml-api'
      };

      const response = await axios({
        method: 'POST',
        url: `${this.decisionSystemUrl}/health`,
        timeout: this.timeout,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.apiKey ? `Bearer ${this.apiKey}` : undefined
        },
        data: testData
      });

      return {
        status: 'connected',
        responseTime: response.headers['x-response-time'],
        statusCode: response.status
      };

    } catch (error) {
      logger.error('Decision system connection test failed:', error.message);
      return {
        status: 'error',
        error: error.message,
        code: error.response?.status
      };
    }
  }

  getStatus() {
    return {
      enabled: this.isEnabled,
      url: this.decisionSystemUrl,
      timeout: this.timeout,
      hasApiKey: !!this.apiKey
    };
  }
}

module.exports = new DecisionService();
