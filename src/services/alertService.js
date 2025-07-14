const AWS = require('aws-sdk');
const logger = require('../utils/logger');

class AlertService {
  constructor() {
    this.sns = new AWS.SNS({
      region: process.env.AWS_REGION || 'us-west-2',
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    });
    
    this.topicArn = process.env.AWS_SNS_TOPIC_ARN;
    this.isEnabled = !!this.topicArn;
    
    if (!this.isEnabled) {
      logger.warn('AWS SNS not configured. Alerts will be logged only.');
    }
  }

  async sendFraudAlert(alertData) {
    const {
      transactionId,
      userId,
      amount,
      currency,
      txnType,
      fraudScore,
      timestamp,
      features
    } = alertData;

    const alertMessage = {
      alertType: 'FRAUD_DETECTION',
      severity: this.getSeverity(fraudScore),
      timestamp: new Date().toISOString(),
      transaction: {
        id: transactionId,
        userId,
        amount,
        currency,
        type: txnType,
        timestamp
      },
      riskAssessment: {
        fraudScore: Math.round(fraudScore * 10000) / 10000, // 4 decimal places
        riskLevel: this.getRiskLevel(fraudScore),
        confidence: this.getConfidenceLevel(fraudScore)
      },
      riskFactors: this.extractRiskFactors(features),
      metadata: {
        modelVersion: '1.0.0',
        systemId: 'vpbank-ml-api',
        environment: process.env.NODE_ENV || 'development'
      }
    };

    try {
      if (this.isEnabled) {
        await this.sendSNSAlert(alertMessage);
      } else {
        // Log alert when SNS is not available
        this.logAlert(alertMessage);
      }

      logger.info('Fraud alert processed', {
        transactionId,
        fraudScore,
        severity: alertMessage.severity
      });

    } catch (error) {
      logger.error('Failed to send fraud alert:', {
        transactionId,
        error: error.message,
        stack: error.stack
      });
      
      // Fallback to logging
      this.logAlert(alertMessage);
    }
  }

  async sendSNSAlert(alertMessage) {
    const message = {
      Message: JSON.stringify(alertMessage, null, 2),
      Subject: `🚨 Fraud Alert - ${alertMessage.riskAssessment.riskLevel} Risk Transaction`,
      TopicArn: this.topicArn,
      MessageAttributes: {
        AlertType: {
          DataType: 'String',
          StringValue: alertMessage.alertType
        },
        Severity: {
          DataType: 'String',
          StringValue: alertMessage.severity
        },
        TransactionId: {
          DataType: 'String',
          StringValue: alertMessage.transaction.id
        },
        FraudScore: {
          DataType: 'Number',
          StringValue: alertMessage.riskAssessment.fraudScore.toString()
        }
      }
    };

    const result = await this.sns.publish(message).promise();
    
    logger.info('SNS alert sent successfully', {
      messageId: result.MessageId,
      transactionId: alertMessage.transaction.id
    });

    return result;
  }

  logAlert(alertMessage) {
    logger.warn('FRAUD ALERT (SNS Disabled)', {
      alert: alertMessage
    });
  }

  getSeverity(fraudScore) {
    if (fraudScore >= 0.9) return 'CRITICAL';
    if (fraudScore >= 0.7) return 'HIGH';
    if (fraudScore >= 0.5) return 'MEDIUM';
    return 'LOW';
  }

  getRiskLevel(fraudScore) {
    if (fraudScore >= 0.8) return 'VERY_HIGH';
    if (fraudScore >= 0.6) return 'HIGH';
    if (fraudScore >= 0.4) return 'MEDIUM';
    if (fraudScore >= 0.2) return 'LOW';
    return 'VERY_LOW';
  }

  getConfidenceLevel(fraudScore) {
    // Confidence based on distance from 0.5 threshold
    const distance = Math.abs(fraudScore - 0.5);
    if (distance >= 0.4) return 'HIGH';
    if (distance >= 0.2) return 'MEDIUM';
    return 'LOW';
  }

  extractRiskFactors(features) {
    const factors = [];

    if (features.amount > 10000) {
      factors.push({
        factor: 'HIGH_AMOUNT',
        description: `Large transaction amount: ${features.amount}`,
        severity: 'HIGH'
      });
    }

    if (features.isNewDevice) {
      factors.push({
        factor: 'NEW_DEVICE',
        description: 'Transaction from new device',
        severity: 'MEDIUM'
      });
    }

    if (features.geoDistance > 1000) {
      factors.push({
        factor: 'UNUSUAL_LOCATION',
        description: `Large geographical distance: ${features.geoDistance}km`,
        severity: 'HIGH'
      });
    }

    if (features.location && (features.location.billing_lat === 0 || features.location.billing_long === 0)) {
      factors.push({
        factor: 'INVALID_LOCATION',
        description: 'Invalid or missing location data',
        severity: 'MEDIUM'
      });
    }

    return factors;
  }

  async sendSystemAlert(alertData) {
    const message = {
      alertType: 'SYSTEM_ALERT',
      severity: alertData.severity || 'MEDIUM',
      timestamp: new Date().toISOString(),
      message: alertData.message,
      details: alertData.details,
      metadata: {
        systemId: 'vpbank-ml-api',
        environment: process.env.NODE_ENV || 'development'
      }
    };

    try {
      if (this.isEnabled) {
        await this.sendSNSAlert(message);
      } else {
        this.logAlert(message);
      }
    } catch (error) {
      logger.error('Failed to send system alert:', error);
    }
  }

  getStatus() {
    return {
      enabled: this.isEnabled,
      topicArn: this.topicArn,
      region: process.env.AWS_REGION
    };
  }
}

module.exports = new AlertService();
