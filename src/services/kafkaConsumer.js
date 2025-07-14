const { Kafka } = require('kafkajs');
const logger = require('../utils/logger');
const fraudDetectionService = require('./fraudDetectionService');

class KafkaConsumer {
  constructor() {
    this.kafka = new Kafka({
      clientId: process.env.KAFKA_CLIENT_ID || 'vpbank-ml-api',
      brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(',')
    });

    this.consumer = this.kafka.consumer({
      groupId: process.env.KAFKA_GROUP_ID || 'fraud-detection-group',
      sessionTimeout: 30000,
      heartbeatInterval: 3000
    });

    this.isRunning = false;
  }

  async start() {
    try {
      logger.info('Starting Kafka consumer...');
      
      await this.consumer.connect();
      await this.consumer.subscribe({
        topic: process.env.KAFKA_TOPIC || 'vpbank-enriched-features',
        fromBeginning: false
      });

      await this.consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          try {
            const data = JSON.parse(message.value.toString());
            
            logger.info('Received transaction for processing', {
              transactionId: data.transaction_id,
              userId: data.user_id,
              txnType: data.transaction?.txn_type,
              amount: data.transaction?.amount_raw,
              partition,
              offset: message.offset
            });

            // Process the transaction through fraud detection
            await this.processTransaction(data);

          } catch (error) {
            logger.error('Error processing Kafka message:', {
              error: error.message,
              topic,
              partition,
              offset: message.offset,
              stack: error.stack
            });
          }
        }
      });

      this.isRunning = true;
      logger.info('Kafka consumer started successfully');

    } catch (error) {
      logger.error('Failed to start Kafka consumer:', error);
      throw error;
    }
  }

  async processTransaction(transactionData) {
    const startTime = Date.now();
    
    try {
      // Validate transaction data
      if (!this.validateTransactionData(transactionData)) {
        logger.warn('Invalid transaction data received', {
          transactionId: transactionData.transaction_id
        });
        return;
      }

      // Extract transaction type
      const txnType = transactionData.transaction?.txn_type;
      if (!txnType) {
        logger.warn('Missing transaction type', {
          transactionId: transactionData.transaction_id
        });
        return;
      }

      // Process through fraud detection service
      const result = await fraudDetectionService.detectFraud(transactionData);
      
      const processingTime = Date.now() - startTime;
      
      logger.info('Transaction processed', {
        transactionId: transactionData.transaction_id,
        txnType,
        fraudScore: result.fraudScore,
        isFraud: result.isFraud,
        processingTimeMs: processingTime,
        modelUsed: result.modelType
      });

      // Log high-risk transactions
      if (result.isFraud || result.fraudScore > 0.7) {
        logger.warn('High-risk transaction detected', {
          transactionId: transactionData.transaction_id,
          userId: transactionData.user_id,
          amount: transactionData.transaction?.amount_raw,
          fraudScore: result.fraudScore,
          features: result.processedFeatures
        });
      }

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('Error processing transaction:', {
        transactionId: transactionData.transaction_id,
        error: error.message,
        processingTimeMs: processingTime,
        stack: error.stack
      });
    }
  }

  validateTransactionData(data) {
    const requiredFields = [
      'transaction_id',
      'user_id',
      'timestamp',
      'transaction',
      'temporal',
      'location',
      'aggregation',
      'behavioral'
    ];

    for (const field of requiredFields) {
      if (!data[field]) {
        logger.warn(`Missing required field: ${field}`, {
          transactionId: data.transaction_id
        });
        return false;
      }
    }

    return true;
  }

  async disconnect() {
    if (this.isRunning) {
      try {
        logger.info('Disconnecting Kafka consumer...');
        await this.consumer.disconnect();
        this.isRunning = false;
        logger.info('Kafka consumer disconnected');
      } catch (error) {
        logger.error('Error disconnecting Kafka consumer:', error);
        throw error;
      }
    }
  }

  getStatus() {
    return {
      isRunning: this.isRunning,
      topic: process.env.KAFKA_TOPIC || 'vpbank-enriched-features',
      groupId: process.env.KAFKA_GROUP_ID || 'fraud-detection-group'
    };
  }
}

module.exports = new KafkaConsumer();
