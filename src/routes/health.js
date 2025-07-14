const express = require('express');
const router = express.Router();
const fraudDetectionService = require('../services/fraudDetectionService');
const kafkaConsumer = require('../services/kafkaConsumer');
const alertService = require('../services/alertService');
const decisionService = require('../services/decisionService');
const logger = require('../utils/logger');

router.get('/', (req, res) => {
  const startTime = Date.now();
  
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    responseTime: Date.now() - startTime,
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    services: {
      fraudDetection: fraudDetectionService.getModelStatus(),
      kafka: kafkaConsumer.getStatus(),
      alerts: alertService.getStatus(),
      decisionSystem: decisionService.getStatus()
    },
    system: {
      nodeVersion: process.version,
      platform: process.platform,
      architecture: process.arch,
      memory: {
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
        external: Math.round(process.memoryUsage().external / 1024 / 1024),
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024)
      },
      cpu: process.cpuUsage()
    }
  };

  // Determine overall health status
  const isKafkaHealthy = kafkaConsumer.getStatus().isRunning;
  const areModelsHealthy = fraudDetectionService.getModelStatus().initialized;
  
  if (!isKafkaHealthy || !areModelsHealthy) {
    health.status = 'degraded';
    health.issues = [];
    
    if (!isKafkaHealthy) {
      health.issues.push('Kafka consumer not running');
    }
    
    if (!areModelsHealthy) {
      health.issues.push('Fraud detection models not initialized');
    }
  }

  const statusCode = health.status === 'healthy' ? 200 : 503;
  
  res.status(statusCode).json(health);
});

router.get('/ready', (req, res) => {
  const isReady = fraudDetectionService.getModelStatus().initialized && 
                  kafkaConsumer.getStatus().isRunning;
  
  if (isReady) {
    res.status(200).json({ status: 'ready' });
  } else {
    res.status(503).json({ status: 'not ready' });
  }
});

router.get('/live', (req, res) => {
  // Simple liveness check
  res.status(200).json({ 
    status: 'alive',
    timestamp: new Date().toISOString()
  });
});

module.exports = router;
