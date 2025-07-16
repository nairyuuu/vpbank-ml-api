const fraudService = require('../src/services/fraudDetectionService');

async function debugQRModel() {
  console.log('Debugging QR Model Structure...\n');
  
  try {
    // Initialize the service
    await fraudService.initialize();
    
    // Get the QR model
    const qrModelConfig = fraudService.models.get('qr');
    console.log('QR Model Config:', qrModelConfig);
    
    if (qrModelConfig && qrModelConfig.model) {
      const session = qrModelConfig.model;
      console.log('Session type:', typeof session);
      console.log('Session constructor:', session.constructor.name);
      console.log('Session inputNames:', session.inputNames);
      console.log('Session outputNames:', session.outputNames);
      
      // Try to access input names
      if (session.inputNames && session.inputNames.length > 0) {
        console.log('First input name:', session.inputNames[0]);
      } else {
        console.log('No inputNames found');
      }
    } else {
      console.log('QR model not found or invalid structure');
    }
    
  } catch (error) {
    console.error('Error:', error);
  }
}

debugQRModel();
