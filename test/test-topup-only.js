const ort = require('onnxruntime-node');

async function testTopUpModel() {
  try {
    console.log('Testing TopUp model...');
    const session = await ort.InferenceSession.create('./models/topup_model.onnx');
    
    console.log('✓ TopUp model loaded successfully');
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    
    // Test with 28 features (as per featureMapper)
    const inputData = new Float32Array(28).fill(0.5);
    const tensor = new ort.Tensor('float32', inputData, [1, 28]);
    
    const feeds = { [session.inputNames[0]]: tensor };
    const output = await session.run(feeds);
    
    console.log('✓ Inference successful');
    console.log('Output keys:', Object.keys(output));
    
  } catch (error) {
    console.log('✗ TopUp model failed:', error.message);
    console.log('Stack:', error.stack);
  }
}

testTopUpModel().catch(console.error);
