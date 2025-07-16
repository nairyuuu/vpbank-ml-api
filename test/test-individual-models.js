const ort = require('onnxruntime-node');
const path = require('path');

// Test individual model loading
async function testIndividualModels() {
  const models = [
    { name: 'QR XGBoost', path: './models/xgb_qr.onnx' },
    { name: 'IBFT', path: './models/ibft_model.onnx' },
    { name: 'TopUp', path: './models/topup_model.onnx' }
  ];

  for (const model of models) {
    try {
      console.log(`\n=== Testing ${model.name} ===`);
      console.log(`Loading from: ${model.path}`);
      
      const session = await ort.InferenceSession.create(model.path);
      
      console.log(`✓ ${model.name} loaded successfully`);
      console.log(`  Input names: ${session.inputNames.join(', ')}`);
      console.log(`  Output names: ${session.outputNames.join(', ')}`);
      
      // Test with minimal input
      const inputSize = model.name === 'QR XGBoost' ? 34 : 21; // Different sizes for different models
      
      let inputData, tensor;
      if (model.name === 'QR XGBoost') {
        inputData = new BigInt64Array(inputSize).fill(BigInt(1));
        tensor = new ort.Tensor('int64', inputData, [1, inputSize]);
      } else {
        inputData = new Float32Array(inputSize).fill(0.5);
        tensor = new ort.Tensor('float32', inputData, [1, inputSize]);
      }
      
      const feeds = { [session.inputNames[0]]: tensor };
      const output = await session.run(feeds);
      
      console.log(`  ✓ Inference successful`);
      console.log(`  Output keys: ${Object.keys(output).join(', ')}`);
      
      // Show output details
      for (const key of Object.keys(output)) {
        const outputTensor = output[key];
        const data = outputTensor.data;
        console.log(`  Output '${key}': type=${outputTensor.type}, shape=[${outputTensor.dims}], sample=${data.slice(0, 3)}`);
      }
      
    } catch (error) {
      console.log(`  ✗ ${model.name} failed: ${error.message}`);
      if (error.stack) {
        console.log(`  Stack: ${error.stack.split('\n')[0]}`);
      }
    }
  }
}

testIndividualModels().catch(console.error);
