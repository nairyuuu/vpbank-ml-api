const PythonONNXService = require('../src/services/pythonONNXService');
const fs = require('fs');
const os = require('os');

function testPythonPathDetection() {
  console.log('Testing Python Path Detection...\n');
  
  console.log('Platform:', os.platform());
  console.log('Architecture:', os.arch());
  
  const pythonService = new PythonONNXService();
  
  console.log('Detected Python path:', pythonService.pythonPath);
  console.log('Script path:', pythonService.scriptPath);
  
  // Check if paths exist
  const pythonExists = fs.existsSync(pythonService.pythonPath);
  const scriptExists = fs.existsSync(pythonService.scriptPath);
  
  console.log('\nPath validation:');
  console.log('✓ Python executable exists:', pythonExists ? '✓' : '✗');
  console.log('✓ Python script exists:', scriptExists ? '✓' : '✗');
  
  if (!pythonExists) {
    console.log('\nAlternative Python paths to check:');
    console.log('- System python3:', 'python3');
    console.log('- System python:', 'python');
    console.log('- Virtual env (if created):', pythonService.pythonPath);
  }
  
  return { pythonExists, scriptExists };
}

// Test the service
async function testPythonService() {
  const { pythonExists, scriptExists } = testPythonPathDetection();
  
  if (!pythonExists || !scriptExists) {
    console.log('\n⚠️  Python environment not set up. Please run:');
    console.log('1. Create virtual environment: python3 -m venv .venv');
    console.log('2. Activate it: source .venv/bin/activate (Linux/WSL) or .venv\\Scripts\\activate (Windows)');
    console.log('3. Install dependencies: pip install numpy onnxruntime');
    return;
  }
  
  console.log('\n=== Testing Python ONNX Service ===');
  
  const pythonService = new PythonONNXService();
  
  // Test with a simple array
  const testFeatures = [1, 0, 2, 0, 0, 1, 40, 13.8, 1, 1, 0, 0, 0, 0, 10.8, 106.6, 3600, 0, 2.5, 0, 1, 1000000, 1000000, 1, 2000000, 1000000, 2];
  
  try {
    console.log('Testing IBFT model...');
    const result = await pythonService.runInference('ibft', testFeatures);
    
    if (result.success) {
      console.log('✓ IBFT model test successful!');
      console.log('  Predictions:', result.predictions);
    } else {
      console.log('✗ IBFT model test failed:', result.error);
    }
    
  } catch (error) {
    console.log('✗ Error testing Python service:', error.message);
  }
}

testPythonService().catch(console.error);
