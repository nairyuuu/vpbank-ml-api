const { spawn } = require('child_process');
const path = require('path');

class PythonONNXService {
  constructor() {
    this.pythonPath = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
    this.scriptPath = path.join(__dirname, 'python_model_runner.py');
  }

  /**
   * Run inference using Python ONNX runtime
   * @param {string} modelType - Model type (qr, ibft, topup)
   * @param {Array} features - Feature array
   * @returns {Promise<Object>} - Prediction results
   */
  async runInference(modelType, features) {
    return new Promise((resolve, reject) => {
      // Convert BigInt to regular numbers for JSON serialization
      const serializedFeatures = features.map(f => 
        typeof f === 'bigint' ? Number(f) : f
      );
      
      const pythonProcess = spawn(this.pythonPath, [
        this.scriptPath,
        modelType,
        JSON.stringify(serializedFeatures)
      ]);

      let output = '';
      let error = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output.trim());
            resolve(result);
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Python process failed with code ${code}: ${error}`));
        }
      });

      pythonProcess.on('error', (err) => {
        reject(new Error(`Failed to start Python process: ${err.message}`));
      });
    });
  }
}

module.exports = PythonONNXService;
