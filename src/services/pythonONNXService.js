const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

class PythonONNXService {
  constructor() {
    // Cross-platform Python path detection with fallbacks
    this.pythonPath = this.detectPythonPath();
    this.scriptPath = path.join(__dirname, 'python_model_runner.py');
  }

  detectPythonPath() {
    const candidates = [];
    
    if (os.platform() === 'win32') {
      // Windows candidates
      candidates.push(
        path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe'),
        'python',
        'python3'
      );
    } else {
      // Linux/macOS/WSL candidates
      candidates.push(
        path.join(__dirname, '..', '..', '.venv', 'bin', 'python'),
        'python3',
        'python'
      );
    }
    
    // Check each candidate
    for (const candidate of candidates) {
      if (path.isAbsolute(candidate)) {
        // Check if absolute path exists
        if (fs.existsSync(candidate)) {
          return candidate;
        }
      } else {
        // For system python, just return the command (will be resolved by spawn)
        return candidate;
      }
    }
    
    // Default fallback
    return os.platform() === 'win32' ? 'python' : 'python3';
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
