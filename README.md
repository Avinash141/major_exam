# MLOps Major Assignment

## Description

This project implements a comprehensive MLOps pipeline demonstrating the complete machine learning lifecycle from model training to deployment. The system features a Linear Regression model with manual quantization implementation, automated testing, and CI/CD integration using GitHub Actions.

**Key Features:**
- **Model Training**: Automated Linear Regression training with performance evaluation
- **Manual Quantization**: Custom 8-bit quantization implementation for model compression
- **Prediction Pipeline**: Inference system with comprehensive performance metrics
- **Testing Framework**: Unit tests ensuring code reliability and model performance
- **Containerization**: Docker support for consistent deployment environments
- **CI/CD Integration**: Automated testing and validation through GitHub Actions

The project showcases modern MLOps practices including model versioning, automated testing, containerization, and continuous integration, making it suitable for production deployment scenarios.

## Comparison Table

| Feature | Original Model | Quantized Model | Impact |
|---------|---------------|-----------------|---------|
| **Model Type** | Linear Regression | Linear Regression (Quantized) | Same algorithm, different precision |
| **Parameter Precision** | Float64 (64-bit) | UInt8 (8-bit) | 8x memory reduction |
| **Model Size** | ~8 bytes per parameter | ~1 byte per parameter | 87.5% size reduction |
| **Training Time** | Standard | N/A (uses pre-trained) | Quantization is post-training |
| **Inference Speed** | Baseline | Potentially faster | Reduced memory bandwidth |
| **Memory Usage** | High | Low | Significant reduction for large models |
| **Accuracy** | Full precision | Slight degradation | Minimal impact on Linear Regression |
| **Storage Requirements** | Standard | Compressed | 8x reduction in storage |
| **Deployment Suitability** | Server/Cloud | Edge/Mobile devices | Better for resource-constrained environments |
| **Implementation Complexity** | Simple | Manual quantization logic | Custom quantization/dequantization functions |

### Performance Metrics Comparison

| Metric | Original Model | Quantized Model | Difference |
|--------|---------------|-----------------|------------|
| **R² Score** | 0.999999 | 0.998317 | 0.001682 |
| **MSE (Training)** | 0.0093 | N/A | Post-training quantization |
| **MSE (Test)** | 0.0095 | N/A | Post-training quantization |
| **Max Prediction Diff** | Baseline | 16.196 | Absolute difference |
| **Mean Prediction Diff** | Baseline | 4.142 | Average absolute difference |
| **Model Loading Time** | Standard | Faster | Smaller file size |
| **Memory Footprint** | 100% | 12.5% | 87.5% reduction |

### Important Notes on R² Score

**R² Score Range**: R² can range from -∞ to 1.0
- **R² = 1.0**: Perfect predictions (our original model achieves 0.999999)
- **R² = 0.0**: Model performs as well as predicting the mean
- **R² < 0.0**: Model performs worse than predicting the mean (possible with poor models)

In this implementation:
- Original model R² = **0.999999** (near-perfect performance)
- Quantized model R² = **0.998317** (excellent performance with minimal degradation)
- Difference = **0.001682** (very small impact from quantization)

## Project Structure
```
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Manual quantization implementation
│   ├── predict.py        # Model prediction script
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests for training pipeline
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD pipeline
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## Usage

### Training
```bash
python src/train.py
```

### Quantization
```bash
python src/quantize.py
```

### Prediction
```bash
python src/predict.py
```

### Testing
```bash
pytest tests/
```


Name: Avinash Singh
Roll Number: G24ai1027

### Docker
```bash
docker build -t mlops-assignment .
docker run mlops-assignment
```
