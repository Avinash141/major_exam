# MLOps Major Assignment

This project implements a complete MLOps pipeline with Linear Regression model training, quantization, testing, and CI/CD.

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

### Docker
```bash
docker build -t mlops-assignment .
docker run mlops-assignment
```