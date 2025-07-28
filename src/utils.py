"""
Utility functions for MLOps assignment
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_dataset(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """
    Load and prepare regression dataset
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        noise: Noise level in the data
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Split dataset
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.joblib')
    
    return X_train, X_test, y_train, y_test


def save_model(model, filename):
    """Save model using joblib"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """Load model using joblib"""
    return joblib.load(filename)


def calculate_r2_score(y_true, y_pred):
    """Calculate R² score manually"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def quantize_to_uint8(values, min_val=None, max_val=None):
    """
    Quantize float values to unsigned 8-bit integers
    
    Args:
        values: Array of float values to quantize
        min_val: Minimum value for scaling (if None, use min of values)
        max_val: Maximum value for scaling (if None, use max of values)
    
    Returns:
        quantized_values: Quantized uint8 values
        scale: Scale factor used
        zero_point: Zero point used
    """
    if min_val is None:
        min_val = np.min(values)
    if max_val is None:
        max_val = np.max(values)
    
    # Handle edge case where min_val == max_val
    if abs(max_val - min_val) < 1e-8:
        # All values are the same, use a small range
        range_val = max(abs(min_val) * 0.01, 1e-6)
        min_val = min_val - range_val / 2
        max_val = max_val + range_val / 2
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / 255.0
    zero_point = int(np.round(-min_val / scale))
    zero_point = np.clip(zero_point, 0, 255)  # Ensure zero_point is in valid range
    
    # Quantize
    quantized = np.round(values / scale + zero_point)
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    
    return quantized, scale, zero_point


def dequantize_from_uint8(quantized_values, scale, zero_point):
    """
    Dequantize uint8 values back to float
    
    Args:
        quantized_values: Quantized uint8 values
        scale: Scale factor used during quantization
        zero_point: Zero point used during quantization
    
    Returns:
        dequantized_values: Float values
    """
    return scale * (quantized_values.astype(np.float32) - zero_point)