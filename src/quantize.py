"""
Manual quantization script for MLOps assignment
"""
import numpy as np
import joblib
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, quantize_to_uint8, dequantize_from_uint8


def quantize_model():
    """
    Load trained model, extract parameters, quantize them, and perform inference
    """
    print("Loading trained model...")
    try:
        model = load_model('trained_model.joblib')
    except FileNotFoundError:
        print("Error: trained_model.joblib not found. Please run train.py first.")
        return
    
    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    print(f"Coefficient range: [{np.min(coef):.4f}, {np.max(coef):.4f}]")
    
    # Save raw parameters (unquantized)
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(raw_params, 'unquant_params.joblib')
    print("Raw parameters saved to unquant_params.joblib")
    
    # Manually quantize coefficients to unsigned 8-bit integers
    print("\nQuantizing coefficients...")
    quant_coef, coef_scale, coef_zero_point = quantize_to_uint8(coef)
    
    print("\nQuantizing intercept...")
    # Intercept is a scalar, so we need to handle it separately
    quant_intercept, intercept_scale, intercept_zero_point = quantize_to_uint8(
        np.array([intercept])
    )
    quant_intercept = quant_intercept[0]  # Extract scalar
    
    print(f"Quantized coefficients shape: {quant_coef.shape}")
    print(f"Quantized coefficients dtype: {quant_coef.dtype}")
    print(f"Quantized coefficients range: [{np.min(quant_coef)}, {np.max(quant_coef)}]")
    print(f"Coefficient scale: {coef_scale:.6f}")
    print(f"Coefficient zero point: {coef_zero_point}")
    
    print(f"Quantized intercept: {quant_intercept} (dtype: {type(quant_intercept)})")
    print(f"Intercept scale: {intercept_scale:.6f}")
    print(f"Intercept zero point: {intercept_zero_point}")
    
    # Save quantized parameters
    quant_params = {
        'quant_coef': quant_coef,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'quant_intercept': quant_intercept,
        'intercept_scale': intercept_scale,
        'intercept_zero_point': intercept_zero_point
    }
    joblib.dump(quant_params, 'quant_params.joblib')
    print("Quantized parameters saved to quant_params.joblib")
    
    # Perform inference with de-quantized weights
    print("\n=== Inference Comparison ===")
    
    # Load test data
    try:
        X_test, y_test = joblib.load('test_data.joblib')
    except FileNotFoundError:
        print("Warning: test_data.joblib not found. Creating sample data for inference test.")
        from utils import load_dataset
        _, X_test, _, y_test = load_dataset()
    
    # Original model predictions
    y_pred_original = model.predict(X_test)
    
    # De-quantize parameters
    dequant_coef = dequantize_from_uint8(quant_coef, coef_scale, coef_zero_point)
    dequant_intercept = dequantize_from_uint8(
        np.array([quant_intercept]), intercept_scale, intercept_zero_point
    )[0]
    
    # Manual prediction with de-quantized weights
    y_pred_dequant = X_test @ dequant_coef + dequant_intercept
    
    # Calculate differences
    max_diff = np.max(np.abs(y_pred_original - y_pred_dequant))
    mean_diff = np.mean(np.abs(y_pred_original - y_pred_dequant))
    
    print(f"Original prediction range: [{np.min(y_pred_original):.4f}, {np.max(y_pred_original):.4f}]")
    print(f"Dequantized prediction range: [{np.min(y_pred_dequant):.4f}, {np.max(y_pred_dequant):.4f}]")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Calculate R² scores for comparison
    from sklearn.metrics import r2_score
    r2_original = r2_score(y_test, y_pred_original)
    r2_dequant = r2_score(y_test, y_pred_dequant)
    
    print(f"Original model R² score: {r2_original:.6f}")
    print(f"Dequantized model R² score: {r2_dequant:.6f}")
    print(f"R² score difference: {abs(r2_original - r2_dequant):.6f}")
    
    # Show sample predictions
    print(f"\nSample predictions (first 5):")
    print("Original  | Dequantized | Difference")
    print("-" * 40)
    for i in range(min(5, len(y_pred_original))):
        diff = abs(y_pred_original[i] - y_pred_dequant[i])
        print(f"{y_pred_original[i]:8.4f} | {y_pred_dequant[i]:10.4f} | {diff:9.6f}")
    
    return quant_params


if __name__ == "__main__":
    quantize_model()
    print("\nQuantization completed successfully!")