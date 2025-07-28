"""
Model prediction script for MLOps assignment
"""
import numpy as np
import joblib
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model


def run_predictions():
    """Load trained model and run predictions on test set"""
    print("=== Model Prediction Script ===")
    
    # Load trained model
    try:
        model = load_model('trained_model.joblib')
        print("+ Trained model loaded successfully")
    except FileNotFoundError:
        print("X Error: trained_model.joblib not found. Please run train.py first.")
        return False
    
    # Load test data
    try:
        X_test, y_test = joblib.load('test_data.joblib')
        print("+ Test data loaded successfully")
        print(f"  Test set shape: {X_test.shape}")
    except FileNotFoundError:
        print("X Warning: test_data.joblib not found. Creating sample data...")
        from utils import load_dataset
        _, X_test, _, y_test = load_dataset()
        print("+ Sample data created")
    
    # Make predictions
    print("\nRunning predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"+ Predictions completed")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    # Print sample outputs
    print(f"\n=== Sample Predictions ===")
    print("Index | True Value | Predicted  | Difference")
    print("-" * 45)
    
    n_samples = min(10, len(y_test))
    for i in range(n_samples):
        true_val = y_test[i]
        pred_val = y_pred[i]
        diff = abs(true_val - pred_val)
        print(f"{i:5d} | {true_val:10.4f} | {pred_val:10.4f} | {diff:10.4f}")
    
    # Summary statistics
    print(f"\n=== Prediction Summary ===")
    print(f"Total predictions: {len(y_pred)}")
    print(f"True values range: [{np.min(y_test):.4f}, {np.max(y_test):.4f}]")
    print(f"Predicted range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    print(f"Mean absolute error: {np.mean(np.abs(y_test - y_pred)):.4f}")
    
    # Verify model coefficients
    print(f"\n=== Model Information ===")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(model.coef_)}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Coefficient range: [{np.min(model.coef_):.4f}, {np.max(model.coef_):.4f}]")
    
    print("\n+ Prediction script completed successfully!")
    return True


if __name__ == "__main__":
    success = run_predictions()
    if not success:
        sys.exit(1)