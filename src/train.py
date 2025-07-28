"""
Model training script for MLOps assignment
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_dataset, save_model, calculate_r2_score, calculate_mse


def train_model():
    """Train Linear Regression model and evaluate performance"""
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize and train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MSE (Loss): {train_mse:.4f}")
    print(f"Test MSE (Loss): {test_mse:.4f}")
    
    # Verify model has been trained (coefficients exist)
    print(f"\nModel coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_:.4f}")
    print(f"First 5 coefficients: {model.coef_[:5]}")
    
    # Save the trained model
    save_model(model, 'trained_model.joblib')
    
    # Save test data for later use
    import joblib
    joblib.dump((X_test, y_test), 'test_data.joblib')
    print("Test data saved for prediction script")
    
    return model, test_r2


if __name__ == "__main__":
    model, r2_score = train_model()
    print(f"\nTraining completed successfully! Final R² Score: {r2_score:.4f}")