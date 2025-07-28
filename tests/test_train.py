"""
Unit tests for training pipeline
"""
import pytest
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_dataset, save_model, load_model
from train import train_model


class TestDatasetLoading:
    """Test dataset loading functionality"""
    
    def test_dataset_loading(self):
        """Unit test dataset loading"""
        X_train, X_test, y_train, y_test = load_dataset()
        
        # Check shapes
        assert X_train.shape[0] == 800  # 80% of 1000 samples
        assert X_test.shape[0] == 200   # 20% of 1000 samples
        assert X_train.shape[1] == 10   # 10 features
        assert X_test.shape[1] == 10    # 10 features
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check no NaN values
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(y_test).any()
    
    def test_dataset_reproducibility(self):
        """Test that dataset loading is reproducible"""
        X_train1, X_test1, y_train1, y_test1 = load_dataset(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_dataset(random_state=42)
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestModelCreation:
    """Test model creation and training"""
    
    def test_model_creation(self):
        """Validate model creation (LinearRegression instance)"""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        
        # Check initial state
        assert not hasattr(model, 'coef_') or model.coef_ is None
        assert not hasattr(model, 'intercept_') or model.intercept_ is None
    
    def test_model_training(self):
        """Check if model was trained (coefficients exist)"""
        # Load data and train model
        X_train, X_test, y_train, y_test = load_dataset()
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check that model has been trained
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape
        assert model.coef_.shape == (10,)  # 10 features
        assert isinstance(model.intercept_, (float, np.floating))


class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_r2_score_threshold(self):
        """Ensure R² score exceeds minimum threshold"""
        model, r2_score = train_model()
        
        # Set minimum threshold (should be reasonable for synthetic data)
        MIN_R2_THRESHOLD = 0.8
        
        assert r2_score >= MIN_R2_THRESHOLD, f"R² score {r2_score:.4f} is below threshold {MIN_R2_THRESHOLD}"
        assert r2_score <= 1.0, f"R² score {r2_score:.4f} is above maximum possible value"
    
    def test_model_predictions(self):
        """Test that model can make predictions"""
        X_train, X_test, y_train, y_test = load_dataset()
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check prediction shape and type
        assert y_pred.shape == y_test.shape
        assert isinstance(y_pred, np.ndarray)
        assert not np.isnan(y_pred).any()


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_save_load(self):
        """Test model saving and loading functionality"""
        # Train a simple model
        X_train, X_test, y_train, y_test = load_dataset()
        original_model = LinearRegression()
        original_model.fit(X_train, y_train)
        
        # Save model
        test_filename = 'test_model.joblib'
        save_model(original_model, test_filename)
        
        # Load model
        loaded_model = load_model(test_filename)
        
        # Compare predictions
        original_pred = original_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
        # Clean up
        import os
        if os.path.exists(test_filename):
            os.remove(test_filename)


if __name__ == "__main__":
    pytest.main([__file__])