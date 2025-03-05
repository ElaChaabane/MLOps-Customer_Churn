import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import evaluate_model
import lightgbm as lgb

def test_evaluate_model():
    """Test model evaluation using dummy data"""
    # Create dummy training data
    X_train = np.random.rand(100, 8)  # 100 samples, 8 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)
    
    # Create dummy test data
    X_test = np.random.rand(50, 8)  # 50 samples, 8 features
    y_test = np.random.randint(0, 2, 50)  # Binary labels (0 or 1)

       # Train the model using LightGBM
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Check that accuracy is a valid probability
    assert 0 <= accuracy <= 1