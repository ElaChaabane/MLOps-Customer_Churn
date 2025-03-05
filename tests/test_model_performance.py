import pytest
import mlflow
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import prepare_data, train_model

@pytest.fixture(autouse=True)
def clear_mlflow_state():
    """
    Clear MLflow state before each test.
    """
    mlflow.end_run()  # End any active run
    mlflow.start_run()  # Start a new run
    yield
    mlflow.end_run()  # End the run after the test

def test_train_model_with_dummy_data():
    """
    Test the model training functionality with dummy data.
    """
    X_train = np.random.rand(100, 8)  # Dummy data
    y_train = np.random.randint(0, 2, 100)  # Dummy labels
    
    model = train_model(X_train, y_train)
    assert model is not None  # Verify that the model is created

def test_model_accuracy_on_real_data():
    """
    Test the model's accuracy on real data.
    """
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Verify that the accuracy is above a certain threshold
    assert accuracy > 0.7, f"Model accuracy {accuracy} is below the expected threshold"