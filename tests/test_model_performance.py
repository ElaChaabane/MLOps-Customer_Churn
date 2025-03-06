import pytest
import mlflow
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import prepare_data, train_model

@pytest.fixture(autouse=True)
def clear_mlflow_state():
    """
    Clear MLflow state before each test.
    """
    if mlflow.active_run():
        mlflow.end_run()  # End any active run
    # Don't start a new run here - let the tests or train_model handle that
    yield
    if mlflow.active_run():
        mlflow.end_run()  # End the run after the test

def test_train_model_with_dummy_data():
    """
    Test the model training functionality with dummy data.
    """
    print("\n--- Testing model training with dummy data ---")
    X_train = np.random.rand(100, 8)  # Dummy data
    y_train = np.random.randint(0, 2, 100)  # Dummy labels
    
    # Track start time
    import time
    start_time = time.time()
    
    model = train_model(X_train, y_train)
    
    # Track end time
    training_time = time.time() - start_time
    
    # Check predictions on training data
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    assert model is not None, "Model failed to train"
    print("✅ Model training test passed")

def test_model_accuracy_on_real_data():
    """
    Test the model's accuracy on real data.
    """
    print("\n--- Testing model performance on real data ---")
    
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    
    # Check if files exist
    assert os.path.exists(train_path), f"Training data file not found: {train_path}"
    assert os.path.exists(test_path), f"Test data file not found: {test_path}"
    
    print(f"Loading data from:\n- Train: {train_path}\n- Test: {test_path}")
    
    # Track data preparation time
    import time
    start_time = time.time()
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    data_prep_time = time.time() - start_time
    
    print(f"Data preparation time: {data_prep_time:.2f} seconds")
    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Track training time
    start_time = time.time()
    model = train_model(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Model training time: {training_time:.2f} seconds")
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display performance metrics
    print(f"\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Log metrics to MLflow
    with mlflow.start_run(run_name="model_performance_test"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("prediction_time", prediction_time)
    
    # Verify that the accuracy is above the threshold
    threshold = 0.7
    assert accuracy > threshold, f"Model accuracy {accuracy:.4f} is below the expected threshold {threshold}"
    print(f"✅ Model accuracy test passed (threshold: {threshold})")

if __name__ == "__main__":
    # This allows running the tests directly without pytest
    test_train_model_with_dummy_data()
    test_model_accuracy_on_real_data()