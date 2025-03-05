import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import (
    prepare_data, 
    train_model, 
    evaluate_model, 
    impute_outliers, 
    save_model, 
    load_model
)

def test_prepare_data():
    """Test the data preparation function"""
    # Use full paths to the test CSV files
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    
    # Prepare data using explicit paths
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    
    # Check that the returned objects are not empty
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    
    # Check the shape of the data
    assert X_train.shape[1] > 0
    assert y_train.shape[0] > 0
    assert X_test.shape[1] > 0
    assert y_test.shape[0] > 0

def test_impute_outliers():
    """Test the outlier imputation function"""
    # Create a sample dataframe with outliers
    df = pd.DataFrame({
        'A': [1, 2, 3, 100, 200],  # Outliers present
        'B': [10, 20, 30, 40, 50]
    })
    
    columns_to_impute = ['A']
    df_imputed = impute_outliers(df, columns_to_impute)
    
    # Check that outliers have been replaced with mean
    assert df_imputed['A'].mean() == pytest.approx(df['A'].mean())

def test_model_training_and_evaluation():
    """Test model training and evaluation workflow"""
    # Prepare data
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Check that accuracy is a valid probability
    assert 0 <= accuracy <= 1

def test_model_save_and_load():
    """Test model saving and loading"""
    # Prepare data and train model
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    model = train_model(X_train, y_train)
    
    # Save model
    save_model(model, "test_model.joblib")
    
    # Load model
    loaded_model = load_model("test_model.joblib")
    
    # Verify model prediction capabilities
    y_pred = loaded_model.predict(X_test)
    assert len(y_pred) == len(X_test)
    
    # Clean up test model file
    os.remove("test_model.joblib")