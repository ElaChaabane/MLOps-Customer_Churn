import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import (
    prepare_data, 
    train_model, 
    evaluate_model, 
    save_model, 
    load_model
)

def test_model_pipeline_integration():
    """
    Test the entire model pipeline integration.
    """
    # Prepare data
    train_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv")
    test_path = os.path.expanduser("~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
    
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, "integration_test_model.joblib")
    
    # Load model
    loaded_model = load_model("integration_test_model.joblib")
    
    # Verify model prediction capabilities
    y_pred = loaded_model.predict(X_test)
    
    # Assertions
    assert model is not None
    assert len(y_pred) == len(X_test)
    assert 0 <= accuracy <= 1
    
    # Clean up test model file
    #os.remove("integration_test_model.joblib")