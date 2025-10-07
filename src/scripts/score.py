"""
Scoring script for purchase predictor model deployment.
This script is used by Azure ML managed online endpoint for inference.
"""

import json
import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from preprocessing import PurchaseDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """
    Initialize the model for scoring.
    This function is called when the container is initialized/started.
    """
    global model, preprocessor
    
    logger.info("Initializing model for scoring...")
    
    # Get the path to the model
    model_path = os.environ.get('AZUREML_MODEL_DIR')
    if model_path is None:
        # Fallback for local testing
        model_path = 'models'
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Load MLFlow model
        mlflow_model_path = os.path.join(model_path, "model")
        if os.path.exists(mlflow_model_path):
            model = mlflow.sklearn.load_model(mlflow_model_path)
            logger.info("MLFlow model loaded successfully")
        else:
            # Fallback to joblib model
            joblib_model_path = os.path.join(model_path, "model.pkl")
            if os.path.exists(joblib_model_path):
                model = joblib.load(joblib_model_path)
                logger.info("Joblib model loaded successfully")
            else:
                raise FileNotFoundError("No model found in the specified path")
        
        # Load fitted preprocessor
        preprocessor = PurchaseDataPreprocessor.load_fitted_preprocessor()
        logger.info("Preprocessor loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e
    
    logger.info("Model initialization completed successfully")

def run(raw_data):
    """
    Make predictions on the input data.
    
    Args:
        raw_data (str): JSON string containing input data
        
    Returns:
        str: JSON string containing predictions
    """
    try:
        logger.info("Processing prediction request...")
        
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Received data: {data}")
        
        # Convert to DataFrame for processing
        if 'data' in data:
            # Handle structured input with 'data' key
            input_data = data['data']
        else:
            # Handle direct array input
            input_data = data
        
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Handle different input formats
        if df.shape[1] == 4:
            # Preprocessed input: [price, user_rating, category_encoded, previously_purchased_encoded]
            feature_names = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
            df.columns = feature_names
        elif df.shape[1] == 5 and 'label' not in df.columns:
            # Raw input: [price, user_rating, category, previously_purchased, extra_column]
            feature_names = ['price', 'user_rating', 'category', 'previously_purchased', 'extra']
            df.columns = feature_names
            df = preprocess_raw_input(df)
        else:
            # Handle named columns
            if 'category' in df.columns and 'category_encoded' not in df.columns:
                df = preprocess_raw_input(df)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        
        # Format response
        response = {
            'predictions': predictions.tolist()
        }
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
        
        logger.info(f"Generated predictions: {response}")
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        error_response = {
            'error': str(e),
            'message': 'Prediction failed'
        }
        return json.dumps(error_response)

def preprocess_raw_input(df):
    """
    Preprocess raw input data using shared preprocessor.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for model
    """
    logger.info("Preprocessing raw input data using shared preprocessor...")
    
    try:
        # Use shared preprocessor for consistent transformation
        processed_features = preprocessor.transform_inference_data(df)
        return processed_features
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        # Fallback to basic preprocessing if shared preprocessor fails
        return _fallback_preprocessing(df)


def _fallback_preprocessing(df):
    """Fallback preprocessing if shared preprocessor is not available."""
    logger.warning("Using fallback preprocessing - results may not be optimal")
    
    processed_df = df.copy()
    
    # Basic category encoding
    if 'category' in processed_df.columns:
        category_mapping = {'electronics': 0, 'books': 1, 'clothes': 2, 'home': 3, 'sports': 4}
        processed_df['category_encoded'] = processed_df['category'].map(category_mapping).fillna(0)
    
    # Handle previously_purchased encoding
    if 'previously_purchased' in processed_df.columns:
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({'yes': 1, 'no': 0})
    
    # Select final features
    feature_columns = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
    return processed_df[feature_columns]


# For local testing
if __name__ == "__main__":
    # Initialize model
    init()
    
    # Test with sample data
    test_data_raw = json.dumps({
        "data": [
            [25.99, 4, "electronics", "yes"],
            [150.00, 2, "books", "no"]
        ]
    })
    
    test_data_processed = json.dumps({
        "data": [
            [25.99, 4, 0, 1],  # price, user_rating, category_encoded, previously_purchased_encoded
            [150.00, 2, 1, 0]
        ]
    })
    
    print("Testing with raw data:")
    result = run(test_data_raw)
    print(result)
    
    print("\nTesting with processed data:")
    result = run(test_data_processed)
    print(result)