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
        # Fallback for local testing - try multiple possible paths
        possible_paths = [
            'models',           # Current directory
            '../models',        # Parent directory (for server/ subdirectory)
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')  # Project root models
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            model_path = '../models'  # Default fallback
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Environment variables:")
    for key, value in os.environ.items():
        if 'AZURE' in key or 'MODEL' in key:
            logger.info(f"  {key}: {value}")
    
    # List contents of model directory
    if os.path.exists(model_path):
        logger.info(f"Contents of {model_path}:")
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                logger.info(f"  FILE: {item}")
            else:
                logger.info(f"  DIR:  {item}/")
    else:
        logger.warning(f"Model path {model_path} does not exist!")
    
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
        logger.info("Attempting to load fitted preprocessor...")
        try:
            preprocessor = PurchaseDataPreprocessor.load_fitted_preprocessor()
            logger.info("Preprocessor loaded successfully")
        except Exception as preprocessor_error:
            logger.error(f"Failed to load fitted preprocessor: {preprocessor_error}")
            logger.info("Creating fallback preprocessor...")
            # Create a basic preprocessor instance for fallback
            preprocessor = None
            logger.info("Will use fallback preprocessing for all requests")
            
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
        
        # Handle different input formats - detect by content, not just shape
        if df.shape[1] == 4:
            # Check if this is raw or preprocessed data by looking at data types
            # If we have string data in columns 2 or 3, it's raw input
            has_string_data = (
                df.iloc[:, 2].dtype == 'object' or  # category column
                df.iloc[:, 3].dtype == 'object'     # previously_purchased column
            )
            
            if has_string_data:
                # Raw input: [price, user_rating, category, previously_purchased]
                feature_names = ['price', 'user_rating', 'category', 'previously_purchased']
                df.columns = feature_names
                df = preprocess_raw_input(df)
            else:
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
        return response  # Return Python dict, not JSON string
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        error_response = {
            'error': str(e),
            'message': 'Prediction failed'
        }
        return error_response  # Return Python dict, not JSON string

def preprocess_raw_input(df):
    """
    Preprocess raw input data using shared preprocessor.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for model
    """
    logger.info("Preprocessing raw input data using shared preprocessor...")
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Input DataFrame columns: {list(df.columns)}")
    logger.info(f"Input DataFrame sample: {df.head().to_dict('records') if len(df) > 0 else 'No data'}")
    
    # If preprocessor failed to load, go straight to fallback
    if preprocessor is None:
        logger.warning("No preprocessor available, using fallback preprocessing")
        return _fallback_preprocessing(df)
    
    try:
        # Use shared preprocessor for consistent transformation
        processed_features = preprocessor.transform_inference_data(df)
        logger.info(f"Preprocessed features shape: {processed_features.shape}")
        logger.info(f"Preprocessed features columns: {list(processed_features.columns)}")
        logger.info(f"Preprocessed sample: {processed_features.head().to_dict('records') if len(processed_features) > 0 else 'No data'}")
        
        # Check for NaN values
        nan_counts = processed_features.isnull().sum()
        if nan_counts.any():
            logger.error(f"NaN values found after preprocessing: {nan_counts.to_dict()}")
            logger.error("Falling back to manual preprocessing")
            return _fallback_preprocessing(df)
        
        return processed_features
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        logger.error("Falling back to manual preprocessing")
        # Fallback to basic preprocessing if shared preprocessor fails
        return _fallback_preprocessing(df)


def _fallback_preprocessing(df):
    """Fallback preprocessing if shared preprocessor is not available."""
    logger.warning("Using fallback preprocessing - results may not be optimal")
    
    processed_df = df.copy()
    
    # Basic category encoding - match the ACTUAL training data encoder mapping
    if 'category' in processed_df.columns:
        category_mapping = {
            'books': 0,       # Actual trained mapping
            'clothes': 1,     # Actual trained mapping  
            'electronics': 2, # Actual trained mapping
            'home': 3,        # Actual trained mapping
            'sports': 4       # Actual trained mapping
        }
        processed_df['category_encoded'] = processed_df['category'].map(category_mapping)
        
        # Handle unknown categories by setting to 2 (electronics) as a reasonable default
        processed_df['category_encoded'] = processed_df['category_encoded'].fillna(2)
    
    # Handle previously_purchased encoding
    if 'previously_purchased' in processed_df.columns:
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({'yes': 1, 'no': 0})
        # Handle unknown values by setting to 0 (no)
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased_encoded'].fillna(0)
    
    # Ensure all numeric columns are properly typed
    for col in ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    # Select final features
    feature_columns = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
    result_df = processed_df[feature_columns]
    
    logger.info(f"Fallback preprocessing result shape: {result_df.shape}")
    logger.info(f"Fallback preprocessing result columns: {list(result_df.columns)}")
    logger.info(f"Sample values: {result_df.iloc[0].to_dict() if len(result_df) > 0 else 'No data'}")
    
    return result_df


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
    print(json.dumps(result, indent=2))  # Pretty print the actual JSON
    
    print("\nTesting with processed data:")
    result = run(test_data_processed)
    print(json.dumps(result, indent=2))  # Pretty print the actual JSON