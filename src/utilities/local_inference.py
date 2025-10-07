"""
Local inference script for purchase predictor model.
This provides a simple REST API using Flask for local model serving.
"""

import os
import json
import logging
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utilities.preprocessing import PurchaseDataPreprocessor
from config.config_loader import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
config = None

# Create Flask app
app = Flask(__name__)

def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, config
    
    logger.info("Loading model and preprocessor...")
    
    # Load configuration
    config = load_config()
    
    # Load model
    model_path = config.get('artifacts', {}).get('model_file', 'models/model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Load preprocessor
    try:
        preprocessor = PurchaseDataPreprocessor.load_fitted_preprocessor()
        logger.info("Fitted preprocessor loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load fitted preprocessor: {e}")
        logger.info("Will use basic preprocessing fallback")
        preprocessor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        logger.info(f"Received prediction request: {data}")
        
        # Handle different input formats
        if 'data' in data:
            input_data = data['data']
        elif 'instances' in data:
            input_data = data['instances']
        else:
            input_data = data if isinstance(data, list) else [data]
        
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Preprocess the data
        processed_df = preprocess_input(df)
        
        # Make predictions
        predictions = model.predict(processed_df)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_df)
        
        # Format response
        response = {
            'predictions': predictions.tolist(),
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': getattr(model, '_model_version', 'unknown'),
            'input_count': len(input_data)
        }
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
            response['confidence'] = [max(prob) for prob in probabilities.tolist()]
        
        logger.info(f"Generated predictions: {len(predictions)} predictions")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

def preprocess_input(df):
    """Preprocess input data for prediction."""
    logger.info(f"Preprocessing input data with shape: {df.shape}")
    
    if preprocessor is not None:
        try:
            # Use fitted preprocessor
            processed_df = preprocessor.transform_inference_data(df)
            logger.info("Used fitted preprocessor for data transformation")
            return processed_df
        except Exception as e:
            logger.warning(f"Fitted preprocessor failed: {e}")
    
    # Fallback preprocessing
    logger.info("Using fallback preprocessing")
    return fallback_preprocessing(df)

def fallback_preprocessing(df):
    """Basic preprocessing fallback."""
    processed_df = df.copy()
    
    # Handle different input formats
    if df.shape[1] == 4 and all(col in df.columns for col in ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']):
        # Already preprocessed
        return processed_df
    
    # Handle named columns
    if df.shape[1] >= 4:
        # Try to map columns to expected format
        if 'category' in processed_df.columns and 'category_encoded' not in processed_df.columns:
            # Encode category
            category_mapping = {
                'electronics': 0, 'books': 1, 'clothes': 2, 'home': 3, 'sports': 4
            }
            processed_df['category_encoded'] = processed_df['category'].map(category_mapping).fillna(0)
        
        if 'previously_purchased' in processed_df.columns and 'previously_purchased_encoded' not in processed_df.columns:
            # Encode previously_purchased
            processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({
                'yes': 1, True: 1, 'no': 0, False: 0
            }).fillna(0)
    
    # Select final feature columns
    expected_features = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
    
    # Handle case where we have numerical columns in order
    if df.shape[1] == 4 and all(col not in df.columns for col in expected_features):
        processed_df.columns = expected_features
    
    # Ensure we have the right columns
    final_df = processed_df[expected_features]
    
    # Convert to float64 for consistency
    final_df = final_df.astype('float64')
    
    logger.info(f"Preprocessed data shape: {final_df.shape}")
    return final_df

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'model_type': str(type(model).__name__) if model else None,
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'expected_features': ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded'],
        'input_examples': {
            'raw_format': {
                'data': [
                    ['price', 'user_rating', 'category', 'previously_purchased'],
                    [25.99, 4, 'electronics', 'yes'],
                    [150.00, 2, 'books', 'no']
                ]
            },
            'processed_format': {
                'data': [
                    ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded'],
                    [25.99, 4, 0, 1],
                    [150.00, 2, 1, 0]
                ]
            }
        },
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Make predictions (POST)',
            '/info': 'Model information (GET)',
            '/test': 'Test with sample data (GET)'
        }
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test the model with sample data."""
    sample_data = {
        'data': [
            [25.99, 4, 0, 1],  # Low price, good rating, electronics, previous customer -> likely purchase
            [150.00, 2, 1, 0],  # High price, poor rating, books, new customer -> unlikely purchase
            [75.00, 5, 2, 1]   # Medium price, excellent rating, clothes, previous customer -> likely purchase
        ]
    }
    
    # Make prediction using the predict function
    with app.test_client() as client:
        response = client.post('/predict', 
                             data=json.dumps(sample_data),
                             content_type='application/json')
        
        result = response.get_json()
        
        # Add interpretations
        if 'predictions' in result:
            interpretations = []
            for i, pred in enumerate(result['predictions']):
                sample = sample_data['data'][i]
                interpretation = interpret_prediction(sample, pred, result.get('probabilities', [None] * len(result['predictions']))[i])
                interpretations.append(interpretation)
            
            result['interpretations'] = interpretations
    
    return jsonify(result)

def interpret_prediction(features, prediction, probabilities):
    """Interpret a prediction for a given set of features."""
    price, rating, category_encoded, previously_purchased = features
    
    category_names = {0: 'electronics', 1: 'books', 2: 'clothes', 3: 'home', 4: 'sports'}
    category = category_names.get(int(category_encoded), 'unknown')
    prev_customer = 'yes' if previously_purchased else 'no'
    
    confidence = max(probabilities) if probabilities else 'unknown'
    
    return {
        'input': {
            'price': price,
            'rating': rating,
            'category': category,
            'previous_customer': prev_customer
        },
        'prediction': int(prediction),
        'prediction_text': 'Will Purchase' if prediction else 'Will Not Purchase',
        'confidence': confidence,
        'factors': analyze_factors(price, rating, category_encoded, previously_purchased)
    }

def analyze_factors(price, rating, category_encoded, previously_purchased):
    """Analyze factors that might influence the prediction."""
    factors = []
    
    if price < 30:
        factors.append("Low price - positive factor")
    elif price > 100:
        factors.append("High price - negative factor")
    else:
        factors.append("Medium price - neutral factor")
    
    if rating >= 4:
        factors.append("Good rating - positive factor")
    elif rating <= 2:
        factors.append("Poor rating - negative factor")
    else:
        factors.append("Average rating - neutral factor")
    
    if previously_purchased:
        factors.append("Previous customer - positive factor")
    else:
        factors.append("New customer - less predictable")
    
    return factors

if __name__ == '__main__':
    # Load model and preprocessor
    try:
        load_model_and_preprocessor()
        logger.info("Model and preprocessor loaded successfully!")
        
        # Start Flask server
        print("\n" + "="*50)
        print("🚀 Purchase Predictor Local Inference Server")
        print("="*50)
        print("Server starting on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  GET  /health  - Health check")
        print("  POST /predict - Make predictions")
        print("  GET  /info    - Model information")
        print("  GET  /test    - Test with sample data")
        print("\nExample prediction request:")
        print("""
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"data": [[25.99, 4, 0, 1], [150.00, 2, 1, 0]]}'
        """)
        print("="*50)
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)