"""
Model training script for purchase predictor project.
Trains a binary classifier using scikit-learn and saves it with MLFlow.
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config_loader import load_config
from src.utilities.preprocessing import PurchaseDataPreprocessor, load_processed_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load training and test data using shared preprocessing utilities."""
    logger.info("Loading training and test data...")
    
    # Load configuration to get data paths and processing settings
    config = load_config()
    train_path = config.get('data', {}).get('train_path', 'sample_data/train.csv')
    test_path = config.get('data', {}).get('test_path', 'sample_data/test.csv')
    
    # Get data processing configuration
    data_processing = config.get('data_processing', {})
    handle_missing = data_processing.get('handle_missing', 'drop')
    use_float_types = data_processing.get('use_float_types', True)
    drop_threshold = data_processing.get('drop_threshold', 0.1)
    
    # Try to load processed data first
    processed_data = load_processed_data()
    if processed_data is not None:
        logger.info("Using previously processed data")
        return processed_data
    
    # If processed data doesn't exist, load raw data and process it
    elif os.path.exists(train_path):
        logger.info("Processed data not found, loading and preprocessing raw data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Use shared preprocessor with configuration
        preprocessor = PurchaseDataPreprocessor(
            handle_missing=handle_missing,
            use_float_types=use_float_types,
            drop_threshold=drop_threshold
        )
        logger.info(f"Preprocessor configured: handle_missing={handle_missing}, use_float_types={use_float_types}")
        
        X_train, y_train = preprocessor.fit_transform_training_data(train_df)
        X_test, y_test = preprocessor.transform_test_data(test_df)
        
        return X_train, X_test, y_train, y_test
    
    else:
        raise FileNotFoundError(f"No training data found at {train_path}. Please run src/pipeline/data_prep.py first.")

def create_model(config):
    """Create and return a model based on configuration."""
    model_type = config.get('model', {}).get('type', 'random_forest')
    random_state = config.get('model', {}).get('random_state', 42)
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=5
        )
        logger.info("Created Random Forest classifier")
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000
        )
        logger.info("Created Logistic Regression classifier")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_model(X_train, y_train, model):
    """Train the model."""
    logger.info(f"Training model on {len(X_train)} samples with {X_train.shape[1] if hasattr(X_train, 'shape') else 'unknown'} features...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info(f"Evaluating model on {len(X_test)} test samples...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("Classification Report (Test Set):")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred
    }

def save_model_with_mlflow(model, X_train, config, metrics):
    """Save model using MLFlow."""
    logger.info("Saving model with MLFlow...")
    
    # Get artifact paths from config
    models_dir = config.get('artifacts', {}).get('models_dir', 'models')
    run_id_file = config.get('artifacts', {}).get('run_id_file', 'models/run_id.txt')
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Use local MLFlow tracking (Azure ML will sync automatically if configured)
    # This avoids the azureml:// URI compatibility issue
    logger.info("Using local MLFlow tracking (compatible with Azure ML)")
    
    # Set MLFlow experiment
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'purchase_predictor')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=config.get('mlflow', {}).get('run_name', 'training_run')):
        # Log parameters
        if hasattr(model, 'n_estimators'):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, 'max_depth'):
            mlflow.log_param("max_depth", model.max_depth)
        if hasattr(model, 'random_state'):
            mlflow.log_param("random_state", model.random_state)
        
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        
        # Log model with explicit schema definition
        registered_model_name = config.get('mlflow', {}).get('registered_model_name', 'purchase_predictor_model')
        
        # Create input example ensuring float64 types for robustness
        input_example = X_train.iloc[:3].copy() if hasattr(X_train, 'iloc') else X_train[:3].copy()
        
        # Ensure all columns are float64 to handle potential missing values
        if hasattr(input_example, 'astype'):
            input_example = input_example.astype('float64')
        
        # Create explicit signature to avoid warnings
        from mlflow.types.schema import Schema, ColSpec
        from mlflow.types import DataType
        from mlflow.models import ModelSignature
        
        # Define schema explicitly with float64 for all features
        input_schema = Schema([
            ColSpec(DataType.double, "price"),
            ColSpec(DataType.double, "user_rating"), 
            ColSpec(DataType.double, "category_encoded"),
            ColSpec(DataType.double, "previously_purchased_encoded")
        ])
        
        output_schema = Schema([ColSpec(DataType.long)])  # Binary classification output
        
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature  # Explicit schema prevents inference warnings
        )
        
        # Get run ID for later use
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Model saved with run ID: {run_id}")
        
        # Save run ID to file for registration script
        with open(run_id_file, 'w') as f:
            f.write(run_id)
        
        # Log Azure ML details for easier registration
        try:
            logger.info(f"Azure ML Workspace: {config['azure']['workspace_name']}")
            logger.info(f"Experiment: {experiment_name}")
            logger.info("Run completed successfully for Azure ML registration")
        except Exception as e:
            logger.debug(f"Could not log Azure ML details: {e}")
    
    return run_id

def main():
    """Main training function."""
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Get artifact paths from config
    models_dir = config.get('artifacts', {}).get('models_dir', 'models')
    local_model_file = config.get('artifacts', {}).get('local_model_file', 'models/model.pkl')
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create model
    model = create_model(config)
    
    # Train model
    trained_model = train_model(X_train, y_train, model)
    
    # Evaluate model
    metrics = evaluate_model(trained_model, X_test, y_test)
    
    # Save model with MLFlow
    run_id = save_model_with_mlflow(trained_model, X_train, config, metrics)
    
    # Also save model locally for backup
    joblib.dump(trained_model, local_model_file)
    logger.info(f"Model also saved locally to {local_model_file}")
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"MLFlow run ID: {run_id}")

if __name__ == "__main__":
    main()