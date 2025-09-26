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
from config.config_loader import load_config
from src.modules.preprocessing import PurchaseDataPreprocessor, load_processed_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load training and test data using shared preprocessing utilities."""
    logger.info("Loading training and test data...")
    
    # Load configuration to get data paths
    config = load_config()
    train_path = config.get('data', {}).get('train_path', 'sample_data/train.csv')
    test_path = config.get('data', {}).get('test_path', 'sample_data/test.csv')
    
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
        
        # Use shared preprocessor
        preprocessor = PurchaseDataPreprocessor()
        X_train, y_train = preprocessor.fit_transform_training_data(train_df)
        X_test, y_test = preprocessor.transform_test_data(test_df)
        
        return X_train, X_test, y_train, y_test
    
    else:
        raise FileNotFoundError(f"No training data found at {train_path}. Please run src/utilities/data_prep.py first.")

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
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
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
        
        # Log model
        artifact_path = config.get('mlflow', {}).get('artifact_path', 'model')
        registered_model_name = config.get('mlflow', {}).get('registered_model_name', 'purchase_predictor_model')
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        
        # Get run ID for later use
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Model saved with run ID: {run_id}")
        
        # Save run ID to file for registration script
        with open(run_id_file, 'w') as f:
            f.write(run_id)
    
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