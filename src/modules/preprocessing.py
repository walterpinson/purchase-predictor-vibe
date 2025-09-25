"""
Shared preprocessing utilities for purchase predictor project.
Ensures consistency between data preparation and training phases.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class PurchaseDataPreprocessor:
    """Handles all preprocessing operations for purchase prediction data."""
    
    def __init__(self):
        self.le_category = LabelEncoder()
        self.feature_columns = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
        self.target_column = 'label'
    
    def fit_transform_training_data(self, df):
        """
        Fit preprocessor on training data and transform.
        
        Args:
            df (pd.DataFrame): Raw training dataframe with columns: 
                              price, user_rating, category, previously_purchased, label
        
        Returns:
            tuple: (X_train, y_train) - features and target
        """
        logger.info("Fitting preprocessor on training data...")
        processed_df = df.copy()
        
        # Encode categorical variables - fit on training data
        processed_df['category_encoded'] = self.le_category.fit_transform(processed_df['category'])
        logger.info(f"Category encoder fitted with classes: {list(self.le_category.classes_)}")
        
        # Convert previously_purchased to binary
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({'yes': 1, 'no': 0})
        
        # Save encoder for later use (training, inference, deployment)
        self._save_encoders()
        
        return self._extract_features_target(processed_df)
    
    def transform_test_data(self, df):
        """
        Transform test data using fitted preprocessor.
        
        Args:
            df (pd.DataFrame): Raw test dataframe
        
        Returns:
            tuple: (X_test, y_test) - features and target
        """
        logger.info("Transforming test data using fitted preprocessor...")
        processed_df = df.copy()
        
        # Use already fitted encoder (no fitting on test data)
        processed_df['category_encoded'] = self.le_category.transform(processed_df['category'])
        
        # Convert previously_purchased to binary
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({'yes': 1, 'no': 0})
        
        return self._extract_features_target(processed_df)
    
    def transform_inference_data(self, df):
        """
        Transform new data for inference (may not have target column).
        
        Args:
            df (pd.DataFrame): Raw inference dataframe
        
        Returns:
            pd.DataFrame: Processed features ready for model prediction
        """
        logger.info("Transforming inference data...")
        processed_df = df.copy()
        
        # Use saved encoder
        processed_df['category_encoded'] = self.le_category.transform(processed_df['category'])
        processed_df['previously_purchased_encoded'] = processed_df['previously_purchased'].map({'yes': 1, 'no': 0})
        
        # Return only features (no target for inference)
        return processed_df[self.feature_columns]
    
    def _extract_features_target(self, df):
        """Extract features and target from processed dataframe."""
        X = df[self.feature_columns]
        y = df[self.target_column] if self.target_column in df.columns else None
        
        logger.info(f"Extracted features shape: {X.shape}")
        if y is not None:
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _save_encoders(self):
        """Save fitted encoders for later use."""
        os.makedirs('models', exist_ok=True)
        
        # Save category encoder
        encoder_path = 'models/label_encoder.pkl'
        joblib.dump(self.le_category, encoder_path)
        logger.info(f"Category encoder saved to {encoder_path}")
        
        # Save feature column info for consistency
        metadata = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'category_classes': list(self.le_category.classes_)
        }
        metadata_path = 'models/preprocessing_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        logger.info(f"Preprocessing metadata saved to {metadata_path}")
    
    @classmethod
    def load_fitted_preprocessor(cls):
        """
        Load a previously fitted preprocessor from saved encoders.
        
        Returns:
            PurchaseDataPreprocessor: Loaded preprocessor instance
        """
        instance = cls()
        
        encoder_path = 'models/label_encoder.pkl'
        metadata_path = 'models/preprocessing_metadata.pkl'
        
        if os.path.exists(encoder_path):
            instance.le_category = joblib.load(encoder_path)
            logger.info(f"Category encoder loaded from {encoder_path}")
            
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                instance.feature_columns = metadata['feature_columns']
                instance.target_column = metadata['target_column']
                logger.info("Preprocessing metadata loaded")
        else:
            logger.warning("No saved encoders found. Preprocessor needs to be fitted first.")
        
        return instance


def save_processed_data(X_train, y_train, X_test, y_test, processed_dir='processed_data'):
    """
    Save processed data to CSV files for later use.
    
    Args:
        X_train, y_train, X_test, y_test: Processed datasets
        processed_dir: Directory to save processed data
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Combine features and target for saving
    train_processed = X_train.copy()
    train_processed['label'] = y_train
    
    test_processed = X_test.copy()
    test_processed['label'] = y_test
    
    # Save processed data
    train_path = os.path.join(processed_dir, 'train_processed.csv')
    test_path = os.path.join(processed_dir, 'test_processed.csv')
    
    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    
    logger.info(f"Processed training data saved to {train_path}")
    logger.info(f"Processed test data saved to {test_path}")
    
    return train_path, test_path


def load_processed_data(processed_dir='processed_data'):
    """
    Load previously processed data from CSV files.
    
    Args:
        processed_dir: Directory containing processed data
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) or None if not found
    """
    train_path = os.path.join(processed_dir, 'train_processed.csv')
    test_path = os.path.join(processed_dir, 'test_processed.csv')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Extract features and target
        feature_columns = ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded']
        
        X_train = train_df[feature_columns]
        y_train = train_df['label']
        X_test = test_df[feature_columns]
        y_test = test_df['label']
        
        logger.info(f"Loaded processed training data: {X_train.shape}")
        logger.info(f"Loaded processed test data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    logger.info("No processed data found")
    return None