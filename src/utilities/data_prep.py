"""
Data preparation script for purchase predictor project.
Generates synthetic training and test data, then preprocesses it for model training.
"""

import os
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
import logging
from src.modules.preprocessing import PurchaseDataPreprocessor, save_processed_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()
np.random.seed(42)  # For reproducibility

def generate_synthetic_data(n_samples=500):
    """Generate synthetic data matching the schema."""
    logger.info(f"Generating {n_samples} synthetic data samples...")
    
    # Product categories
    categories = ['electronics', 'books', 'clothes', 'home', 'sports']
    
    data = []
    for _ in range(n_samples):
        # Generate realistic price based on category
        category = np.random.choice(categories)
        if category == 'electronics':
            price = np.random.uniform(10.0, 500.0)
        elif category == 'books':
            price = np.random.uniform(5.0, 50.0)
        elif category == 'clothes':
            price = np.random.uniform(15.0, 200.0)
        elif category == 'home':
            price = np.random.uniform(8.0, 300.0)
        else:  # sports
            price = np.random.uniform(12.0, 400.0)
        
        # User rating (1-5)
        user_rating = np.random.randint(1, 6)
        
        # Previously purchased
        previously_purchased = np.random.choice(['yes', 'no'], p=[0.3, 0.7])
        
        # Generate label based on features (with some noise)
        # Higher rating, lower price, and previous purchase increase likelihood
        prob_like = 0.1  # Base probability
        if user_rating >= 4:
            prob_like += 0.4
        if price < 50:
            prob_like += 0.3
        elif price < 100:
            prob_like += 0.1
        if previously_purchased == 'yes':
            prob_like += 0.2
            
        # Add some randomness
        prob_like += np.random.normal(0, 0.1)
        prob_like = np.clip(prob_like, 0.05, 0.95)
        
        label = 1 if np.random.random() < prob_like else 0
        
        data.append({
            'price': round(price, 2),
            'user_rating': user_rating,
            'category': category,
            'previously_purchased': previously_purchased,
            'label': label
        })
    
    return pd.DataFrame(data)

def save_raw_data(train_df, test_df, data_dir='sample_data'):
    """Save raw CSV data for reference."""
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved raw training data to {train_path}")
    logger.info(f"Saved raw test data to {test_path}")
    
    return train_path, test_path

def main():
    """Main function to generate and prepare data."""
    logger.info("Starting data preparation...")
    
    # Generate synthetic data
    full_data = generate_synthetic_data(n_samples=500)
    
    # Split into train and test
    train_data, test_data = train_test_split(
        full_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=full_data['label']
    )
    
    # Save raw CSV files
    train_path, test_path = save_raw_data(train_data, test_data)
    
    # Preprocess data using shared preprocessor
    preprocessor = PurchaseDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform_training_data(train_data)
    X_test, y_test = preprocessor.transform_test_data(test_data)
    
    # Display data info
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Label distribution in training: {y_train.value_counts().to_dict()}")
    logger.info(f"Label distribution in test: {y_test.value_counts().to_dict()}")
    
    # Save processed data using shared utility
    save_processed_data(X_train, y_train, X_test, y_test)
    
    train_processed = pd.concat([X_train, y_train], axis=1)
    test_processed = pd.concat([X_test, y_test], axis=1)
    
    train_processed.to_csv(os.path.join(processed_dir, 'train_processed.csv'), index=False)
    test_processed.to_csv(os.path.join(processed_dir, 'test_processed.csv'), index=False)
    
    logger.info("Data preparation completed successfully!")
    
    # Display sample data
    print("\nSample training data:")
    print(train_data.head())
    print("\nSample processed features:")
    print(X_train.head())

if __name__ == "__main__":
    main()