# Data Schema and Model Specifications

This document provides comprehensive information about the data schema, feature engineering, model specifications, and preprocessing pipeline for the Purchase Predictor project.

## Overview

The Purchase Predictor uses a binary classification model to predict whether a user will purchase a product based on product characteristics and user history. The model processes both raw and preprocessed data formats through a comprehensive feature engineering pipeline.

## Data Schema

### Input Data Format

#### Raw Data Schema

**File Location:** `sample_data/train.csv`, `sample_data/test.csv`

| Column | Type | Description | Example Values | Constraints |
|--------|------|-------------|----------------|-------------|
| `price` | float | Product price in USD | 25.99, 150.00, 75.50 | > 0.0 |
| `user_rating` | int/float | User rating (1-5 scale) | 1, 2, 3, 4, 5 | 1 ≤ rating ≤ 5 |
| `category` | string | Product category | "electronics", "books", "clothes" | Categorical |
| `previously_purchased` | string | User's purchase history | "yes", "no" | Binary categorical |
| `purchased` | int | Target variable (training only) | 0, 1 | Binary (0=No, 1=Yes) |

**Example Raw Data:**
```csv
price,user_rating,category,previously_purchased,purchased
25.99,4,electronics,yes,1
150.00,2,books,no,0
75.50,5,clothes,yes,1
12.95,3,electronics,no,0
```

#### Preprocessed Data Schema

**File Location:** `processed_data/train_processed.csv`, `processed_data/test_processed.csv`

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `price` | float | Normalized product price | 0.234, 0.876, 0.543 | Min-max scaled |
| `user_rating` | float | Normalized user rating | 0.75, 0.25, 1.0 | Min-max scaled |
| `category_encoded` | int | Label-encoded category | 0, 1, 2 | Electronics=0, Books=1, Clothes=2 |
| `previously_purchased_encoded` | int | Label-encoded purchase history | 0, 1 | No=0, Yes=1 |
| `purchased` | int | Target variable | 0, 1 | Training data only |

**Example Preprocessed Data:**
```csv
price,user_rating,category_encoded,previously_purchased_encoded,purchased
0.234,0.75,0,1,1
0.876,0.25,1,0,0
0.543,1.0,2,1,1
0.087,0.5,0,0,0
```

### API Input Format

The API accepts both raw and preprocessed data formats:

#### Raw Data API Format (Recommended)
```json
{
  "data": [
    [25.99, 4, "electronics", "yes"],
    [150.00, 2, "books", "no"]
  ]
}
```

#### Preprocessed Data API Format
```json
{
  "data": [
    [0.234, 0.75, 0, 1],
    [0.876, 0.25, 1, 0]
  ]
}
```

### Data Types and Validation

#### Numeric Features

**Price Validation:**
- Type: float
- Range: > 0.0 (positive values only)
- Currency: USD
- Precision: 2 decimal places recommended

**User Rating Validation:**
- Type: int or float
- Range: 1 ≤ rating ≤ 5
- Description: Likert scale rating
- Missing values: Not allowed

#### Categorical Features

**Category Values:**
- Valid categories: ["electronics", "books", "clothes", "home", "sports", "beauty"]
- Case sensitivity: Case-insensitive (normalized to lowercase)
- Missing values: Not allowed
- New categories: Handled as "unknown" during preprocessing

**Previously Purchased Values:**
- Valid values: ["yes", "no", "true", "false", "1", "0"]
- Case sensitivity: Case-insensitive
- Normalization: Converted to ["yes", "no"]
- Missing values: Treated as "no"

#### Target Variable

**Purchased (Training Only):**
- Type: int
- Values: 0 (not purchased), 1 (purchased)
- Distribution: Should be reasonably balanced (30-70% either class)

## Feature Engineering Pipeline

### Preprocessing Architecture

The preprocessing pipeline consists of the following stages:

1. **Data Validation** - Input format and type checking
2. **Missing Value Handling** - Imputation and cleaning
3. **Feature Transformation** - Encoding and scaling
4. **Feature Selection** - Relevant feature extraction
5. **Data Validation** - Output format verification

### Preprocessing Components

#### 1. Data Validation

```python
def validate_input_data(data):
    """Validate input data format and types"""
    required_columns = ['price', 'user_rating', 'category', 'previously_purchased']
    
    # Check column presence
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(data['price']):
        raise ValueError("Price must be numeric")
    
    if not pd.api.types.is_numeric_dtype(data['user_rating']):
        raise ValueError("User rating must be numeric")
    
    # Validate ranges
    if (data['price'] <= 0).any():
        raise ValueError("Price must be positive")
    
    if (data['user_rating'] < 1).any() or (data['user_rating'] > 5).any():
        raise ValueError("User rating must be between 1 and 5")
```

#### 2. Missing Value Handling

```python
def handle_missing_values(data):
    """Handle missing values in the dataset"""
    # Fill missing ratings with median
    data['user_rating'].fillna(data['user_rating'].median(), inplace=True)
    
    # Fill missing prices with median
    data['price'].fillna(data['price'].median(), inplace=True)
    
    # Fill missing categories with 'unknown'
    data['category'].fillna('unknown', inplace=True)
    
    # Fill missing purchase history with 'no'
    data['previously_purchased'].fillna('no', inplace=True)
    
    return data
```

#### 3. Feature Transformation

**Categorical Encoding:**
```python
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(data, encoders=None):
    """Encode categorical features using label encoding"""
    if encoders is None:
        encoders = {}
    
    # Category encoding
    if 'category' not in encoders:
        encoders['category'] = LabelEncoder()
        data['category_encoded'] = encoders['category'].fit_transform(data['category'])
    else:
        # Handle unseen categories
        known_categories = set(encoders['category'].classes_)
        unknown_mask = ~data['category'].isin(known_categories)
        data.loc[unknown_mask, 'category'] = 'unknown'
        
        # Add 'unknown' to encoder if not present
        if 'unknown' not in known_categories:
            encoders['category'].classes_ = np.append(encoders['category'].classes_, 'unknown')
        
        data['category_encoded'] = encoders['category'].transform(data['category'])
    
    # Previously purchased encoding
    purchase_mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
    data['previously_purchased_encoded'] = data['previously_purchased'].str.lower().map(purchase_mapping)
    
    return data, encoders
```

**Numerical Scaling:**
```python
from sklearn.preprocessing import MinMaxScaler

def scale_numerical_features(data, scalers=None):
    """Scale numerical features using Min-Max scaling"""
    if scalers is None:
        scalers = {}
    
    numerical_features = ['price', 'user_rating']
    
    for feature in numerical_features:
        if feature not in scalers:
            scalers[feature] = MinMaxScaler()
            data[feature] = scalers[feature].fit_transform(data[[feature]])
        else:
            data[feature] = scalers[feature].transform(data[[feature]])
    
    return data, scalers
```

### Feature Engineering Metadata

The preprocessing pipeline saves metadata for consistent transformation:

**File Location:** `models/preprocessing_metadata.pkl`

**Metadata Contents:**
```python
preprocessing_metadata = {
    'encoders': {
        'category': LabelEncoder(),  # Fitted category encoder
    },
    'scalers': {
        'price': MinMaxScaler(),     # Fitted price scaler
        'user_rating': MinMaxScaler() # Fitted rating scaler
    },
    'feature_names': ['price', 'user_rating', 'category_encoded', 'previously_purchased_encoded'],
    'target_name': 'purchased',
    'preprocessing_version': '1.0.0',
    'created_at': '2025-01-01T12:00:00Z'
}
```

## Model Specifications

### Model Architecture

**Model Type:** Random Forest Classifier

**Framework:** scikit-learn

**Model File:** `models/model.pkl`

### Model Configuration

```yaml
# From config.yaml
model:
  type: "RandomForestClassifier"
  parameters:
    n_estimators: 100          # Number of trees
    max_depth: 10              # Maximum tree depth
    min_samples_split: 2       # Minimum samples to split
    min_samples_leaf: 1        # Minimum samples per leaf
    random_state: 42           # Reproducibility seed
    n_jobs: -1                 # Use all available cores
```

### Model Performance Metrics

**Target Metrics:**
- **Accuracy:** ≥ 0.85
- **Precision:** ≥ 0.80 (purchase prediction)
- **Recall:** ≥ 0.75 (purchase prediction)
- **F1-Score:** ≥ 0.80
- **AUC-ROC:** ≥ 0.85

**Cross-Validation:**
- **Strategy:** 5-fold cross-validation
- **Stratification:** Stratified by target variable
- **Evaluation:** Mean and standard deviation of metrics

### Feature Importance

Expected feature importance ranking (model-dependent):

1. **Previously Purchased** (0.35-0.45) - Strong predictor of future behavior
2. **Price** (0.25-0.35) - Major factor in purchase decisions
3. **User Rating** (0.15-0.25) - Quality indicator
4. **Category** (0.10-0.20) - Product type preferences

## Data Processing Workflow

### Training Data Processing

```bash
# Step 1: Load raw training data
raw_data = pd.read_csv('sample_data/train.csv')

# Step 2: Validate input format
validate_input_data(raw_data)

# Step 3: Handle missing values
clean_data = handle_missing_values(raw_data)

# Step 4: Encode categorical features
encoded_data, encoders = encode_categorical_features(clean_data)

# Step 5: Scale numerical features
scaled_data, scalers = scale_numerical_features(encoded_data)

# Step 6: Save preprocessing metadata
save_preprocessing_metadata(encoders, scalers)

# Step 7: Save processed data
scaled_data.to_csv('processed_data/train_processed.csv', index=False)
```

### Inference Data Processing

```bash
# Step 1: Load inference data
inference_data = load_inference_data(input_data)

# Step 2: Load preprocessing metadata
metadata = load_preprocessing_metadata()

# Step 3: Apply same transformations
processed_data = apply_preprocessing(inference_data, metadata)

# Step 4: Make predictions
predictions = model.predict(processed_data)
```

## Data Quality Guidelines

### Data Collection Best Practices

1. **Price Data:**
   - Ensure prices are in consistent currency (USD)
   - Validate against reasonable ranges ($0.01 - $10,000)
   - Remove obvious outliers or data entry errors

2. **Rating Data:**
   - Enforce 1-5 scale consistently
   - Handle missing ratings appropriately
   - Consider rating source reliability

3. **Category Data:**
   - Maintain consistent category taxonomy
   - Use standardized category names
   - Plan for new category introduction

4. **Purchase History:**
   - Ensure consistent binary format
   - Track data collection methodology
   - Consider temporal aspects of purchase history

### Data Validation Checks

#### Automated Validation

```python
def validate_data_quality(data):
    """Comprehensive data quality validation"""
    issues = []
    
    # Check for duplicates
    if data.duplicated().any():
        issues.append("Duplicate rows found")
    
    # Check price distribution
    price_outliers = (data['price'] > data['price'].quantile(0.99)) | (data['price'] < 0.01)
    if price_outliers.any():
        issues.append(f"{price_outliers.sum()} price outliers detected")
    
    # Check rating distribution
    invalid_ratings = (data['user_rating'] < 1) | (data['user_rating'] > 5)
    if invalid_ratings.any():
        issues.append(f"{invalid_ratings.sum()} invalid ratings detected")
    
    # Check category distribution
    category_counts = data['category'].value_counts()
    rare_categories = category_counts[category_counts < 10]
    if len(rare_categories) > 0:
        issues.append(f"{len(rare_categories)} rare categories with <10 samples")
    
    # Check target balance (training data)
    if 'purchased' in data.columns:
        target_balance = data['purchased'].mean()
        if target_balance < 0.1 or target_balance > 0.9:
            issues.append(f"Imbalanced target: {target_balance:.2%} positive")
    
    return issues
```

#### Data Quality Report

```python
def generate_data_quality_report(data):
    """Generate comprehensive data quality report"""
    report = {
        'dataset_info': {
            'num_samples': len(data),
            'num_features': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum()
        },
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict(),
        'numerical_stats': data.describe().to_dict(),
        'categorical_stats': {
            col: data[col].value_counts().to_dict() 
            for col in data.select_dtypes(include=['object']).columns
        },
        'quality_issues': validate_data_quality(data)
    }
    return report
```

## Schema Evolution and Versioning

### Version Management

**Current Schema Version:** 1.0.0

**Schema Versioning Strategy:**
- **Major version** (x.0.0): Breaking changes to column names or data types
- **Minor version** (1.x.0): New optional columns or non-breaking changes
- **Patch version** (1.0.x): Bug fixes or clarifications

### Backward Compatibility

The preprocessing pipeline handles multiple schema versions:

```python
def handle_schema_version(data, schema_version='1.0.0'):
    """Handle different schema versions"""
    if schema_version.startswith('1.0'):
        return process_schema_v1_0(data)
    elif schema_version.startswith('1.1'):
        return process_schema_v1_1(data)
    else:
        raise ValueError(f"Unsupported schema version: {schema_version}")
```

### Migration Guidelines

When updating the schema:

1. **Document changes** in this file
2. **Update preprocessing pipeline** to handle new format
3. **Maintain backward compatibility** for existing data
4. **Version preprocessing metadata** to track changes
5. **Test with both old and new data formats**

## Performance Considerations

### Data Processing Performance

**Recommended Batch Sizes:**
- **Training:** 10,000-50,000 samples per batch
- **Inference:** 100-1,000 samples per request
- **Preprocessing:** Memory-dependent, typically 50,000 samples

**Memory Usage:**
- **Raw data:** ~100 bytes per sample
- **Processed data:** ~50 bytes per sample
- **Model size:** ~5-10 MB (Random Forest with 100 trees)

### Optimization Strategies

1. **Vectorized Operations:** Use pandas/numpy vectorized operations
2. **Chunked Processing:** Process large datasets in chunks
3. **Caching:** Cache preprocessing metadata and transformations
4. **Parallel Processing:** Use multiprocessing for large-scale preprocessing

## Related Documentation

- [README.md](../README.md) - Quick start guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment strategies