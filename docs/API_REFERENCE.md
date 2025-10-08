# API Reference

This document provides comprehensive information about the Purchase Predictor API endpoints, request/response formats, and integration examples.

## Overview

The Purchase Predictor provides two API deployment options:

1. **Azure ML Managed Endpoint** - Production deployment with scaling and monitoring
2. **Local Inference Server** - Development and testing environment

Both options provide identical API interfaces for seamless development-to-production workflows.

## Endpoint Information

### Azure ML Managed Endpoint

After deployment, endpoint details are saved in `models/endpoint_info.yaml`:

```yaml
endpoint_name: "purchase-predictor-endpoint"
scoring_uri: "https://your-endpoint-uri.azure.com/score"
deployment_name: "purchase-predictor-deployment"
model_name: "purchase_predictor_model"
model_version: "1"
```

### Local Development Server

Start the local server:

```bash
python src/utilities/local_inference.py
```

**Local Endpoints:**
- Base URL: `http://localhost:5000`
- Health Check: `GET /health`
- Predictions: `POST /predict`
- Model Info: `GET /info`
- Test Endpoint: `GET /test`

## API Endpoints

### Health Check

**Endpoint:** `GET /health` (local) or `GET /` (Azure ML)

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### Model Information

**Endpoint:** `GET /info` (local only)

**Response:**
```json
{
  "model_name": "purchase_predictor_model",
  "model_version": "1.0.0",
  "features": ["price", "user_rating", "category", "previously_purchased"],
  "model_type": "RandomForestClassifier",
  "preprocessing": "PurchaseDataPreprocessor"
}
```

### Make Predictions

**Endpoint:** `POST /predict` (local) or `POST /score` (Azure ML)

**Content-Type:** `application/json`

#### Request Format

The API accepts both raw and preprocessed data formats:

**Raw Data Format (Recommended):**
```json
{
  "data": [
    [25.99, 4, "electronics", "yes"],
    [150.00, 2, "books", "no"]
  ]
}
```

**Preprocessed Data Format:**
```json
{
  "data": [
    [25.99, 4, 0, 1],
    [150.00, 2, 1, 0]
  ]
}
```

**Feature Order:**
1. `price` (float): Product price in USD
2. `user_rating` (int/float): User rating (1-5)
3. `category` (string or encoded int): Product category
4. `previously_purchased` (string or encoded int): Purchase history

#### Response Format

```json
{
  "predictions": [1, 1],
  "probabilities": [
    [0.024229079008882656, 0.9757709209911173],
    [0.48167746065348316, 0.5183225393465167]
  ]
}
```

### Test Endpoint

**Endpoint:** `GET /test` (local only)

Tests the API with sample data and returns predictions.

**Response:**
```json
{
  "test_data": [
    [25.99, 4, "electronics", "yes"],
    [150.00, 2, "books", "no"]
  ],
  "predictions": [1, 1],
  "probabilities": [
    [0.024229079008882656, 0.9757709209911173],
    [0.48167746065348316, 0.5183225393465167]
  ],
  "message": "Test completed successfully"
}
```

## Response Format Details

### Understanding Predictions

#### `predictions` Array
- **Type:** Array of integers
- **Values:** `0` or `1` for each input sample
- **Meaning:**
  - `0` = **Not Purchased** (user will not purchase this product)
  - `1` = **Purchased** (user will purchase this product)

#### `probabilities` Array
- **Type:** Array of arrays (nested)
- **Format:** Each inner array contains `[probability_not_purchased, probability_purchased]`
- **Values:** Decimal numbers between 0.0 and 1.0 that sum to 1.0
- **Meaning:**
  - **First number:** Probability the user will **NOT purchase** the product (class 0)
  - **Second number:** Probability the user **WILL purchase** the product (class 1)

### Example Response Interpretation

**Request:**
```json
{
  "data": [
    [25.99, 4, "electronics", "yes"],
    [150.00, 2, "books", "no"]
  ]
}
```

**Response:**
```json
{
  "predictions": [1, 1],
  "probabilities": [
    [0.024, 0.976],
    [0.482, 0.518]
  ]
}
```

**Interpretation:**

**Sample 1: Low-priced electronics for previous customer**
- **Prediction:** `1` (Purchased)
- **Confidence:** 97.6% likelihood of purchase
- **Interpretation:** High-confidence purchase prediction

**Sample 2: High-priced books for new customer**
- **Prediction:** `1` (Purchased)
- **Confidence:** 51.8% likelihood of purchase
- **Interpretation:** Low-confidence purchase prediction (borderline case)

### Confidence Levels

| Probability Range | Confidence Level | Business Action |
|-------------------|------------------|-----------------|
| 0.9 - 1.0 | **Very High** | Strong recommendation, premium placement |
| 0.7 - 0.9 | **High** | Good recommendation, standard promotion |
| 0.6 - 0.7 | **Medium** | Moderate recommendation, targeted offers |
| 0.5 - 0.6 | **Low** | Weak recommendation, discount consideration |
| 0.0 - 0.5 | **Not Recommended** | Alternative products, different strategy |

## Integration Examples

### Python Integration

#### Basic Usage

```python
import requests
import json

# Azure ML Endpoint
endpoint_uri = "https://your-endpoint-uri.azure.com/score"
# OR Local Endpoint
# endpoint_uri = "http://localhost:5000/predict"

headers = {"Content-Type": "application/json"}

# Prepare data
data = {
    "data": [
        [25.99, 4, "electronics", "yes"],
        [150.00, 2, "books", "no"]
    ]
}

# Make request
try:
    response = requests.post(endpoint_uri, json=data, headers=headers)
    response.raise_for_status()
    predictions = response.json()
    print(json.dumps(predictions, indent=2))
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

#### Advanced Processing

```python
import requests
import pandas as pd

class PurchasePredictionClient:
    def __init__(self, endpoint_uri):
        self.endpoint_uri = endpoint_uri
        self.headers = {"Content-Type": "application/json"}
    
    def predict(self, data):
        """Make predictions for purchase behavior"""
        payload = {"data": data}
        
        try:
            response = requests.post(
                self.endpoint_uri, 
                json=payload, 
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Prediction request failed: {e}")
    
    def predict_dataframe(self, df):
        """Predict for pandas DataFrame"""
        # Convert DataFrame to list format
        data = df[['price', 'user_rating', 'category', 'previously_purchased']].values.tolist()
        
        # Get predictions
        result = self.predict(data)
        
        # Add predictions to DataFrame
        df_result = df.copy()
        df_result['predicted_purchase'] = result['predictions']
        df_result['purchase_probability'] = [probs[1] for probs in result['probabilities']]
        df_result['confidence'] = [max(probs) for probs in result['probabilities']]
        
        return df_result
    
    def get_recommendations(self, data, threshold=0.7):
        """Get high-confidence purchase recommendations"""
        result = self.predict(data)
        
        recommendations = []
        for i, (pred, probs) in enumerate(zip(result['predictions'], result['probabilities'])):
            if probs[1] >= threshold:  # High purchase probability
                recommendations.append({
                    'product_index': i,
                    'prediction': pred,
                    'purchase_probability': probs[1],
                    'confidence_level': 'High' if probs[1] > 0.8 else 'Medium',
                    'recommended': True
                })
        
        return recommendations

# Usage
client = PurchasePredictionClient("https://your-endpoint-uri.azure.com/score")

# Single prediction
data = [[25.99, 4, "electronics", "yes"]]
result = client.predict(data)

# DataFrame prediction
df = pd.DataFrame({
    'price': [25.99, 150.00, 75.50],
    'user_rating': [4, 2, 5],
    'category': ['electronics', 'books', 'clothes'],
    'previously_purchased': ['yes', 'no', 'yes']
})
df_with_predictions = client.predict_dataframe(df)

# High-confidence recommendations
recommendations = client.get_recommendations(data, threshold=0.8)
```

### JavaScript/Node.js Integration

```javascript
const axios = require('axios');

class PurchasePredictionClient {
  constructor(endpointUri) {
    this.endpointUri = endpointUri;
    this.headers = { 'Content-Type': 'application/json' };
  }

  async predict(data) {
    try {
      const response = await axios.post(this.endpointUri, 
        { data: data }, 
        { 
          headers: this.headers,
          timeout: 30000
        }
      );
      return response.data;
    } catch (error) {
      throw new Error(`Prediction request failed: ${error.message}`);
    }
  }

  async predictWithConfidence(data) {
    const result = await this.predict(data);
    
    return result.predictions.map((prediction, index) => ({
      prediction: prediction,
      purchased: prediction === 1,
      confidence: Math.max(...result.probabilities[index]),
      purchaseProbability: result.probabilities[index][1],
      confidenceLevel: this.getConfidenceLevel(result.probabilities[index][1])
    }));
  }

  getConfidenceLevel(probability) {
    if (probability >= 0.9) return 'Very High';
    if (probability >= 0.7) return 'High';
    if (probability >= 0.6) return 'Medium';
    if (probability >= 0.5) return 'Low';
    return 'Not Recommended';
  }
}

// Usage
const client = new PurchasePredictionClient('https://your-endpoint-uri.azure.com/score');

async function example() {
  const data = [
    [25.99, 4, "electronics", "yes"],
    [150.00, 2, "books", "no"]
  ];

  try {
    const predictions = await client.predictWithConfidence(data);
    console.log('Predictions:', predictions);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

example();
```

### cURL Examples

#### Basic Prediction

```bash
curl -X POST "https://your-endpoint-uri.azure.com/score" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         [25.99, 4, "electronics", "yes"],
         [150.00, 2, "books", "no"]
       ]
     }'
```

#### Local Development Testing

```bash
# Start local server first
python src/utilities/local_inference.py

# Health check
curl http://localhost:5000/health

# Model information
curl http://localhost:5000/info

# Make prediction
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         [25.99, 4, "electronics", "yes"]
       ]
     }'

# Quick test
curl http://localhost:5000/test
```

## Error Handling

### Common Error Responses

#### Invalid Input Format

**Request:**
```json
{
  "data": [
    [25.99, 4]  // Missing features
  ]
}
```

**Response:**
```json
{
  "error": "Invalid input format: expected 4 features, got 2",
  "message": "Prediction failed"
}
```

#### Model Loading Error

**Response:**
```json
{
  "error": "Model not loaded",
  "message": "Prediction failed"
}
```

#### Server Error

**Response:**
```json
{
  "error": "Internal server error",
  "message": "Prediction failed"
}
```

### Error Handling Best Practices

```python
import requests
import time

def predict_with_retry(endpoint_uri, data, max_retries=3, timeout=30):
    """Make prediction with retry logic"""
    headers = {"Content-Type": "application/json"}
    payload = {"data": data}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                endpoint_uri, 
                json=payload, 
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise Exception("Request timed out after all retries")
            time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Request failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)
    
    raise Exception("All retry attempts failed")
```

## Performance Considerations

### Request Optimization

1. **Batch Requests:** Send multiple samples in a single request
2. **Timeout Settings:** Set appropriate timeouts (30-60 seconds)
3. **Connection Pooling:** Reuse connections for multiple requests
4. **Error Handling:** Implement retry logic with exponential backoff

### Response Caching

```python
import time
from functools import lru_cache

class CachedPredictionClient:
    def __init__(self, endpoint_uri, cache_ttl=300):  # 5 minute cache
        self.client = PurchasePredictionClient(endpoint_uri)
        self.cache_ttl = cache_ttl
        
    @lru_cache(maxsize=1000)
    def _cached_predict(self, data_hash, timestamp_bucket):
        # Convert back from hash to data
        return self.client.predict(data)
    
    def predict(self, data):
        # Create hash of data for caching
        data_str = str(sorted(data))
        data_hash = hash(data_str)
        
        # Create time bucket for cache expiry
        timestamp_bucket = int(time.time() // self.cache_ttl)
        
        return self._cached_predict(data_hash, timestamp_bucket)
```

## Testing and Validation

### API Testing Scripts

See `scripts/test_api.py` for comprehensive API testing examples.

### Load Testing

```bash
# Install load testing tool
pip install locust

# Run load test against local server
locust -f tests/load_test.py --host=http://localhost:5000
```

## Related Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment options and strategies
- [DATA_SCHEMA.md](DATA_SCHEMA.md) - Data format and feature details
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions