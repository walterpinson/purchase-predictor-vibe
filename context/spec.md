# Technical Specification

### Tech Stack

- Python 3.9+
- Azure Machine Learning SDK v2
- MLFlow (as packaging format for model)
- Scikit-learn (for model implementation)
- Pandas (for data processing)
- YAML and Conda (for environment configuration)

### Module Overview

- **data_prep.py**: Loads and splits CSV data into train/test sets.
- **train.py**: Trains the classifier, persists model using MLFlow.
- **register.py**: Registers model with Azure ML workspace.
- **deploy.py**: Deploys the model to an online endpoint.
- **score.py**: Scoring script used for deployment (REST endpoint).
- **conda.yaml**: Environment definition (dependencies).
- **config.yaml**: Project settings (resource group, workspace, endpoint names).

### File Structure

``` bash
/project-root
  |- data_prep.py
  |- train.py
  |- register.py
  |- deploy.py
  |- score.py
  |- conda.yaml
  |- config.yaml
  |- sample_data/
      |- train.csv
      |- test.csv
```

### Azure ML Guidance

- Use `azure.ai.ml` (SDK v2) for all Azure interactions.
- Use `mlflow.sklearn` for saving the model.
- Register and deploy MLFlow model via Azure ML for full compatibility.
- Use managed online endpoints for serving.

Below is an addendum for the training and test data sets and their schema, which can be appended to any of your project documents (e.g. the spec.md or plan.md).

## Training and Test Data Schema

### Data Set Overview

The project uses tabular CSV data featuring product attributes and user interactions to train and evaluate a binary classifier for predicting user preference (like/dislike) for products.

### Data Schema

| Column Name          | Type      | Description                                         | Example Values              |
|----------------------|-----------|-----------------------------------------------------|-----------------------------|
| price                | float     | Product price in US dollars                         | 9.99, 15.00, 22.49          |
| user_rating          | integer   | User rating for the product (scale 1–5)             | 4, 2, 5, 3                  |
| category             | string    | Product category                                    | electronics, books, clothes |
| previously_purchased | string    | Whether user has bought from this category before    | yes, no                     |
| label                | integer   | Target: 1 if liked, 0 if not                        | 1, 0                        |

### Example Training Data (CSV)

``` csv
price,user_rating,category,previously_purchased,label
9.99,4,electronics,yes,1
15.99,2,books,no,0
23.45,5,clothes,yes,1
8.49,3,electronics,no,0
```

Use at least 20–30 rows for training. All columns except "label" are input features; "label" is the prediction target.

### Example Test Data (CSV)

``` csv
price,user_rating,category,previously_purchased,label
10.99,5,clothes,no,1
16.99,2,books,no,0
```

Test set should match the same schema as the training set, but contain distinct rows for unbiased evaluation. Typically, use 20% of the full data for testing.

#### Formatting Requirements

- CSV files must contain a header row with the exact column names listed above.
- Categories and previously_purchased columns can be one-hot or label-encoded in preprocessing; store original values in raw CSV.
- Ensure all categorical values are consistent between train and test sets.

## Key Implementation Guidance

- Refer explicitly to Azure ML SDK v2 (`azure-ai-ml`) in all orchestration scripts.
- Use MLFlow format for saving and registering models, as it's natively supported in Azure ML endpoints.
- Avoid legacy Azure ML SDK v1, as it’s being deprecated.