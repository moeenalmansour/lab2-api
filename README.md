# Customer Churn Prediction API

## Project Overview
This project implements a Machine Learning model to predict customer churn using a REST API built with FastAPI.

## Model
- Algorithm: Logistic Regression
- Preprocessing: StandardScaler and OneHotEncoder
- Dataset: Customer churn dataset
- Target: Churn prediction

## API Endpoints

### Health Check
`GET /health`

Checks whether the API is running and the model is loaded.

### Prediction
`POST /predict`

Accepts customer information and returns:
- Prediction
- Churn probability
- Prediction result

## Input Features
- `tenure_months`
- `monthly_charges`
- `support_calls`
- `contract`
- `internet_service`

## Testing
The project includes five Pytest tests covering:
- Health endpoint
- Model loading
- Valid prediction input
- Invalid input validation
- Deterministic predictions

Run tests using:

```bash
python -m pytest -v
```

## Docker
Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -d -p 8001:8000 churn-api
```

## Technologies
- Python
- Scikit-learn
- FastAPI
- Pytest
- Docker
- GitHub Actions
- Trivy

## Author
Moeen Almnsour
