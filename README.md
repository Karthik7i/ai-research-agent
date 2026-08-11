# AI Research Agent

An end-to-end machine learning application for market data analysis and price prediction.

The project combines a PostgreSQL data layer, machine learning pipeline, model versioning, and a FastAPI service into a modular application that can be run locally or through Docker.

## Overview

The application currently provides a workflow for:

* Loading market data from PostgreSQL
* Preparing data and creating machine-learning features
* Training a Random Forest regression model
* Evaluating the trained model
* Saving and versioning trained models
* Loading models for inference
* Exposing predictions through a REST API
* Running the application with Docker and Docker Compose
* Testing application components with pytest

The project is being developed incrementally, with the goal of adding AI-powered research capabilities on top of the existing prediction service.

## Architecture

```text
                    PostgreSQL
                        |
                        v
                  Data Loader
                        |
                        v
              Feature Engineering
                        |
                        v
                Random Forest
                        |
                 +------+------+
                 |             |
                 v             v
             model.pkl    metadata.json
                 |
                 v
            Model Service
                 |
                 v
              Predictor
                 |
                 v
              FastAPI
                 |
                 v
             REST API
```

## Project Structure

```text
ai-research-agent/
│
├── api/
│   └── main.py
│
├── src/
│   ├── config.py
│   ├── database.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── model_service.py
│   └── predictor.py
│
├── pipelines/
│   └── train.py
│
├── tests/
│
├── models/
│   └── v1/
│       ├── model.pkl
│       └── metadata.json
│
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

* Python
* PostgreSQL
* pandas
* scikit-learn
* Random Forest
* FastAPI
* pytest
* Docker
* Docker Compose
* Git

## Machine Learning Pipeline

The training workflow is organized as:

```text
PostgreSQL
    ↓
Load Data
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Random Forest Regression
    ↓
Model Evaluation
    ↓
Save Model
    ↓
Save Metadata
```

The trained model is stored separately from the application code.

Models are versioned so that different training runs can be preserved:

```text
models/
├── v1/
│   ├── model.pkl
│   └── metadata.json
│
└── v2/
    ├── model.pkl
    └── metadata.json
```

## API

The prediction service is exposed through FastAPI.

### Health Check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

```bash
curl http://localhost:8000/predict/IBM
```

Example response:

```json
{
  "symbol": "IBM",
  "latest_date": "2026-01-01",
  "latest_close": 200.0,
  "predicted_next_close": 205.0
}
```

The API is intentionally kept thin. Prediction logic lives in the application layer rather than being duplicated inside the API routes.

## Running Locally

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables in `.env`:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=postgres
DB_PASSWORD=your_password
MODEL_VERSION=v1
```

Run the test suite:

```bash
pytest
```

Start the API:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

## Running with Docker

Make sure Docker Desktop is running.

Start the application:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl http://localhost:8000/predict/IBM
```

Stop the application:

```bash
docker compose down
```

## Configuration

Application configuration is managed through environment variables.

For Docker-based development, PostgreSQL running on the host machine can be accessed using:

```text
DB_HOST=host.docker.internal
```

Sensitive configuration such as database passwords and API keys is stored in `.env`.

The `.env` file is excluded from version control.

## Testing

The project uses pytest for automated testing.

Current tests cover areas including:

* Database connectivity
* Feature engineering
* Model behavior
* API endpoints
* Prediction responses
* Mocked prediction services

Run all tests with:

```bash
pytest
```

## Design

The application is organized into separate modules for data access, feature engineering, model management, prediction, and API handling.

### Separation of Concerns

Database operations, machine-learning logic, and HTTP handling are kept separate so each component can evolve independently.

### Model Service

The trained model is loaded into memory when the API starts and reused for subsequent prediction requests.

### Prediction Service

Prediction logic is implemented independently of FastAPI so it can be reused by other interfaces in the future.

### Model Versioning

Models are stored using explicit versions, allowing different model iterations to coexist and be compared.

### Testing and Mocking

External dependencies can be mocked when testing API behavior, allowing individual components to be tested without requiring the complete data pipeline.

## Roadmap

The project will continue to evolve toward an AI-powered research system.

Planned areas include:

* Automated model retraining
* Model performance tracking
* Improved model evaluation
* CI/CD
* Structured application logging
* Monitoring
* API authentication
* AI-powered research workflows
* MCP-based tools
* LLM integration
* Automated research and analysis

