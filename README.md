# Regulated-Fraud-Detection-End-to-End-MLOps-Pipeline

## Problem Statement & Regulated Context

Fraud is one of the fastest-growing forms of financial crime, particularly in card-not-present and digital payment scenarios.  
Financial institutions process large volumes of transactions daily, making it increasingly difficult for manual controls alone to identify suspicious activity in a timely and consistent manner.

Machine learning models can support fraud operations by assigning a risk score to each transaction, helping prioritize analyst review. However, because fraud detection systems can directly impact customers and financial outcomes, their use is subject to strict governance, monitoring, and audit requirements.

This project demonstrates how an end-to-end MLOps pipeline can be designed to deploy a fraud detection model in a **regulated environment**, balancing detection performance with traceability, reproducibility, and operational control.


## Regulated Angle

The model is intended to support analyst-led fraud triage and does not perform automated enforcement actions.  
Key regulatory considerations addressed in this project include:

- Reproducible training and configuration management
- Model versioning and lineage
- Controlled promotion to production
- Monitoring for data drift and performance degradation
- Documented rollback procedures

## Architecture Overview

The system follows a simple, modular architecture:

```mermaid
flowchart LR
    A[Data] --> B[Feature Engineering]
    B --> C[Model Training & Evaluation]
    C --> D[Model Registry]
    D --> E[Inference API]
    E --> F[PSI Drift Monitoring]
```

A detailed architecture diagram and implementation details are provided in later sections.


## Training & Evaluation

Model training is implemented through a reproducible pipeline defined in `src/models/train.py` and configured through `configs/train.yaml`.

The training workflow includes:

- Synthetic transaction data generation to simulate a fraud detection scenario
- Feature preparation and schema validation
- Train/test split with reproducibility controlled by configuration
- A baseline Logistic Regression model implemented using a Scikit-learn pipeline
- Model evaluation using multiple metrics

Evaluation metrics include:

- ROC-AUC (overall discrimination)
- PR-AUC (performance on the positive class)
- Brier score (probability calibration)
- Positive prediction rate

All experiment parameters, metrics and artifacts are logged using **MLflow**, enabling reproducible experimentation and traceability of model runs.

## Model Registry & Promotion

Trained models are registered in the **MLflow Model Registry**, which provides versioning and stage management.

The project uses a stage-based lifecycle:

- **Staging** – candidate models produced during training
- **Production** – approved models used for inference

The registry workflow enables:

- tracking model lineage through MLflow run IDs
- promoting models between stages without changing application code
- maintaining reproducibility of model artifacts and parameters

The inference service loads the model dynamically from the registry using a model URI of the form:

models:/<model_name>/<stage>

This decouples model training from deployment and allows safe model promotion or rollback.

## Serving & Deployment

Model inference is exposed through a **FastAPI service** implemented in `src/serving/app.py`.

The service performs the following steps:

1. Loads the production model from the MLflow registry at application startup
2. Validates incoming prediction requests using **Pydantic schemas**
3. Converts requests into a tabular format compatible with the trained model
4. Generates fraud probability scores
5. Applies a simple decision threshold to produce a binary prediction
6. Logs prediction metadata for monitoring purposes

The API exposes two endpoints:

- `/health` – service health check
- `/predict` – fraud probability prediction

This architecture reflects common patterns used in ML-powered risk systems where models are served through a stable inference interface.

## Serving & Deployment

Model inference is exposed through a **FastAPI service** implemented in `src/serving/app.py`.

The service performs the following steps:

1. Loads the production model from the MLflow registry at application startup
2. Validates incoming prediction requests using **Pydantic schemas**
3. Converts requests into a tabular format compatible with the trained model
4. Generates fraud probability scores
5. Applies a simple decision threshold to produce a binary prediction
6. Logs prediction metadata for monitoring purposes

The API exposes two endpoints:

- `/health` – service health check
- `/predict` – fraud probability prediction

This architecture reflects common patterns used in ML-powered risk systems where models are served through a stable inference interface.

## Governance & Documentation

- [Model Card](docs/model_card.md)
- [Data Sheet](docs/data_sheet.md)
- [Governance Checklist](docs/governance_checklist.md)
