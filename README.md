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


## Training & evaluation

## Model registry & promotion

## Serving & deployment

## Monitoring & governance

##Quickstart

## Governance & Documentation

- [Model Card](docs/model_card.md)
- [Data Sheet](docs/data_sheet.md)
- [Governance Checklist](docs/governance_checklist.md)
