# Stacked Predictive Maintenance Model

## Project Overview
**Summary:**
This document details the progress of an ongoing predictive maintenance project, employing Logistic Regression, XGBoost, Support Vector Classification, and ensemble stacking techniques to develop and evaluate machine learning models for predicting machine failures. Utilizing the _"AI4I 2020 Predictive Maintenance Dataset"_ from the UCI Machine Learning Repository [1], the project aims to create high-performance models capable of accurate machine failure prediction (binary task) and precise failure mode identification (multi-class task).
**Objectives:**
Develop and deploy a predictive maintenance solution that does the following:
- Predicts whether a machine will fail (Binary classification); and
- Identifies the type of failure (Multi-class classification)
**Platform:**
Azure Machine Learning (Python SDK)
**Status:**
Deployed to Managed Online Endpoint

## Background Information
### Predictive Maintenance
**Predictive Maintenance (PdM)** is a proactive maintenance strategy that leverages sensor data, machine learning, and statistical models to predict equipment failures before they occur. Unlike reactive maintenance (which responds to failures) or preventive maintenance (which follows fixed schedules), PdM relies on real-time monitoring of machine conditions to forecast when maintenance should be performed. This allows for timely interventions, minimizing downtime and extending equipment life.
Predictive Maintenance plays a crucial role in modern industrial operations for several reasons:
- Reduced Downtime: By predicting failures in advance, machines can be serviced before breakdowns, avoiding costly unplanned downtime.
- Cost Efficiency: It prevents unnecessary routine maintenance and helps allocate maintenance resources more efficiently.
- Extended Equipment Lifespan: Early fault detection reduces the risk of severe damage, prolonging machine life.
- Safety and Reliability: Early intervention helps prevent hazardous failures, ensuring a safer working environment.
  
### References
This study builds upon **Gerardo Cappa's 2020 Predictive Maintenance Project** on Kaggle [2], which explored the performance of five machine learning models on the AI4I 2020 Predictive Maintenance dataset. The models tested were:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classification (SVC)
- Random Forest
- XGBoost
The project evaluated these models on two classification tasks:
- **Binary Classification**: Predicting whether a machine will fail or not.
- **Multi-Class Classification**: Predicting the type of failure (e.g., Tool Wear, Overstrain, etc.).
The findings indicated that although KNN was the fastest to train, it had the lowest accuracy. XGBoost consistently outperformed the other models in both tasks in terms of accuracy and robustness.

This study also incorporates insights from the 2024 research paper, "**Machine Learning Models for Predictive Maintenance in Industrial Engineering**" by Charlene Magena [3]. This research emphasizes:
- The importance of **hybrid ML models** for improved predictive accuracy.
- The integration of **domain-specific knowledge** for reliability.
- The use of **IoT-enabled real-time data collection** for proactive interventions.
To align with these recommendations, this study implements _ensemble stacking_, a technique that combines the strengths of multiple models to achieve higher performance.

### Ensemble Stacking
**Ensemble stacking** is a machine learning technique where multiple base models are trained independently, and their predictions are used as input features for a higher-level model (called a meta-learner). The meta-learner combines these predictions to make the final decision.
Benefits of ensemble stacking include:
- Improved generalization and accuracy.
- Reduced risk of overfitting.
- The ability to leverage diverse modeling strategies.
In this study, stacking is applied to both the binary and multi-class tasks, using confidence scores from the base models as inputs to the meta-learner.

The following models are used as base learners in both classification tasks:
**Logistic Regression (LR)**: A linear model used for classification, estimating the probability of class membership.
**Support Vector Machine (SVM)**: A model that finds an optimal hyperplane to separate classes in the feature space, often effective in high-dimensional data.
**XGBoost (Extreme Gradient Boosting)**: A powerful and efficient gradient boosting algorithm that builds an ensemble of decision trees, known for its high accuracy and speed.
The predictions (confidence scores) of these models are then passed to a **Logistic Regression meta-learner** that produces the final prediction.

### Dataset
This study uses the **AI4I 2020 Predictive Maintenance Dataset** [1], available on the UCI Machine Learning Repository. The dataset is synthetically generated to reflect real-world manufacturing environments and includes:
10,000 samples and 14 features
Sensor data such as:
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
Machine type and failure types (binary and multi-class labels)
The target labels are:
Binary task: Machine failure (Yes/No)
Multi-class task: Type of failure – Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failures (RNF)

## Architecture Overview
The Predictive Maintenance system is built using modular components in **Azure Machine Learning** for scalable training, inference, and monitoring. Below is a high-level overview of the architecture:
- **Data Ingestion**
Sensor and operational data is ingested from Azure Blob Storage into Azure ML for preprocessing and model training.
- **Preprocessing & Feature Engineering**
A custom Azure ML _command jobs_ handles data cleaning, type conversion, scaling with a fitted StandardScaler, and label encoding for categorical features. The same preprocessing logic is reused during inference.
- **Model Training**
The system trains two types of models:
  - _Binary Classification_: Determines whether a failure will occur (failure or no failure).
  - _Multi-Class Classification_: If a failure is predicted, a second model classifies the specific failure type(e.g., Power, Tool Wear, Overstrain, Heat Dissipation).
- **Model Stacking & Meta-Learning**
Both binary and multi-class pipelines use a _stacked ensemble architecture_:
  - Three base learners: Logistic Regression, SVM, and XGBoost
  - One meta-learner: Logistic Regression (trained on base model confidence scores)
- **Model Wrapping**
A unified MLflow custom model wraps preprocessing logic, base learners, and meta-learners. It handles raw input, transforms it, and outputs a JSON prediction.
- **Deployment**
The wrapped model is deployed as a _real-time REST API via a Managed Online Endpoint_ in Azure ML.
- **Monitoring & Logging**
  - _Azure Application Insights_ captures request/response logs and latency.
  - _MLflow_ tracks experiments, model versions, metrics, and artifacts.
- **Security & Versioning**
  - Endpoint access is secured via _API key authentication_.
  - All models, environments, and artifacts are version-controlled in Azure ML.
