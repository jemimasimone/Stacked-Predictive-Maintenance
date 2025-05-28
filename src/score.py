import os
import json
import logging
import pandas as pd
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
model = None

def init():
    global model

    try:
        base_model_dir = os.getenv("AZUREML_MODEL_DIR")

        mlflow_model_path = os.path.join(base_model_dir, "predictive_maintenance_model")

        logging.info(f"Attempting to load MLflow model from: {mlflow_model_path}")

        model = mlflow.pyfunc.load_model(mlflow_model_path)

        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def run(raw_data):
    try:
        logging.info("Received request data.")

        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        elif isinstance(raw_data, dict):
            data = raw_data
        else:
            return json.dumps({"error": "Invalid input format. Expected JSON string or dictionary."})

        result = model.predict(data)

        logging.info("Prediction successful.")
        return json.dumps(result)
    except Exception as e:
        error_message = f"Error during inference: {e}"
        logging.error(error_message)
        return json.dumps({"error": error_message})
