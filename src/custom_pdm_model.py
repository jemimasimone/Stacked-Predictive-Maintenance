import json
import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc
from typing import Dict

class PredictiveMaintenanceModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load scaler
        self.scaler = joblib.load(context.artifacts["scaler"])

        # Load binary base models
        self.binary_models = [
            mlflow.sklearn.load_model(context.artifacts["binary_lr"]),
            mlflow.sklearn.load_model(context.artifacts["binary_xgb"]),
            mlflow.sklearn.load_model(context.artifacts["binary_svc"])
        ]
        self.binary_meta = mlflow.pyfunc.load_model(context.artifacts["binary_meta"])

        # Load multiclass base models
        self.multiclass_models = [
            mlflow.sklearn.load_model(context.artifacts["multi_lr"]),
            mlflow.sklearn.load_model(context.artifacts["multi_xgb"]),
            mlflow.sklearn.load_model(context.artifacts["multi_svc"])
        ]
        self.multiclass_meta = mlflow.pyfunc.load_model(context.artifacts["multi_meta"])

    def preprocess(self, input_data: Dict):
        df = pd.DataFrame([input_data])
        df["Machine type"] = df["Machine type"].map({"L": 0, "M": 1, "H": 2})

        float_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        df[float_cols] = df[float_cols].astype(float)
        df[float_cols] = self.scaler.transform(df[float_cols])

        return df[['Machine type'] + float_cols]

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            parsed_input_dict = model_input
        elif isinstance(model_input, pd.DataFrame):
            if not model_input.empty:
                parsed_input_dict = model_input.iloc[0].to_dict()
            else:
                return {"error": "Invalid input format. Received an empty DataFrame."}
        else:
            return {"error": "Invalid input format. Expected dictionary or DataFrame."}

        # Validate input
        required_keys = ['Machine type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        if not all(k in parsed_input_dict for k in required_keys):
            return {"error": f"Missing one or more required keys in input: {required_keys}. Received keys: {list(parsed_input_dict.keys())}"}

        X_processed = self.preprocess(parsed_input_dict)

        # Predict with binary models
        binary_probs = np.array([m.predict(X_processed) for m in self.binary_models]).reshape(1, -1)
        binary_input = pd.DataFrame({
            "LR": binary_probs[0, 0],
            "XGB": binary_probs[0, 1],
            "SVC": binary_probs[0, 2]
        }, index=[0])

        failure_pred = self.binary_meta.predict(binary_input)[0]
        if failure_pred == 0:
            return {"failure": "no", "failure_type": None}

        # Multiclass prediction
        lr_probs = self.multiclass_models[0].predict_proba(X_processed).flatten()
        xgb_probs = self.multiclass_models[1].predict_proba(X_processed).flatten()
        svc_probs = self.multiclass_models[2].predict_proba(X_processed).flatten()

        # Concatenate and reshape
        # This will naturally result in float64 for all if base model outputs are float64
        probs_concat = np.concatenate([
            lr_probs,
            xgb_probs, # This will now be float64
            svc_probs
        ])

        assert probs_concat.shape == (15,), f"Expected 15 features, got {probs_concat.shape}"

        multiclass_input = pd.DataFrame([probs_concat], columns=[
            "LR_class_0", "LR_class_1", "LR_class_2", "LR_class_3", "LR_class_4",
            "XGB_class_0", "XGB_class_1", "XGB_class_2", "XGB_class_3", "XGB_class_4",
            "SVC_class_0", "SVC_class_1", "SVC_class_2", "SVC_class_3", "SVC_class_4",
        ])

        multiclass_result = self.multiclass_meta.predict(multiclass_input)[0]
        failure_types = {
            0: "Power Failure",
            1: "Overstrain Failure",
            2: "Heat Dissipation Failure",
            3: "Tool Wear Failure"
        }

        return {"failure": "yes", "failure_type": failure_types.get(multiclass_result, "Unknown")}
