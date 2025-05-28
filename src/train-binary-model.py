
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, fbeta_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import time
import argparse
import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

def main(args):
    prepared_data = args.prepared_data

    with mlflow.start_run():
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id

        df = read_data(prepared_data)
        print("Data loaded successfully:")
        print(df.info())

        print("Splitting data...")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

        print("Binary task...")
        lr = LogisticRegression()
        svc = SVC(probability=True)
        xgb = XGBClassifier() 

        clf_binary = [lr,svc,xgb]
        clf_str_binary = ['LR','SVC','XGB'] 

        lr_params_binary = {
                'penalty': ['l2'],
                'C': [0.01, 0.1, 1, 10], 
                'solver': ['lbfgs'], 
                'max_iter': [100, 200, 500],
                'random_state': [0]}
        svc_params_binary = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
        xgb_params_binary = {'n_estimators':[300,500,700],
              'max_depth':[5,7],
              'learning_rate':[0.01,0.1],
              'objective':['binary:logistic']}
        
        params_binary = pd.Series(data=[lr_params_binary, svc_params_binary, xgb_params_binary], index=clf_binary)

        metrics_df, fitted_models, confusion_matrices = fit_base_models(clf_binary, clf_str_binary, X_train, X_val, y_train, y_val)
        plot_model_metrics(metrics_df, "binary_base_model_metrics.png")
        mlflow.log_artifact("binary_base_model_metrics.png")

        for name, cm in confusion_matrices.items():
            plot_confusion_matrix(cm, ['No Failure', 'Failure'], f'confusion_matrix_binary_{name}.png')
            mlflow.log_artifact(f'confusion_matrix_binary_{name}.png')

        fitted_models_binary = {name: tune_and_fit(model, X_train, y_train, params_binary[model]) for model, name in zip(clf_binary, clf_str_binary)}
        for name, model in fitted_models_binary.items():
            input_example = pd.DataFrame(X_test)
            signature = infer_signature(input_example, model.predict(X_test))
            mlflow.sklearn.log_model(model, artifact_path=f"models/{name}", signature=signature)

        for name in fitted_models_binary.keys():
            model_uri = f"runs:/{run_id}/models/{name}"
            registered_model_name = f"predictive_maintenance_binary_{name.lower()}"
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"Registered binary base model: {registered_model_name}")

        base_model_predictions_val_binary = create_base_model_predictions(fitted_models_binary, X_val)

        meta_learner_binary = LogisticRegression()
        meta_learner_binary = train_meta_learner(meta_learner_binary, base_model_predictions_val_binary, y_val)

        base_model_predictions_test_binary = create_base_model_predictions(fitted_models_binary, X_test)
        final_predictions_test_binary = meta_learner_binary.predict(base_model_predictions_test_binary)

        cm_test_stacked_binary, metrics_test_stacked_binary = evaluate_stacked_model(meta_learner_binary, base_model_predictions_test_binary, y_test, final_predictions_test_binary)
        plot_confusion_matrix(cm_test_stacked_binary, ['No Failure', 'Failure'], 'stacked_binary_confusion_matrix.png')
        mlflow.log_artifact("stacked_binary_confusion_matrix.png")

        mlflow.log_metric("accuracy_stacked_binary", metrics_test_stacked_binary['ACC'])
        mlflow.log_metric("auc_stacked_binary", metrics_test_stacked_binary['AUC'])
        mlflow.log_metric("f2_stacked_binary", metrics_test_stacked_binary['F2'])

        mlflow.sklearn.log_model(meta_learner_binary, "models/meta_learner",
                                 signature=infer_signature(base_model_predictions_test_binary, final_predictions_test_binary))
        meta_model_uri = f"runs:/{run_id}/models/meta_learner"
        meta_registered_name = "predictive_maintenance_binary_meta_learner"
        mlflow.register_model(model_uri=meta_model_uri, name=meta_registered_name)
        print(f"Registered binary meta-learner model: {meta_registered_name}")

        y_pred_base_test_binary, cm_dict_base_test_binary, metrics_base_test_binary = predict_and_evaluate(list(fitted_models_binary.values()), X_test, y_test, list(fitted_models_binary.keys()))
        comparison_data_binary = {
            'Model': list(fitted_models_binary.keys()) + ['Stacked'],
            'Accuracy': list(metrics_base_test_binary['ACC']) + [metrics_test_stacked_binary['ACC']],
            'AUC': list(metrics_base_test_binary['AUC']) + [metrics_test_stacked_binary['AUC']],
            'F2': list(metrics_base_test_binary['F2']) + [metrics_test_stacked_binary['F2']]
        }
        comparison_df_binary = pd.DataFrame(comparison_data_binary)
        comparison_df_binary.set_index('Model', inplace=True)
        plot_model_metrics(comparison_df_binary, "binary_models_comparison.png")
        mlflow.log_artifact("binary_models_comparison.png")

def read_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df):
    X = df.drop(columns=['Machine failure', 'Failure type'])
    y = df[['Machine failure', 'Failure type']]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=y['Failure type'], random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure type'], random_state=0)
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_model_metrics(metrics_df, file_name):
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(title='Metric')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def plot_confusion_matrix(cm, labels, file_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(file_name.replace(".png", ""))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def eval_preds(model, X, y_true):
    y_true_binary = y_true['Machine failure']
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true_binary, y_pred)
    proba = model.predict_proba(X)[:, 1]
    metrics = pd.Series(dtype='float64')
    metrics['ACC'] = accuracy_score(y_true_binary, y_pred)
    metrics['AUC'] = roc_auc_score(y_true_binary, proba)
    metrics['F1'] = f1_score(y_true_binary, y_pred, pos_label=1)
    metrics['F2'] = fbeta_score(y_true_binary, y_pred, pos_label=1, beta=2)
    return cm, round(metrics, 3)

def fit_base_models(clf_list, clf_names, X_train, X_val, y_train, y_val):
    metrics_df = pd.DataFrame(columns=clf_names)
    fitted_models = {}
    confusion_matrices = {}
    for model, name in zip(clf_list, clf_names):
        model.fit(X_train, y_train['Machine failure'])
        fitted_models[name] = model
        cm, scores = eval_preds(model, X_val, y_val)
        metrics_df[name] = scores
        confusion_matrices[name] = cm
    return metrics_df.T, fitted_models, confusion_matrices

def tune_and_fit(clf, X, y, params):
    scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    grid_model = GridSearchCV(clf, param_grid=params, cv=5, scoring=scorer)
    grid_model.fit(X, y['Machine failure'])
    print(f'Best params for {clf.__class__.__name__}: {grid_model.best_params_}')
    return grid_model

def create_base_model_predictions(fitted_models, X):
    preds = {model_name: model.predict_proba(X)[:, 1] for model_name, model in fitted_models.items()}
    return pd.DataFrame(preds)

def train_meta_learner(meta_learner, base_predictions, y_true):
    meta_learner.fit(base_predictions, y_true['Machine failure'])
    return meta_learner

def evaluate_stacked_model(meta_learner, base_predictions, y_true, y_pred):
    cm, metrics = eval_preds(meta_learner, base_predictions, y_true)
    print("--- Stacked Model Test Set Evaluation (Binary) ---")
    print(metrics)
    return cm, metrics

def predict_and_evaluate(fitted_models, X, y_true, clf_str):
    cm_dict = {}
    metrics_df = pd.DataFrame(columns=clf_str)
    y_pred_df = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models, clf_str):
        y_pred = fit_model.predict(X)
        y_pred_df[model_name] = y_pred
        cm, scores = eval_preds(fit_model, X, y_true)
        cm_dict[model_name] = cm
        metrics_df[model_name] = scores
    return y_pred_df, cm_dict, metrics_df.T

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_data", type=str, help="Path to the prepared training dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
