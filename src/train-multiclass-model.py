
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score, make_scorer
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

        print("Defining multi-class models...")
        lr = LogisticRegression()
        svc = SVC(decision_function_shape='ovr', probability=True)
        xgb = XGBClassifier()

        clf_multi = [lr, svc, xgb]
        clf_str_multi = ['LR', 'SVC', 'XGB']

        print("Defining hyperparameters...")
        lr_params_multi = {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs'],
            'multi_class': ['multinomial'],
            'max_iter': [100, 200, 500],
            'random_state': [0]}
        svc_params_multi = {
            'C': [1, 10, 100],
            'gamma': [0.1, 1],
            'kernel': ['rbf'],
            'random_state': [0]}
        xgb_params_multi  = {'n_estimators':[100,300,500],
              'max_depth':[5,7,10],
              'learning_rate':[0.01,0.1],
              'objective':['multi:softprob']}
        params_multi = pd.Series(data=[lr_params_multi, svc_params_multi, xgb_params_multi], index=clf_multi)

        failure_labels = ['No Failure', 'PWF', 'OSF', 'HDF', 'TWF']
        metrics_df, fitted_models, confusion_matrices = fit_base_models(clf_multi, clf_str_multi, X_train, X_val, y_train, y_val, failure_labels)
        print(f"Base Model Metrics: {metrics_df}")
        plot_model_metrics(metrics_df, "multiclass_base_model_metrics.png")
        mlflow.log_artifact("multiclass_base_model_metrics.png")

        for name, cm in confusion_matrices.items():
            plot_confusion_matrix(cm, failure_labels, f'confusion_matrix_multi_{name}.png')
            mlflow.log_artifact(f'confusion_matrix_multi_{name}.png')

        failure_label_map = {0: 'NoFailure', 1: 'PWF', 2: 'OSF', 3: 'HDF', 4: 'TWF',}
        print("Tuning hyperparameters...")
        fitted_models_multi = {name: tune_and_fit(model, X_train, y_train, params_multi[model]) for model, name in zip(clf_multi, clf_str_multi)}
        for name, model in fitted_models_multi.items():
            input_example = pd.DataFrame(X_test)
            signature = infer_signature(input_example, model.predict(X_test))
            mlflow.sklearn.log_model(model, artifact_path=f"models/{name}", signature=signature)

        for name in fitted_models_multi.keys():
            model_uri = f"runs:/{run_id}/models/{name}"
            registered_model_name = f"predictive_maintenance_multi_{name.lower()}"
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"Registered multiclass model: {registered_model_name}")

        print("Creating predictions...")
        base_model_predictions_val_multi = create_base_model_predictions(fitted_models_multi, X_val, failure_label_map)

        print("Training meta learner...")
        meta_learner_multi = LogisticRegression()
        meta_learner_multi = train_meta_learner(meta_learner_multi, base_model_predictions_val_multi, y_val,)
        base_model_predictions_test_multi = create_base_model_predictions(fitted_models_multi, X_test, failure_label_map)

        final_predictions_test_multi = meta_learner_multi.predict(base_model_predictions_test_multi)
        cm_test_stacked_multi, metrics_test_stacked_multi = evaluate_stacked_model(meta_learner_multi, base_model_predictions_test_multi, y_test, final_predictions_test_multi, failure_labels)
        plot_confusion_matrix(cm_test_stacked_multi, failure_labels, f'stacked_multi_confusion_matrix.png')
        mlflow.log_artifact("stacked_multi_confusion_matrix.png")

        mlflow.log_metric("accuracy_stacked_multi", metrics_test_stacked_multi['ACC'])
        mlflow.log_metric("f2_stacked_multi", metrics_test_stacked_multi['F2'])

        mlflow.sklearn.log_model(meta_learner_multi, "models/meta_learner",
                                 signature=infer_signature(base_model_predictions_test_multi, final_predictions_test_multi))
        meta_model_uri = f"runs:/{run_id}/models/meta_learner"
        meta_registered_name = "predictive_maintenance_multiclass_meta_learner"
        mlflow.register_model(model_uri=meta_model_uri, name=meta_registered_name)
        print(f"Registered multiclass meta-learner model: {meta_registered_name}")

        y_pred_base_test_multi, cm_dict_base_test_multi, metrics_base_test_multi = predict_and_evaluate(list(fitted_models_multi.values()), X_test, y_test, list(fitted_models_multi.keys()), failure_labels)
        comparison_data_multi = {
            'Model': list(fitted_models_multi.keys()) + ['Stacked'],
            'Accuracy': list(metrics_base_test_multi['ACC']) + [metrics_test_stacked_multi['ACC']],
            'F2': list(metrics_base_test_multi['F2']) + [metrics_test_stacked_multi['F2']]
        }
        comparison_df_multi = pd.DataFrame(comparison_data_multi)
        print(comparison_df_multi)
        comparison_df_multi.set_index('Model', inplace=True)
        plot_model_metrics(comparison_df_multi, "multiclass_models_comparison.png")
        mlflow.log_artifact("multiclass_models_comparison.png")

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

def eval_preds(model, X, y_true, failure_labels):
    y_true_multi = y_true['Failure type']
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true_multi, y_pred, labels=range(len(failure_labels)))
    metrics = pd.Series(dtype='float64')
    metrics['ACC'] = accuracy_score(y_true_multi, y_pred)
    metrics['F1'] = f1_score(y_true_multi, y_pred, average='weighted')
    metrics['F2'] = fbeta_score(y_true_multi, y_pred, beta=2, average='weighted')

    return cm, round(metrics, 3)

def fit_base_models(clf_list, clf_names, X_train, X_val, y_train, y_val, failure_labels):
    metrics_df = pd.DataFrame(columns=clf_names)
    fitted_models = {}
    confusion_matrices = {}
    for model, name in zip(clf_list, clf_names):
        model.fit(X_train, y_train['Failure type'])
        fitted_models[name] = model
        cm, scores = eval_preds(model, X_val, y_val, failure_labels)
        metrics_df[name] = scores
        confusion_matrices[name] = cm
    return metrics_df.T, fitted_models, confusion_matrices

def tune_and_fit(clf, X, y, params):
    scorer =  make_scorer(fbeta_score, beta=2, average='weighted')
    start_time = time.time()
    grid_model = GridSearchCV(clf, param_grid=params, cv=5, scoring=scorer)
    grid_model.fit(X, y['Failure type'])
    print(f'Best params for {clf.__class__.__name__}: {grid_model.best_params_}')
    train_time = time.time() - start_time
    mins = int(train_time // 60)
    print(f'Training time: {mins}m {round(train_time - mins * 60)}s')
    return grid_model

def create_base_model_predictions(fitted_models, X, failure_label_map):
    predictions = pd.DataFrame()
    for model_name, model in fitted_models.items():
        preds = model.predict_proba(X)
        preds = preds.astype(np.float64)

        for i in range(preds.shape[1]):
            class_name = failure_label_map.get(i, f'class_{i}')
            predictions[f'{model_name}_class_{i}'] = preds[:, i]
    return predictions

def train_meta_learner(meta_learner, base_predictions, y_true):
    meta_learner.fit(base_predictions, y_true['Failure type'])
    return meta_learner

def evaluate_stacked_model(meta_learner, base_predictions, y_true, y_pred, failure_labels):
    cm, metrics = eval_preds(meta_learner, base_predictions, y_true, failure_labels)
    print("--- Stacked Model Test Set Evaluation (Multi-class) ---")
    print(metrics)
    return cm, metrics

def predict_and_evaluate(fitted_models, X, y_true, clf_str, failure_labels):
    cm_dict = {}
    metrics_df = pd.DataFrame(columns=clf_str)
    y_pred_df = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models, clf_str):
        y_pred = fit_model.predict(X)
        y_pred_df[model_name] = y_pred
        cm, scores = eval_preds(fit_model, X, y_true, failure_labels)
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
