# pipeline.py: Convert notebook logic into a Kubeflow pipeline

import kfp
from kfp import dsl
from kfp.components import create_component_from_func_v2 as create_component_from_func
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def train_model():
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Save model
    model_path = "/tmp/model.pkl"
    joblib.dump(model, model_path)

    # Log with MLflow (optional)
    mlflow.set_experiment("iris-classification")
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 100, "max_depth": 3})
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)

    return model_path

# Convert to KFP component
train_op = create_component_from_func(train_model, output_component_file='train_component.yaml', base_image='python:3.8')

@dsl.pipeline(
    name='Iris Classifier Pipeline',
    description='A simple ML pipeline to train and log a model using Kubeflow'
)
def iris_pipeline():
    train_step = train_op()

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')

