
#!/usr/bin/env python3

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Output, Input, Model, Dataset
from typing import NamedTuple

# Component 1: Data Generation/Loading
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def generate_data(dataset: Output[Dataset]):
    """Generate sample data for training"""
    import pandas as pd
    from sklearn.datasets import make_classification
    import numpy as np
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
        shuffle=False
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    df['target'] = y
    
    # Save dataset
    df.to_csv(dataset.path, index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"Features: {df.columns.tolist()}")

# Component 2: Model Training
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"]
)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model]
) -> NamedTuple('Outputs', [('accuracy', float), ('model_name', str)]):
    """Train a simple classification model"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    
    # Load data
    df = pd.read_csv(dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    model_path = os.path.join(model.path, "model.joblib")
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(clf, model_path)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    from collections import namedtuple
    Outputs = namedtuple('Outputs', ['accuracy', 'model_name'])
    return Outputs(accuracy, "demo-classifier")

# Component 3: Model Validation
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def validate_model(
    dataset: Input[Dataset],
    model: Input[Model],
    accuracy_threshold: float = 0.7
) -> str:
    """Validate model performance"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    import os
    
    # Load data and model
    df = pd.read_csv(dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    model_path = os.path.join(model.path, "model.joblib")
    clf = joblib.load(model_path)
    
    # Use test split for validation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if accuracy >= accuracy_threshold:
        result = "PASSED"
        print(f"✅ Model validation PASSED (accuracy {accuracy:.4f} >= {accuracy_threshold})")
    else:
        result = "FAILED"
        print(f"❌ Model validation FAILED (accuracy {accuracy:.4f} < {accuracy_threshold})")
    
    return result

# Component 4: Create KServe InferenceService
@component(
    base_image="python:3.9",
    packages_to_install=["kubernetes", "pyyaml"]
)
def deploy_model_kserve(
    model: Input[Model],
    model_name: str,
    namespace: str = "default"
) -> str:
    """Deploy model using KServe"""
    import yaml
    import os
    
    # Create KServe InferenceService YAML
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "sklearn": {
                    "storageUri": f"pvc://models/{model_name}",
                    "resources": {
                        "limits": {
                            "cpu": "100m",
                            "memory": "256Mi"
                        },
                        "requests": {
                            "cpu": "50m",
                            "memory": "128Mi"
                        }
                    }
                }
            }
        }
    }
    
    # Save the YAML file
    yaml_path = "/tmp/inference_service.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(inference_service, f, default_flow_style=False)
    
    print(f"Created KServe InferenceService YAML:")
    print(yaml.dump(inference_service, default_flow_style=False))
    
    # In a real scenario, you would apply this YAML to the cluster
    # kubectl apply -f /tmp/inference_service.yaml
    
    deployment_info = f"""
Deployment Configuration Created:
- Model Name: {model_name}
- Namespace: {namespace}
- Framework: SKLearn
- Resources: 50-100m CPU, 128-256Mi Memory

To deploy manually, run:
kubectl apply -f {yaml_path}

Or use the KServe Python SDK in your cluster.
"""
    
    return deployment_info

# Main Pipeline Definition
@pipeline(
    name="demo-ml-pipeline",
    description="Simple ML pipeline demonstrating Kubeflow benefits with KServe deployment"
)
def demo_ml_pipeline(
    accuracy_threshold: float = 0.7,
    model_name: str = "demo-classifier",
    namespace: str = "default"
):
    """
    Demo pipeline showing Kubeflow benefits:
    1. Reproducible ML workflows
    2. Component reusability
    3. Automatic orchestration
    4. Easy scaling and monitoring
    5. Integrated deployment with KServe
    """
    
    # Step 1: Generate/Load Data
    data_task = generate_data()
    data_task.set_display_name("Data Generation")
    
    # Step 2: Train Model
    train_task = train_model(dataset=data_task.outputs["dataset"])
    train_task.set_display_name("Model Training")
    train_task.after(data_task)
    
    # Step 3: Validate Model
    validate_task = validate_model(
        dataset=data_task.outputs["dataset"],
        model=train_task.outputs["model"],
        accuracy_threshold=accuracy_threshold
    )
    validate_task.set_display_name("Model Validation")
    validate_task.after(train_task)
    
    # Step 4: Deploy with KServe (conditional on validation)
    deploy_task = deploy_model_kserve(
        model=train_task.outputs["model"],
        model_name=model_name,
        namespace=namespace
    )
    deploy_task.set_display_name("KServe Deployment")
    deploy_task.after(validate_task)
    
    # Add conditions - only deploy if validation passes
    # Note: In real scenarios, you'd add conditional logic here

# Compile and run the pipeline
if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=demo_ml_pipeline,
        package_path="demo_ml_pipeline.yaml"
    )
    print("Pipeline compiled successfully to 'demo_ml_pipeline.yaml'")
    
    # Example of how to run (uncomment when ready to execute)
    """
    # Connect to Kubeflow
    client = kfp.Client(host='<your-kubeflow-host>')
    
    # Create experiment
    experiment = client.create_experiment('demo-ml-experiment')
    
    # Run pipeline
    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name='demo-ml-run',
        pipeline_package_path='demo_ml_pipeline.yaml',
        params={
            'accuracy_threshold': 0.75,
            'model_name': 'demo-classifier-v1',
            'namespace': 'kubeflow-user-example-com'
        }
    )
    
    print(f"Pipeline run created: {run}")
    """