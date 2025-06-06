#!/usr/bin/env python3

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline

# Simplified Component 1: Data Generation with lighter resources
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3", "scikit-learn==1.3.0", "numpy==1.21.6"]
)
def generate_simple_data() -> str:
    """Generate simple dataset and return as string"""
    import pandas as pd
    from sklearn.datasets import make_classification
    import json
    
    # Generate smaller dataset
    X, y = make_classification(
        n_samples=100,  # Reduced size
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    
    # Convert to simple format
    data = {
        'X': X.tolist(),
        'y': y.tolist(),
        'feature_names': [f'feature_{i}' for i in range(4)]
    }
    
    print("Generated dataset with 100 samples")
    print(f"Features: {data['feature_names']}")
    
    return json.dumps(data)

# Simplified Component 2: Train Model
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3", "scikit-learn==1.3.0", "numpy==1.21.6"]
)
def train_simple_model(data_json: str) -> float:
    """Train model and return accuracy"""
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Load data
    data = json.loads(data_json)
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Simple model with minimal resources
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

# Simplified Component 3: Validate and Deploy
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pyyaml==6.0"]
)
def validate_and_create_deployment(accuracy: float, model_name: str = "demo-model") -> str:
    """Validate model and create deployment config"""
    import yaml
    
    # Validation
    threshold = 0.6
    if accuracy >= threshold:
        status = "PASSED"
        print(f"✅ Validation PASSED: {accuracy:.4f} >= {threshold}")
    else:
        status = "FAILED"
        print(f"❌ Validation FAILED: {accuracy:.4f} < {threshold}")
        return f"Validation failed with accuracy {accuracy:.4f}"
    
    # Create simple KServe config
    config = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": "default"
        },
        "spec": {
            "predictor": {
                "sklearn": {
                    "storageUri": f"gs://your-bucket/models/{model_name}",
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "200m", "memory": "256Mi"}
                    }
                }
            }
        }
    }
    
    config_yaml = yaml.dump(config, default_flow_style=False)
    
    result = f"""
🎉 Pipeline Completed Successfully!

📊 Results:
- Model Accuracy: {accuracy:.4f}
- Validation Status: {status}
- Model Name: {model_name}

🚀 KServe Deployment Config:
{config_yaml}

📝 Next Steps:
1. Save the above YAML to a file
2. Apply with: kubectl apply -f deployment.yaml
3. Check status: kubectl get inferenceservice {model_name}
"""
    
    print(result)
    return result

# Simple Pipeline Definition
@pipeline(
    name="simple-demo-pipeline",
    description="Simplified ML pipeline that actually works"
)
def simple_demo_pipeline(model_name: str = "demo-classifier"):
    """
    Simplified pipeline focusing on execution rather than complexity
    """
    
    # Step 1: Generate Data
    data_task = generate_simple_data()
    data_task.set_display_name("Generate Data")
    data_task.set_cpu_limit("200m")
    data_task.set_memory_limit("256Mi")
    
    # Step 2: Train Model  
    train_task = train_simple_model(data_json=data_task.output)
    train_task.set_display_name("Train Model")
    train_task.set_cpu_limit("300m")
    train_task.set_memory_limit("512Mi")
    train_task.after(data_task)
    
    # Step 3: Validate and Deploy
    deploy_task = validate_and_create_deployment(
        accuracy=train_task.output,
        model_name=model_name
    )
    deploy_task.set_display_name("Validate & Create Deployment")
    deploy_task.set_cpu_limit("200m")
    deploy_task.set_memory_limit("256Mi")
    deploy_task.after(train_task)

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=simple_demo_pipeline,
        package_path="simple_demo_pipeline.yaml"
    )
    print("✅ Simple pipeline compiled to 'simple_demo_pipeline.yaml'")
    print("\n🚀 This version should work reliably!")
    print("\nUpload simple_demo_pipeline.yaml to Kubeflow and run it.")
