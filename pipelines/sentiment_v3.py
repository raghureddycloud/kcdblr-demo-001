from kfp import dsl
from kfp.dsl import component, pipeline, Output, Input, Dataset
import kfp

# Component 1: Data Preparation
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def prepare_data(processed_data: Output[Dataset]):
    """Simulate loading and preparing review data"""
    import pandas as pd
    import pickle
    
    # Sample data
    data = {
        'review': [
            'This product is amazing!',
            'Terrible quality, waste of money',
            'Good value for money',
            'Poor customer service',
            'Excellent product, highly recommend',
            'Not worth the price',
            'Great experience overall',
            'Disappointed with purchase'
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Save processed data to the output path provided by KFP
    with open(processed_data.path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Processed {len(df)} reviews")

# Component 2: Feature Engineering
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def extract_features(
    processed_data: Input[Dataset], 
    features_X: Output[Dataset],
    features_y: Output[Dataset],
    vectorizer_model: Output[Dataset]
):
    """Extract features from text data"""
    import pandas as pd
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Load data from the input provided by KFP
    with open(processed_data.path, 'rb') as f:
        df = pickle.load(f)
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    
    # Save outputs to paths provided by KFP
    with open(features_X.path, 'wb') as f:
        pickle.dump(X, f)
    with open(features_y.path, 'wb') as f:
        pickle.dump(y, f)
    with open(vectorizer_model.path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Extracted features with shape: {X.shape}")

# Component 3: Model Training
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_model(
    features_X: Input[Dataset],
    features_y: Input[Dataset],
    trained_model: Output[Dataset],
    test_X: Output[Dataset],
    test_y: Output[Dataset]
):
    """Train sentiment classification model"""
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    with open(features_X.path, 'rb') as f:
        X = pickle.load(f)
    with open(features_y.path, 'rb') as f:
        y = pickle.load(f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Save outputs
    with open(trained_model.path, 'wb') as f:
        pickle.dump(model, f)
    with open(test_X.path, 'wb') as f:
        pickle.dump(X_test, f)
    with open(test_y.path, 'wb') as f:
        pickle.dump(y_test, f)
    
    print("Model training completed")

# Component 4: Model Evaluation
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def evaluate_model(
    trained_model: Input[Dataset],
    test_X: Input[Dataset],
    test_y: Input[Dataset],
    evaluation_metrics: Output[Dataset]
):
    """Evaluate model performance"""
    import pickle
    from sklearn.metrics import accuracy_score
    import json
    
    with open(trained_model.path, 'rb') as f:
        model = pickle.load(f)
    with open(test_X.path, 'rb') as f:
        X_test = pickle.load(f)
    with open(test_y.path, 'rb') as f:
        y_test = pickle.load(f)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {'accuracy': accuracy}
    
    with open(evaluation_metrics.path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model Accuracy: {accuracy:.2f}")


@component(
    base_image="python:3.9-slim", 
    packages_to_install=["scikit-learn"]
)
def verify_model(trained_model: Input[Dataset]):
    """Verify the model was saved correctly"""
    import pickle
    
    # Load the model
    with open(trained_model.path, 'rb') as f:
        model = pickle.load(f)
    
    # Check model properties
    print(f" Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Model classes: {model.classes_}")
    print(f"Number of features: {model.n_features_in_}")
    print(f"Model coefficients shape: {model.coef_.shape}")
    
    # Test prediction capability
    import numpy as np
    dummy_input = np.random.random((1, model.n_features_in_))
    prediction = model.predict(dummy_input)
    probability = model.predict_proba(dummy_input)
    
    print(f"Test prediction: {prediction[0]}")
    print(f"Test probabilities: {probability[0]}")
    print("Model verification completed successfully!")


@component(
    base_image="python:3.9-slim",
    packages_to_install=["kubernetes>=20.0.0,<26.0.0", "pyyaml==6.0.1", "scikit-learn"]
)
def deploy_to_kserve(
    trained_model: Input[Dataset],
    vectorizer_model: Input[Dataset],
    deployment_status: Output[Dataset]
):
    """Deploy model to KServe in app-dev-001 namespace"""
    import pickle
    import yaml
    import json
    import base64
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    import time
    
    try:
        # Load Kubernetes config (assumes in-cluster config)
        config.load_incluster_config()
    except:
        # Fallback to local config for testing
        config.load_kube_config()
    
    v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()
    
    # Load model artifacts
    with open(trained_model.path, 'rb') as f:
        model_data = pickle.load(f)
    with open(vectorizer_model.path, 'rb') as f:
        vectorizer_data = pickle.load(f)
    
    # Serialize models to bytes for ConfigMap
    model_bytes = pickle.dumps(model_data)
    vectorizer_bytes = pickle.dumps(vectorizer_data)
    
    # Encode as base64 for ConfigMap storage
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    vectorizer_b64 = base64.b64encode(vectorizer_bytes).decode('utf-8')
    
    namespace = "app-dev-001"
    configmap_name = "sentiment-model-artifacts"
    service_name = "sentiment-classifier"
    
    # Create namespace if it doesn't exist
    try:
        v1.read_namespace(name=namespace)
        print(f"Namespace {namespace} already exists")
    except ApiException as e:
        if e.status == 404:
            namespace_body = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=namespace)
            )
            v1.create_namespace(body=namespace_body)
            print(f"Created namespace {namespace}")
        else:
            raise e
    
    # Create/Update ConfigMap with model artifacts
    configmap_body = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(
            name=configmap_name,
            namespace=namespace
        ),
        binary_data={
            "sentiment_model.pkl": model_b64,
            "vectorizer.pkl": vectorizer_b64
        }
    )
    
    try:
        v1.replace_namespaced_config_map(
            name=configmap_name,
            namespace=namespace,
            body=configmap_body
        )
        print(f"Updated ConfigMap {configmap_name}")
    except ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(
                namespace=namespace,
                body=configmap_body
            )
            print(f"Created ConfigMap {configmap_name}")
        else:
            raise e
    
    # Create InferenceService
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": service_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": "243571642843.dkr.ecr.us-west-2.amazonaws.com/sentiment-predictor:latest",
                        "ports": [
                            {
                                "containerPort": 8080,
                                "protocol": "TCP"
                            }
                        ],
                        "volumeMounts": [
                            {
                                "name": "model-storage",
                                "mountPath": "/mnt/models"
                            }
                        ]
                    }
                ],
                "volumes": [
                    {
                        "name": "model-storage",
                        "configMap": {
                            "name": configmap_name
                        }
                    }
                ]
            }
        }
    }
    
    # Deploy InferenceService
    try:
        custom_api.replace_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=service_name,
            body=inference_service
        )
        print(f"Updated InferenceService {service_name}")
    except ApiException as e:
        if e.status == 404:
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
            print(f"Created InferenceService {service_name}")
        else:
            raise e
    
    # Wait for deployment to be ready (basic check)
    max_wait = 300  # 5 minutes
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            service_status = custom_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name
            )
            
            status = service_status.get("status", {})
            conditions = status.get("conditions", [])
            
            for condition in conditions:
                if condition.get("type") == "Ready" and condition.get("status") == "True":
                    print(f"InferenceService {service_name} is ready!")
                    deployment_result = {
                        "status": "success",
                        "service_name": service_name,
                        "namespace": namespace,
                        "endpoint": f"http://{service_name}.{namespace}.svc.cluster.local/v1/models/{service_name}:predict",
                        "ready_time": wait_time
                    }
                    
                    with open(deployment_status.path, 'w') as f:
                        json.dump(deployment_result, f)
                    
                    return
            
            print(f"Waiting for service to be ready... ({wait_time}s)")
            time.sleep(10)
            wait_time += 10
            
        except ApiException as e:
            print(f"Error checking service status: {e}")
            time.sleep(10)
            wait_time += 10
    
    # Timeout reached
    deployment_result = {
        "status": "timeout",
        "service_name": service_name,
        "namespace": namespace,
        "message": f"Service deployment timed out after {max_wait} seconds"
    }
    
    with open(deployment_status.path, 'w') as f:
        json.dump(deployment_result, f)


# Define the Pipeline
@pipeline(
    name='sentiment-analysis-pipeline',
    description='End-to-end sentiment analysis pipeline'
)
def sentiment_pipeline():
    """Complete ML pipeline for sentiment analysis"""
    
    # Step 1: Prepare data
    data_task = prepare_data()
    
    # Step 2: Extract features
    features_task = extract_features(processed_data=data_task.outputs['processed_data'])
    
    # Step 3: Train model
    training_task = train_model(
        features_X=features_task.outputs['features_X'],
        features_y=features_task.outputs['features_y']
    )
    
    # Step 4: Evaluate model
    evaluation_task = evaluate_model(
        trained_model=training_task.outputs['trained_model'],
        test_X=training_task.outputs['test_X'],
        test_y=training_task.outputs['test_y']
    )

    # Step 5: verify_model
    verify_task = verify_model(
        trained_model=training_task.outputs['trained_model']
    )
    verify_task.after(evaluation_task)
    
    # Step 6 : Add KServe Deployment 
    deploy_task = deploy_to_kserve(
        trained_model=training_task.outputs['trained_model'],
        vectorizer_model=features_task.outputs['vectorizer_model']    
    )
    deploy_task.after(verify_task)

if __name__ == '__main__':
    # Compile the pipeline
    from kfp import compiler
    compiler.Compiler().compile(sentiment_pipeline, 'sentiment_pipeline_v3.yaml')
    print("Pipeline compiled successfully!")
# Verify Model