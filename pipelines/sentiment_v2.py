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
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    print(f"🎯 Model classes: {model.classes_}")
    print(f"🔢 Number of features: {model.n_features_in_}")
    print(f"⚙️  Model coefficients shape: {model.coef_.shape}")
    
    # Test prediction capability
    import numpy as np
    dummy_input = np.random.random((1, model.n_features_in_))
    prediction = model.predict(dummy_input)
    probability = model.predict_proba(dummy_input)
    
    print(f"🧪 Test prediction: {prediction[0]}")
    print(f"🎲 Test probabilities: {probability[0]}")
    print("✅ Model verification completed successfully!")

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

if __name__ == '__main__':
    # Compile the pipeline
    from kfp import compiler
    compiler.Compiler().compile(sentiment_pipeline, 'sentiment_pipeline_output.yaml')
    print("Pipeline compiled successfully!")
#with input and output