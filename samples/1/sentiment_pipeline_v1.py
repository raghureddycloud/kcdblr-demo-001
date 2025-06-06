from kfp import dsl
from kfp.dsl import component, pipeline
import kfp

# Component 1: Data Preparation
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def prepare_data() -> str:
    """Simulate loading and preparing review data"""
    import pandas as pd
    import pickle
    import os
    
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
    
    # Save processed data
    os.makedirs('/tmp/data', exist_ok=True)
    with open('/tmp/data/processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Processed {len(df)} reviews")
    return '/tmp/data/processed_data.pkl'

# Component 2: Feature Engineering
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def extract_features(data_path: str) -> str:
    """Extract features from text data"""
    import pandas as pd
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    
    os.makedirs('/tmp/features', exist_ok=True)
    with open('/tmp/features/X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('/tmp/features/y.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open('/tmp/features/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Extracted features with shape: {X.shape}")
    return '/tmp/features'

# Component 3: Model Training
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_model(features_path: str) -> str:
    """Train sentiment classification model"""
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import os
    
    with open(f'{features_path}/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(f'{features_path}/y.pkl', 'rb') as f:
        y = pickle.load(f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs('/tmp/model', exist_ok=True)
    with open('/tmp/model/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('/tmp/model/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('/tmp/model/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    print("Model training completed")
    return '/tmp/model'

# Component 4: Model Evaluation
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def evaluate_model(model_path: str) -> str:
    """Evaluate model performance"""
    import pickle
    from sklearn.metrics import accuracy_score
    import json
    import os
    
    with open(f'{model_path}/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_path}/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'{model_path}/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {'accuracy': accuracy}
    
    os.makedirs('/tmp/metrics', exist_ok=True)
    with open('/tmp/metrics/evaluation.json', 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    return '/tmp/metrics/evaluation.json'

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
    features_task = extract_features(data_path=data_task.output)
    
    # Step 3: Train model
    training_task = train_model(features_path=features_task.output)
    
    # Step 4: Evaluate model
    evaluation_task = evaluate_model(model_path=training_task.output)

    # Step 5: Docker
    

if __name__ == '__main__':
    # Compile the pipeline
    from kfp import compiler
    compiler.Compiler().compile(sentiment_pipeline, 'sentiment_pipeline.yaml')
    print("Pipeline compiled successfully!")