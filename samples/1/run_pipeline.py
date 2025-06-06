import kfp
from sentiment_pipeline_v1 import sentiment_pipeline

def main():
    # Your Kubeflow endpoint
    kf_endpoint = "http://localhost:8080"
    #kf_endpoint = "http://a8086413e981e443fa4cd1807257dd2b-1397876859.us-west-2.elb.amazonaws.com/"
    auth_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcmdvY2QiLCJzdWIiOiJhZG1pbjpsb2dpbiIsImV4cCI6MTc0OTIwNzAyNywibmJmIjoxNzQ5MTIwNjI3LCJpYXQiOjE3NDkxMjA2MjcsImp0aSI6ImY5OTQ3ZjllLTA3YTQtNGY0MC1hNTVhLTk3OTlhYTljMzU0OSJ9.A0HtMJDKJl6Hn5GIHy0BrJOPRfUav-zd8NK8EN69uHo"
    try:
        # Connect to Kubeflow Pipelines
        client = kfp.Client(
            host=kf_endpoint,
            existing_token=auth_token)
        
        # Test connection
        print("Testing connection to Kubeflow...")
        
        # Compile pipeline
        print("Compiling pipeline...")
        from kfp import compiler
        compiler.Compiler().compile(sentiment_pipeline, 'sentiment_pipeline.yaml')
        print("✅ Pipeline compiled!")
        
        # Submit pipeline run
        print("Submitting pipeline run...")
        run = client.create_run_from_pipeline_func(
            sentiment_pipeline,
            run_name='sentiment-analysis-run',
            experiment_name='sentiment-analysis-experiment'
        )
        
        print(f"✅ Pipeline submitted successfully!")
        print(f"Run ID: {run.run_id}")
        print(f"Monitor at: {kf_endpoint}/#/runs/details/{run.run_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure kubectl port-forward is running")
        print("2. Check if Kubeflow is accessible at the endpoint")
        print("3. Verify your EKS cluster connection")

if __name__ == "__main__":
    main()