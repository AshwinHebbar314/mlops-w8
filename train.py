import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dvc.api
from pygit2 import Repository
import os   # NEW IMPORT for creating directories
import json # NEW IMPORT for writing the JSON file

# Get the current Git branch name to use as a parameter
# This will help identify the run in MLflow UI
try:
    branch_name = Repository('.').head.shorthand
except Exception:
    branch_name = "local" # Fallback if not in a git repo

# Set the MLflow tracking URI to your remote server
mlflow.set_tracking_uri('http://34.67.110.16:8100')

def train_model():
    """
    Trains a model on the current version of the data, logs to MLflow,
    and saves a metrics file for CML.
    """
    print("--- Starting training run ---")
    
    # Use DVC API to get the path to the current data from GCS
    print("Fetching data from DVC remote 'gcsb'...")
    try:
        train_path = dvc.api.get_url('data/train.csv', remote='gcsb')
        test_path = dvc.api.get_url('data/test.csv', remote='gcsb')
    except Exception as e:
        print(f"Error fetching data from DVC: {e}")
        # In a CI environment, it's better to exit if data isn't found
        # In a local run, this might be okay.
        return 

    print("Data loaded. Preparing training and test sets...")
    # The column in the original Iris dataset is 'target', but let's use 'species' as in your file
    X_train = pd.read_csv(train_path).drop('species', axis=1)
    y_train = pd.read_csv(train_path)['species']
    X_test = pd.read_csv(test_path).drop('species', axis=1)
    y_test = pd.read_csv(test_path)['species']

    # This context manager will log to your remote MLflow server
    with mlflow.start_run(run_name=f"run-{branch_name}"):
        print(f"MLflow run started for branch: {branch_name}")

        # Use autolog for convenience, it captures params, metrics, and the model
        mlflow.autolog()
        
        # Manually log the branch name for easy filtering in the MLflow UI
        mlflow.log_param("branch_name", branch_name)

        # Train the model
        print("Training Logistic Regression model...")
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        model.fit(X_train, y_train)

        # Evaluate the model
        print("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

        # Manually log the primary metric (autolog will also capture this)
        mlflow.log_metric("accuracy", accuracy)

        # --- CML METRICS FILE GENERATION ---
        # This is the new section that creates the file for the CML workflow
        print("Saving metrics for CML report...")
        
        # 1. Define the directory and create it if it doesn't exist
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # 2. Structure the metrics into a dictionary
        metrics_data = {'accuracy': accuracy}
        
        # 3. Write the dictionary to a JSON file
        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"Successfully wrote metrics to {os.path.join(results_dir, 'metrics.json')}")
        # --- END OF NEW SECTION ---

    print("--- Training run finished ---")

if __name__ == "__main__":
    train_model()
