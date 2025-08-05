import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dvc.api
from pygit2 import Repository

repo_name = Repository('.').head.shorthand 
# Set the MLflow track

mlflow.set_tracking_uri('http://34.67.110.16:8100') # Or a remote tracking server

def train_model():
    """Trains a model on the current version of the data and logs to MLflow."""

    # Use DVC API to get the path to the current data
    train_path = dvc.api.get_url('data/train.csv', remote='gcsb')
    test_path = dvc.api.get_url('data/test.csv', remote='gcsb')

    X_train = pd.read_csv(train_path).drop('species', axis=1)
    y_train = pd.read_csv(train_path)['species']
    X_test = pd.read_csv(test_path).drop('species', axis=1)
    y_test = pd.read_csv(test_path)['species']

    with mlflow.start_run():
        # Log the DVC version of the data
        mlflow.autolog()
        mlflow.log_param("Repo Name", repo_name)

        # Train the model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log metrics and model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_model()