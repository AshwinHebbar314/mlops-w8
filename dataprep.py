import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def prepare_data(seed=42):
    """Loads the Iris dataset and splits it into training and testing sets."""
    data = pd.read_csv("../data/iris.csv")

    train, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['species'])

    os.makedirs('data', exist_ok=True)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

if __name__ == "__main__":
    prepare_data()