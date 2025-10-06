import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import sys
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    test_dataset_path = '../preprocessing/car_evaluation_preprocessed/test.csv'
    train_dataset_path = '../preprocessing/car_evaluation_preprocessed/train_resampled.csv'

    test_data = pd.read_csv(test_dataset_path)
    train_data = pd.read_csv(train_dataset_path)

    X_train = train_data.drop('class', axis=1)
    y_train = train_data['class']
    X_test = test_data.drop('class', axis=1)
    y_test = test_data['class']

    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )
        model.fit(X_train, y_train)
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)