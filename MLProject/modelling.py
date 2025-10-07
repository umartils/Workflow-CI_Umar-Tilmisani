import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Path dataset relatif terhadap file script
    test_dataset_path = "car_evaluation_processed/test.csv"
    train_dataset_path = "car_evaluation_processed/train_resampled.csv"

    # Load data
    test_data = pd.read_csv(test_dataset_path)
    train_data = pd.read_csv(train_dataset_path)

    X_train = train_data.drop('class', axis=1)
    y_train = train_data['class']
    X_test = test_data.drop('class', axis=1)
    y_test = test_data['class']

    input_example = X_train.iloc[:5]

    # Ambil parameter dari MLflow
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 35

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi
        predicted = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        # Log model dan metrik
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        print(f"✅ Training selesai | Akurasi: {accuracy:.4f}")
