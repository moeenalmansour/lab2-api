import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL = [
    "contract",
    "internet_service",
]


def train_model(data_path, model_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Support the original dataset column name
    if "monthly_fee" in df.columns and "monthly_charges" not in df.columns:
        df = df.rename(columns={"monthly_fee": "monthly_charges"})

    # Check required columns
    required = NUMERIC + CATEGORICAL + ["churn"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Prepare features and target
    X = df[NUMERIC + CATEGORICAL]
    y = df["churn"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preprocessing
    preprocessing = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    # Build model pipeline
    model = Pipeline([
        ("prep", preprocessing),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities > 0.5

    print("F1:", round(f1_score(y_test, predictions), 3))
    print("AUC:", round(roc_auc_score(y_test, probabilities), 3))

    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/churn_v2.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output",
        default="artifacts/model.joblib",
        help="Path to save the trained model",
    )

    args = parser.parse_args()
    train_model(args.data, args.output)
