from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from battle_simulator import generate_dataset


MODEL_PATH = Path("artifacts/model.joblib")
DATA_PATH = Path("artifacts/synthetic_battles.csv")


def train_and_save_model(
    num_samples: int = 7000,
    seed: int = 42,
) -> tuple[Pipeline, pd.DataFrame, float, float]:
    artifacts_dir = MODEL_PATH.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(num_samples=num_samples, seed=seed)
    df.to_csv(DATA_PATH, index=False)

    target = "win_probability"
    X = df.drop(columns=[target])
    y = df[target]

    categorical_columns = ["clan_castle", "siege_machine"]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=14,
                    min_samples_split=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    return model, df, mae, r2


def main() -> None:
    _, _, mae, r2 = train_and_save_model()

    print(f"Saved synthetic dataset to {DATA_PATH}")
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")


if __name__ == "__main__":
    main()
