"""
Linear Regression for Green Building Dataset.

This script predicts predicted_energy_demand using a suitable set of
other parameters and evaluates the linear model.
"""

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    kagglepath = "programmer3/green-building-multi-source-environment-dataset"
    path = kagglehub.dataset_download(kagglepath)

    print("Path to dataset files:", path)
    print("Listing contents:", os.listdir(path))

    df = pd.read_csv(path + "/green_building_dataset.csv")
    df.columns = [c.strip() for c in df.columns]

    print("\nDataset shape:", df.shape)
    print(df.head())

    target = "predicted_energy_demand"

    X_all = df.drop(columns=[target])
    y = df[target]

    # Remove ID and timestamp/date columns
    drop_cols = []
    for c in X_all.columns:
        name = c.lower()
        if name == "id" or name.endswith("_id") or "timestamp" in name or name == "date":
            drop_cols.append(c)

    X_all = X_all.drop(columns=drop_cols, errors="ignore")

    print("\nDropped columns:", drop_cols)
    print("Remaining predictors:", list(X_all.columns))

    numeric_features = X_all.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_all.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    lasso_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LassoCV(cv=5, random_state=42, max_iter=20000))
    ])

    lasso_model.fit(X_train, y_train)

    feature_names = lasso_model.named_steps["preprocessor"].get_feature_names_out()
    coefs = lasso_model.named_steps["model"].coef_

    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    selected_features = coef_table[coef_table["abs_coefficient"] > 1e-8]

    print("\nSelected features using Lasso:")
    print(selected_features)

    X_train_processed = lasso_model.named_steps["preprocessor"].transform(X_train)
    X_test_processed = lasso_model.named_steps["preprocessor"].transform(X_test)

    selected_indices = np.where(np.abs(coefs) > 1e-8)[0]

    if len(selected_indices) == 0:
        print("\nNo features selected by Lasso. Using Ridge regression with all predictors.")
        final_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
        final_model.fit(X_train_processed, y_train)
        y_pred = final_model.predict(X_test_processed)
        used_feature_names = feature_names
        final_coefs = final_model.coef_
    else:
        print("\nUsing Linear Regression with Lasso-selected predictors.")
        final_model = LinearRegression()
        final_model.fit(X_train_processed[:, selected_indices], y_train)
        y_pred = final_model.predict(X_test_processed[:, selected_indices])
        used_feature_names = feature_names[selected_indices]
        final_coefs = final_model.coef_

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nLinear Regression Results")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    final_coef_table = pd.DataFrame({
        "feature": used_feature_names,
        "coefficient": final_coefs,
        "abs_coefficient": np.abs(final_coefs)
    }).sort_values("abs_coefficient", ascending=False)

    print("\nTop model coefficients:")
    print(final_coef_table.head(20))

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual predicted_energy_demand")
    plt.ylabel("Model predicted predicted_energy_demand")
    plt.title("Actual vs Predicted Energy Demand")
    plt.grid(True)
    plt.savefig("figures/linear_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.show()

    residuals = y_test - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.savefig("figures/linear_residual_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    top_features = final_coef_table.head(10)["feature"].tolist()

    print("\nConclusion:")
    print("The most influential selected parameters were:")
    for f in top_features:
        print("-", f)

    print(
        f"\nThe final linear model achieved RMSE = {rmse:.4f}, "
        f"MAE = {mae:.4f}, and R2 = {r2:.4f}. "
        "Linear Regression is useful as a simple and interpretable baseline. "
        "If the residual plot is randomly scattered around zero, the linear model "
        "is suitable. If a pattern appears, nonlinear models should be considered."
    )


if __name__ == "__main__":
    main()
