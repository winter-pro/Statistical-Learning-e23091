"""
Gaussian Process Regression for Energy Efficiency Dataset.

This script explores whether heating load and cooling load can be modeled
as a single latent parameter using Gaussian Process Regression.
"""

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred, name):
    """Return standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Output": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def main():
    kagglepath = "elikplim/eergy-efficiency-dataset"
    path = kagglehub.dataset_download(kagglepath)

    print("Path to dataset files:", path)
    print("Listing contents:", os.listdir(path))

    df = pd.read_csv(path + "/ENB2012_data.csv")
    df.columns = [c.strip() for c in df.columns]

    print("\nDataset shape:", df.shape)
    print(df.head())

    x_cols = [c for c in df.columns if c.startswith("X")]
    y_cols = ["Y1", "Y2"]

    X = df[x_cols].values
    Y = df[y_cols].values

    corr_y1_y2 = df["Y1"].corr(df["Y2"])
    print("\nCorrelation between Heating Load Y1 and Cooling Load Y2:", corr_y1_y2)

    plt.figure(figsize=(6, 5))
    plt.scatter(df["Y1"], df["Y2"])
    plt.xlabel("Heating Load Y1")
    plt.ylabel("Cooling Load Y2")
    plt.title("Relationship between Heating Load and Cooling Load")
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/gp_y1_y2_relationship.png", dpi=300, bbox_inches="tight")
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    y_scaler = StandardScaler()
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_test_scaled = y_scaler.transform(Y_test)

    # Single latent target using PCA
    pca = PCA(n_components=1)
    z_train = pca.fit_transform(Y_train_scaled).ravel()
    z_test = pca.transform(Y_test_scaled).ravel()

    pc1_variance = pca.explained_variance_ratio_[0]
    print("\nVariance explained by first principal component:", pc1_variance)

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(len(x_cols)), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )

    single_gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=2
        )
    )

    single_gp.fit(X_train, z_train)
    z_pred = single_gp.predict(X_test)

    Y_pred_scaled = pca.inverse_transform(z_pred.reshape(-1, 1))
    Y_pred_single = y_scaler.inverse_transform(Y_pred_scaled)

    single_results = pd.DataFrame([
        regression_metrics(Y_test[:, 0], Y_pred_single[:, 0], "Heating Load Y1 - Single GP"),
        regression_metrics(Y_test[:, 1], Y_pred_single[:, 1], "Cooling Load Y2 - Single GP")
    ])

    print("\nSingle-parameter GP results:")
    print(single_results)

    # Separate GP models
    separate_predictions = []

    for j, output_name in enumerate(y_cols):
        gp = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=42,
                n_restarts_optimizer=2
            )
        )
        gp.fit(X_train, Y_train[:, j])
        pred = gp.predict(X_test)
        separate_predictions.append(pred)

    Y_pred_separate = np.column_stack(separate_predictions)

    separate_results = pd.DataFrame([
        regression_metrics(Y_test[:, 0], Y_pred_separate[:, 0], "Heating Load Y1 - Separate GP"),
        regression_metrics(Y_test[:, 1], Y_pred_separate[:, 1], "Cooling Load Y2 - Separate GP")
    ])

    print("\nSeparate GP results:")
    print(separate_results)

    comparison = pd.concat([single_results, separate_results], ignore_index=True)
    print("\nComparison:")
    print(comparison)

    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test[:, 0], Y_pred_single[:, 0], label="Single GP")
    plt.scatter(Y_test[:, 0], Y_pred_separate[:, 0], label="Separate GP")
    plt.xlabel("Actual Heating Load Y1")
    plt.ylabel("Predicted Heating Load Y1")
    plt.title("Heating Load Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/gp_heating_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test[:, 1], Y_pred_single[:, 1], label="Single GP")
    plt.scatter(Y_test[:, 1], Y_pred_separate[:, 1], label="Separate GP")
    plt.xlabel("Actual Cooling Load Y2")
    plt.ylabel("Predicted Cooling Load Y2")
    plt.title("Cooling Load Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/gp_cooling_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nConclusion:")
    print(
        f"The correlation between Y1 and Y2 is {corr_y1_y2:.4f}. "
        f"The first principal component explains {pc1_variance * 100:.2f}% "
        "of the variation in the two outputs. Therefore, a single latent "
        "thermal-load parameter can approximately represent both heating and "
        "cooling load. However, this is an approximation. Separate GP models "
        "or a multi-output GP are better if maximum prediction accuracy is required."
    )


if __name__ == "__main__":
    main()
