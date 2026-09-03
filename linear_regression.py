import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3)(in).csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# =========================================================
# NUMERIC CLEANING
# =========================================================

def to_numeric_clean(series):

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(y_true, predictions):

    mse = mean_squared_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {

        "mse": round(
            float(mse),
            4
        ),

        "rmse": round(
            float(rmse),
            4
        ),

        "mae": round(
            float(mae),
            4
        ),

        "r2": round(
            float(r2),
            4
        )
    }


# =========================================================
# MAIN
# =========================================================

def run_linear_regression(
    regularization="none"
):

    try:

        # =================================================
        # LOAD DATA
        # =================================================

        if not os.path.exists(
            DATASET_PATH
        ):

            return {
                "error":
                f"Dataset not found: {DATASET_PATH}"
            }

        df = pd.read_csv(
            DATASET_PATH
        )

        # =================================================
        # COLUMNS
        # =================================================

        feature = "CGPA"

        target = "Salary Package"

        if feature not in df.columns:

            return {
                "error":
                f"Column '{feature}' not found."
            }

        if target not in df.columns:

            return {
                "error":
                f"Column '{target}' not found."
            }

        # =================================================
        # CLEAN DATA
        # =================================================

        df[feature] = to_numeric_clean(
            df[feature]
        )

        df[target] = to_numeric_clean(
            df[target]
        )

        df = df[
            [feature, target]
        ].dropna()

        if len(df) < 10:

            return {
                "error":
                "Not enough valid data."
            }

        # =================================================
        # X / Y
        # =================================================

        X = df[
            [feature]
        ].values

        y = df[
            target
        ].values

        # =================================================
        # TRAIN TEST SPLIT
        # =================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42

        )

        # =================================================
        # STANDARD SCALING
        # =================================================

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # =================================================
        # REGULARIZATION SELECTION
        # =================================================

        regularization = (
            regularization
            .lower()
            .strip()
        )

        if regularization == "none":

            model = LinearRegression()

            model_name = (
                "Linear Regression - "
                "Without Regularization"
            )

        elif regularization == "ridge":

            model = Ridge(
                alpha=1.0
            )

            model_name = (
                "Ridge Regression - "
                "L2 Regularization"
            )

        elif regularization == "lasso":

            model = Lasso(
                alpha=0.01,
                max_iter=10000
            )

            model_name = (
                "Lasso Regression - "
                "L1 Regularization"
            )

        else:

            return {
                "error":
                "Invalid regularization option."
            }

        # =================================================
        # TRAIN MODEL
        # =================================================

        model.fit(
            X_train_scaled,
            y_train
        )

        # =================================================
        # PREDICTIONS
        # =================================================

        train_predictions = model.predict(
            X_train_scaled
        )

        test_predictions = model.predict(
            X_test_scaled
        )

        # =================================================
        # MODEL METRICS
        # =================================================

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        test_metrics = calculate_metrics(
            y_test,
            test_predictions
        )

        # =================================================
        # ACTUAL VS PREDICTED CHART
        # =================================================

        comparison_chart = (
            f"linear_{regularization}_"
            f"actual_vs_predicted.png"
        )

        comparison_path = os.path.join(
            CHART_DIR,
            comparison_chart
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.scatter(
            y_test,
            test_predictions,
            alpha=0.6
        )

        min_value = min(
            y_test.min(),
            test_predictions.min()
        )

        max_value = max(
            y_test.max(),
            test_predictions.max()
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        plt.xlabel(
            "Actual Salary Package"
        )

        plt.ylabel(
            "Predicted Salary Package"
        )

        plt.title(
            model_name
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            comparison_path
        )

        plt.close()

        # =================================================
        # GRADIENT DESCENT
        # =================================================

        X_gd = X_train_scaled

        X_gd_bias = np.c_[
            np.ones(
                X_gd.shape[0]
            ),
            X_gd
        ]

        weights = np.zeros(
            X_gd_bias.shape[1]
        )

        learning_rate = 0.05

        epochs = 50

        mse_history = []

        # -------------------------------------------------
        # GRADIENT DESCENT TRAINING
        # -------------------------------------------------

        for epoch in range(
            epochs
        ):

            predictions = (
                X_gd_bias @ weights
            )

            error = (
                predictions - y_train
            )

            mse = np.mean(
                error ** 2
            )

            mse_history.append(
                float(mse)
            )

            gradient = (

                2 / len(y_train)

            ) * (

                X_gd_bias.T @ error

            )

            weights -= (
                learning_rate * gradient
            )

        # =================================================
        # GRADIENT DESCENT FINAL PREDICTIONS
        # =================================================

        gd_train_predictions = (
            X_gd_bias @ weights
        )

        # =================================================
        # GRADIENT DESCENT METRICS
        # =================================================

        gd_metrics = calculate_metrics(
            y_train,
            gd_train_predictions
        )

        # =================================================
        # GRADIENT DESCENT CHART
        # =================================================

        gd_mse_chart = (
            f"linear_{regularization}_"
            f"gradient_descent.png"
        )

        gd_chart_path = os.path.join(
            CHART_DIR,
            gd_mse_chart
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            range(
                1,
                epochs + 1
            ),
            mse_history,
            marker="o"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Mean Squared Error"
        )

        plt.title(
            f"Gradient Descent - {model_name}"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            gd_chart_path
        )

        plt.close()

        # =================================================
        # RETURN
        # =================================================

        return {

            "success": True,

            "model_name":
                model_name,

            "regularization":
                regularization,

            "feature":
                feature,

            "target":
                target,

            "samples":
                len(df),

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test),

            # ---------------------------------------------
            # IMPORTANT:
            # HTML EXPECTS train_metrics/test_metrics
            # ---------------------------------------------

            "train_metrics":
                train_metrics,

            "test_metrics":
                test_metrics,

            # ---------------------------------------------
            # REGRESSION PARAMETERS
            # ---------------------------------------------

            "coefficient":
                round(
                    float(
                        model.coef_[0]
                    ),
                    6
                ),

            "intercept":
                round(
                    float(
                        model.intercept_
                    ),
                    6
                ),

            # ---------------------------------------------
            # GRADIENT DESCENT
            # ---------------------------------------------

            "gd_metrics":
                gd_metrics,

            "mse_history":
                mse_history,

            # ---------------------------------------------
            # CHART NAMES
            # ---------------------------------------------

            "comparison_chart":
                comparison_chart,

            "gd_mse_chart":
                gd_mse_chart

        }

    except Exception as e:

        return {
            "error":
            str(e)
        }