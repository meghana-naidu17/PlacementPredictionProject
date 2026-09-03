import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# TARGET CONVERSION
# =========================================================

def convert_target(series):

    if series.dtype == object:

        mapping = {

            "placed": 1,
            "not placed": 0,

            "yes": 1,
            "no": 0,

            "true": 1,
            "false": 0,

            "1": 1,
            "0": 0
        }

        return (
            series
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# =========================================================
# MAIN
# =========================================================

def run_logistic_regression(
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
        # FEATURES
        # =================================================

        features = [

            "CGPA",

            "AptitudeTestScore",

            "CodingTestScore",

            "MockInterviewScore"

        ]

        target = "PlacementStatus"

        # =================================================
        # CHECK COLUMNS
        # =================================================

        missing_columns = [

            column

            for column in
            features + [target]

            if column not in df.columns

        ]

        if missing_columns:

            return {
                "error":
                "Missing columns: "
                + ", ".join(
                    missing_columns
                )
            }

        # =================================================
        # CLEAN FEATURES
        # =================================================

        for column in features:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # =================================================
        # CLEAN TARGET
        # =================================================

        df[target] = convert_target(
            df[target]
        )

        # =================================================
        # DROP MISSING
        # =================================================

        df = df[
            features + [target]
        ].dropna()

        if len(df) < 10:

            return {
                "error":
                "Not enough valid data."
            }

        # =================================================
        # TARGET CHECK
        # =================================================

        if df[target].nunique() != 2:

            return {
                "error":
                "PlacementStatus must contain exactly two classes."
            }

        # =================================================
        # X / Y
        # =================================================

        X = df[
            features
        ]

        y = df[
            target
        ].astype(int)

        # =================================================
        # TRAIN TEST SPLIT
        # =================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y

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
        # REGULARIZATION
        # =================================================

        regularization = (
            regularization
            .lower()
            .strip()
        )

        if regularization == "none":

            model = LogisticRegression(
                penalty=None,
                max_iter=1000
            )

            model_name = (
                "Logistic Regression - "
                "Without Regularization"
            )

        elif regularization == "l2":

            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000
            )

            model_name = (
                "Logistic Regression - "
                "L2 Regularization"
            )

        elif regularization == "l1":

            model = LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=1000
            )

            model_name = (
                "Logistic Regression - "
                "L1 Regularization"
            )

        else:

            return {
                "error":
                "Invalid regularization option."
            }

        # =================================================
        # TRAIN
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
        # METRICS
        # =================================================

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_test,
            test_predictions
        )

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        report = classification_report(

            y_test,

            test_predictions,

            output_dict=True,

            zero_division=0

        )

        # =================================================
        # CONFUSION MATRIX CHART
        # =================================================

        chart_name = (
            f"logistic_"
            f"{regularization}_"
            f"confusion_matrix.png"
        )

        chart_path = os.path.join(
            CHART_DIR,
            chart_name
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            cm,
            interpolation="nearest"
        )

        plt.title(
            model_name
        )

        plt.colorbar()

        classes = [
            "Not Placed",
            "Placed"
        ]

        plt.xticks(
            [0, 1],
            classes
        )

        plt.yticks(
            [0, 1],
            classes
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                plt.text(

                    j,
                    i,

                    str(
                        cm[i, j]
                    ),

                    ha="center",

                    va="center"

                )

        plt.tight_layout()

        plt.savefig(
            chart_path
        )

        plt.close()

        # =================================================
        # COEFFICIENTS
        # =================================================

        coefficients = {}

        for feature, coefficient in zip(

            features,

            model.coef_[0]

        ):

            coefficients[feature] = round(

                float(
                    coefficient
                ),

                4

            )

        # =================================================
        # RETURN
        # =================================================

        return {

            "success": True,

            "model_name":
                model_name,

            "regularization":
                regularization,

            "features":
                features,

            "samples":
                len(df),

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test),

            "train_accuracy":
                round(
                    float(train_accuracy),
                    4
                ),

            "test_accuracy":
                round(
                    float(test_accuracy),
                    4
                ),

            "precision":
                round(
                    float(precision),
                    4
                ),

            "recall":
                round(
                    float(recall),
                    4
                ),

            "f1":
                round(
                    float(f1),
                    4
                ),

            "confusion_matrix":
                cm.tolist(),

            "classification_report":
                report,

            "coefficients":
                coefficients,

            "confusion_chart":
                chart_name

        }

    except Exception as e:

        return {
            "error":
            str(e)
        }