import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3)(in).csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    data = pd.read_csv(DATASET_PATH)

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    X = data[features].copy()

    y = data["PlacementStatus"].copy()

    # -----------------------------------------------------
    # Convert target to 0/1
    # -----------------------------------------------------

    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "placed": 1,
                "not placed": 0,
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            })
        )

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

    # -----------------------------------------------------
    # Remove missing values
    # -----------------------------------------------------

    combined = pd.concat(
        [X, y.rename("PlacementStatus")],
        axis=1
    )

    combined = combined.dropna()

    X = combined[features]

    y = combined["PlacementStatus"].astype(int)

    return X, y


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Training predictions
    train_predictions = model.predict(
        X_train
    )

    # Testing predictions
    test_predictions = model.predict(
        X_test
    )

    # Training accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    # Testing accuracy
    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    return (
        model,
        train_accuracy,
        test_accuracy,
        test_predictions
    )


# =========================================================
# MAIN LOGISTIC REGRESSION
# =========================================================

def run_logistic_regression():

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    X, y = load_data()

    # -----------------------------------------------------
    # Split dataset
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        stratify=y,

        random_state=42
    )


    # =====================================================
    # UNSCALED LOGISTIC REGRESSION
    # =====================================================

    (
        unscaled_model,
        unscaled_train,
        unscaled_test,
        unscaled_predictions
    ) = train_model(

        X_train,
        X_test,
        y_train,
        y_test
    )


    # =====================================================
    # STANDARD SCALER
    # =====================================================

    standard_scaler = StandardScaler()

    X_train_std = standard_scaler.fit_transform(
        X_train
    )

    X_test_std = standard_scaler.transform(
        X_test
    )

    (
        standard_model,
        standard_train,
        standard_test,
        standard_predictions
    ) = train_model(

        X_train_std,
        X_test_std,
        y_train,
        y_test
    )


    # =====================================================
    # MIN-MAX SCALER
    # =====================================================

    minmax_scaler = MinMaxScaler()

    X_train_mm = minmax_scaler.fit_transform(
        X_train
    )

    X_test_mm = minmax_scaler.transform(
        X_test
    )

    (
        minmax_model,
        minmax_train,
        minmax_test,
        minmax_predictions
    ) = train_model(

        X_train_mm,
        X_test_mm,
        y_train,
        y_test
    )


    # =====================================================
    # CHART 1
    # SCALING COMPARISON
    # =====================================================

    methods = [
        "Unscaled",
        "StandardScaler",
        "MinMaxScaler"
    ]

    train_scores = [
        unscaled_train,
        standard_train,
        minmax_train
    ]

    test_scores = [
        unscaled_test,
        standard_test,
        minmax_test
    ]

    plt.figure(
        figsize=(10, 6)
    )

    x = range(
        len(methods)
    )

    width = 0.35

    # Training accuracy bars
    plt.bar(
        [i - width / 2 for i in x],
        train_scores,
        width=width,
        label="Training Accuracy"
    )

    # Testing accuracy bars
    plt.bar(
        [i + width / 2 for i in x],
        test_scores,
        width=width,
        label="Testing Accuracy"
    )

    plt.xticks(
        list(x),
        methods
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.xlabel(
        "Scaling Method"
    )

    plt.title(
        "Logistic Regression Scaling Comparison"
    )

    plt.legend()

    plt.tight_layout()

    scaling_chart = os.path.join(
        CHART_DIR,
        "logistic_scaling_comparison.png"
    )

    plt.savefig(
        scaling_chart,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # CHART 2
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_test,
        standard_predictions
    )

    plt.figure(
        figsize=(7, 6)
    )

    # Display confusion matrix
    plt.imshow(
        cm
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    plt.title(
        "Logistic Regression Confusion Matrix"
    )

    # -----------------------------------------------------
    # AXIS LABELS
    # -----------------------------------------------------

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    # -----------------------------------------------------
    # X AXIS
    # -----------------------------------------------------

    plt.xticks(
        [0, 1],
        [
            "Not Placed",
            "Placed"
        ]
    )

    # -----------------------------------------------------
    # Y AXIS
    # -----------------------------------------------------

    plt.yticks(
        [0, 1],
        [
            "Not Placed",
            "Placed"
        ]
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX VALUES
    #
    # White text is used for darker cells.
    # Black text is used for lighter cells.
    # -----------------------------------------------------

    max_value = cm.max()

    for i in range(2):

        for j in range(2):

            # Decide text colour based on value
            if cm[i, j] < max_value * 0.7:

                text_color = "white"

            else:

                text_color = "black"

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold"
            )

    # -----------------------------------------------------
    # SAVE CONFUSION MATRIX
    # -----------------------------------------------------

    plt.tight_layout()

    confusion_chart = os.path.join(
        CHART_DIR,
        "logistic_confusion_matrix.png"
    )

    plt.savefig(
        confusion_chart,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # RETURN RESULTS TO FLASK
    # =====================================================

    return {

        # -------------------------------------------------
        # Main model accuracy
        # StandardScaler is used as the main model
        # -------------------------------------------------

        "train_accuracy": round(
            standard_train,
            4
        ),

        "test_accuracy": round(
            standard_test,
            4
        ),

        # -------------------------------------------------
        # Dataset sizes
        # -------------------------------------------------

        "training_records": len(
            X_train
        ),

        "testing_records": len(
            X_test
        ),

        # -------------------------------------------------
        # Scaling comparison
        # -------------------------------------------------

        "scaling_comparison": [

            {
                "method": "Unscaled",

                "training_accuracy": round(
                    unscaled_train,
                    4
                ),

                "testing_accuracy": round(
                    unscaled_test,
                    4
                )
            },

            {
                "method": "StandardScaler",

                "training_accuracy": round(
                    standard_train,
                    4
                ),

                "testing_accuracy": round(
                    standard_test,
                    4
                )
            },

            {
                "method": "MinMaxScaler",

                "training_accuracy": round(
                    minmax_train,
                    4
                ),

                "testing_accuracy": round(
                    minmax_test,
                    4
                )
            }

        ],

        # -------------------------------------------------
        # Chart paths
        # -------------------------------------------------

        "scaling_chart":
            "charts/logistic_scaling_comparison.png",

        "confusion_chart":
            "charts/logistic_confusion_matrix.png"
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    result = run_logistic_regression()

    print()
    print(
        "=========================================="
    )

    print(
        "       LOGISTIC REGRESSION"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Train Accuracy:",
        result["train_accuracy"]
    )

    print(
        "Test Accuracy:",
        result["test_accuracy"]
    )

    print(
        "Training Records:",
        result["training_records"]
    )

    print(
        "Testing Records:",
        result["testing_records"]
    )

    print()

    print(
        "Scaling Comparison:"
    )

    for item in result["scaling_comparison"]:

        print(
            item["method"],
            "| Training:",
            item["training_accuracy"],
            "| Testing:",
            item["testing_accuracy"]
        )

    print()

    print(
        "Charts generated successfully."
    )

    print(
        "Scaling Chart:",
        result["scaling_chart"]
    )

    print(
        "Confusion Matrix:",
        result["confusion_chart"]
    )

    print()