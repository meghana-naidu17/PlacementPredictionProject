import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# XGBOOST
# =========================================================

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

except ImportError:
    XGBOOST_AVAILABLE = False


# =========================================================
# LIGHTGBM
# =========================================================

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True

except ImportError:
    LIGHTGBM_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

FILE_PATH = r"C:\Users\Abhiii\Downloads\KL SEM-1\ml\PythonProject1\cleaned_placement_dataset.csv"

TARGET = "PlacementStatus"

CHART_DIR = os.path.join(
    "static",
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    return pd.read_csv(FILE_PATH)


# =========================================================
# CREATE PREPROCESSOR
# =========================================================

def create_preprocessor(X):

    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return (
        preprocessor,
        numerical_features,
        categorical_features
    )


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data():

    data = load_dataset()

    remove_columns = [
        "StudentID",
        "PlacementStatus",
        "Salary Package",
        "IsAnomaly"
    ]

    remove_columns = [
        column
        for column in remove_columns
        if column in data.columns
    ]

    X = data.drop(
        columns=remove_columns
    )

    Y = data[TARGET]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=42,
        stratify=Y
    )

    return (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    )


# =========================================================
# GET FEATURE NAMES
# =========================================================

def get_feature_names(preprocessor):

    feature_names = []

    # Numerical features
    if "numerical" in preprocessor.named_transformers_:

        numerical_transformer = (
            preprocessor.named_transformers_["numerical"]
        )

        if numerical_transformer != "drop":

            numerical_features = (
                preprocessor.transformers_[0][2]
            )

            feature_names.extend(
                numerical_features
            )

    # Categorical features
    if "categorical" in preprocessor.named_transformers_:

        categorical_transformer = (
            preprocessor.named_transformers_["categorical"]
        )

        if categorical_transformer != "drop":

            categorical_features = (
                preprocessor.transformers_[1][2]
            )

            onehot = (
                categorical_transformer
                .named_steps["onehot"]
            )

            try:

                encoded_names = (
                    onehot.get_feature_names_out(
                        categorical_features
                    )
                )

                feature_names.extend(
                    encoded_names
                )

            except Exception:

                feature_names.extend(
                    categorical_features
                )

    return feature_names


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

def create_confusion_matrix_chart(
    model_name,
    Y_test,
    Y_pred
):

    cm = confusion_matrix(
        Y_test,
        Y_pred
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    path = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.colorbar()

    classes = sorted(
        Y_test.unique()
    )

    plt.xticks(
        range(len(classes)),
        classes
    )

    plt.yticks(
        range(len(classes)),
        classes
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    # Write numbers inside matrix
    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return "/" + path.replace("\\", "/")


# =========================================================
# SAVE FEATURE IMPORTANCE CHART
# =========================================================

def create_feature_importance_chart(
    model_name,
    model,
    preprocessor
):

    feature_names = get_feature_names(
        preprocessor
    )

    # -----------------------------------------------------
    # Find actual estimator
    # -----------------------------------------------------

    estimator = model.steps[-1][1]

    if not hasattr(
        estimator,
        "feature_importances_"
    ):

        return None

    importances = (
        estimator.feature_importances_
    )

    # -----------------------------------------------------
    # Match feature names
    # -----------------------------------------------------

    if len(feature_names) != len(importances):

        feature_names = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importances

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_feature_importance.png"
    )

    path = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{model_name} - Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return "/" + path.replace("\\", "/")


# =========================================================
# DECISION TREE CHART
# =========================================================

def create_decision_tree_chart(
    model,
    preprocessor
):

    estimator = model.named_steps[
        "decision_tree"
    ]

    feature_names = get_feature_names(
        preprocessor
    )

    filename = "decision_tree_structure.png"

    path = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(24, 12)
    )

    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=[
            str(x)
            for x in estimator.classes_
        ],
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=4
    )

    plt.title(
        "Decision Tree Structure"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return "/" + path.replace("\\", "/")


# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

def create_prediction_chart(
    model_name,
    Y_test,
    Y_pred
):

    actual_counts = (
        pd.Series(Y_test)
        .value_counts()
        .sort_index()
    )

    predicted_counts = (
        pd.Series(Y_pred)
        .value_counts()
        .sort_index()
    )

    classes = sorted(
        set(actual_counts.index)
        .union(
            set(predicted_counts.index)
        )
    )

    actual_values = [
        actual_counts.get(
            c,
            0
        )
        for c in classes
    ]

    predicted_values = [
        predicted_counts.get(
            c,
            0
        )
        for c in classes
    ]

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_prediction_distribution.png"
    )

    path = os.path.join(
        CHART_DIR,
        filename
    )

    x = range(
        len(classes)
    )

    width = 0.35

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        [i - width / 2 for i in x],
        actual_values,
        width=width,
        label="Actual"
    )

    plt.bar(
        [i + width / 2 for i in x],
        predicted_values,
        width=width,
        label="Predicted"
    )

    plt.xticks(
        list(x),
        classes
    )

    plt.xlabel(
        "Placement Status"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.title(
        f"{model_name} - Actual vs Predicted"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return "/" + path.replace("\\", "/")


# =========================================================
# GENERATE RESULTS
# =========================================================

def generate_results(
    model_name,
    model,
    Y_test,
    Y_pred,
    data,
    X_train,
    X_test
):

    accuracy = accuracy_score(
        Y_test,
        Y_pred
    )

    cm = confusion_matrix(
        Y_test,
        Y_pred
    )

    report = classification_report(
        Y_test,
        Y_pred,
        output_dict=True
    )

    precision = report[
        "weighted avg"
    ]["precision"]

    recall = report[
        "weighted avg"
    ]["recall"]

    f1 = report[
        "weighted avg"
    ]["f1-score"]

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    if cm.shape == (2, 2):

        tn = int(cm[0][0])
        fp = int(cm[0][1])
        fn = int(cm[1][0])
        tp = int(cm[1][1])

    # -----------------------------------------------------
    # Confusion matrix chart
    # -----------------------------------------------------

    confusion_chart = (
        create_confusion_matrix_chart(
            model_name,
            Y_test,
            Y_pred
        )
    )

    # -----------------------------------------------------
    # Prediction chart
    # -----------------------------------------------------

    prediction_chart = (
        create_prediction_chart(
            model_name,
            Y_test,
            Y_pred
        )
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    feature_importance_chart = None

    if hasattr(model, "named_steps"):

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        feature_importance_chart = (
            create_feature_importance_chart(
                model_name,
                model,
                preprocessor
            )
        )

    # -----------------------------------------------------
    # Decision tree structure
    # -----------------------------------------------------

    decision_tree_chart = None

    if model_name == "Decision Tree":

        decision_tree_chart = (
            create_decision_tree_chart(
                model,
                model.named_steps[
                    "preprocessor"
                ]
            )
        )

    # -----------------------------------------------------
    # RETURN RESULTS
    # -----------------------------------------------------

    return {

        "model_name":
            model_name,

        "accuracy":
            round(
                accuracy * 100,
                2
            ),

        "precision":
            round(
                precision * 100,
                2
            ),

        "recall":
            round(
                recall * 100,
                2
            ),

        "f1_score":
            round(
                f1 * 100,
                2
            ),

        "confusion_matrix":
            cm.tolist(),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test),

        "total_rows":
            len(data),

        "total_columns":
            len(data.columns),

        "classification_report":
            report,

        "confusion_chart":
            confusion_chart,

        "prediction_chart":
            prediction_chart,

        "feature_importance_chart":
            feature_importance_chart,

        "decision_tree_chart":
            decision_tree_chart

    }


# =========================================================
# DECISION TREE
# =========================================================

def decision_tree_model():

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "decision_tree",

            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                random_state=42
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "Decision Tree",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# RANDOM FOREST
# =========================================================

def random_forest_model():

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "random_forest",

            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "Random Forest",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# ADABOOST
# =========================================================

def adaboost_model():

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "adaboost",

            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "AdaBoost",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# GRADIENT BOOSTING
# =========================================================

def gradient_boosting_model():

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "gradient_boosting",

            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "Gradient Boosting",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# XGBOOST
# =========================================================

def xgboost_model():

    if not XGBOOST_AVAILABLE:

        return {
            "error":
            "XGBoost is not installed. "
            "Run: pip install xgboost"
        }

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "xgboost",

            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "XGBoost",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# LIGHTGBM
# =========================================================

def lightgbm_model():

    if not LIGHTGBM_AVAILABLE:

        return {
            "error":
            "LightGBM is not installed. "
            "Run: pip install lightgbm"
        }

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    preprocessor, _, _ = create_preprocessor(X)

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "lightgbm",

            LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        )

    ])

    model.fit(
        X_train,
        Y_train
    )

    Y_pred = model.predict(
        X_test
    )

    return generate_results(
        "LightGBM",
        model,
        Y_test,
        Y_pred,
        data,
        X_train,
        X_test
    )


# =========================================================
# GINI INDEX
# =========================================================

def calculate_gini_index():

    (
        data,
        X,
        Y,
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = prepare_data()

    class_counts = Y.value_counts()

    total = len(Y)

    gini = 1.0

    for count in class_counts:

        probability = count / total

        gini -= probability ** 2

    distribution = {}

    for class_value, count in class_counts.items():

        percentage = (
            count / total
        ) * 100

        distribution[str(class_value)] = {

            "count":
                int(count),

            "percentage":
                round(
                    percentage,
                    2
                )

        }

    return {

        "model_name":
            "Gini Index",

        "gini_index":
            round(
                gini,
                4
            ),

        "class_distribution":
            distribution,

        "total_rows":
            int(total),

        "description":
            "Gini Index measures the impurity of the target variable. "
            "A value of 0 means the node is completely pure, while "
            "higher values indicate greater impurity.",

        "formula":
            "Gini = 1 - Σ(pᵢ²)"

    }


# =========================================================
# MAIN FUNCTION
# =========================================================

def run_tree_based(model_name):

    model_name = model_name.lower()

    if model_name == "decision_tree":
        return decision_tree_model()

    elif model_name == "random_forest":
        return random_forest_model()

    elif model_name == "gini":
        return calculate_gini_index()

    elif model_name == "adaboost":
        return adaboost_model()

    elif model_name == "gradient_boosting":
        return gradient_boosting_model()

    elif model_name == "xgboost":
        return xgboost_model()

    elif model_name == "lightgbm":
        return lightgbm_model()

    else:

        return {
            "error":
            "Invalid tree-based model selected."
        }