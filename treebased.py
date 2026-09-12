import os
import re
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

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
# OPTIONAL MODELS
# =========================================================

try:

    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

except ImportError:

    XGBOOST_AVAILABLE = False


try:

    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# DATASET PATH
# =========================================================

FILE_PATH = os.path.join(
    BASE_DIR,
    "cleaned_placement_dataset.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

TARGET = "PlacementStatus"


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
# SAFE FILE NAME
# =========================================================

def safe_filename(name):

    name = str(name).lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    name = name.strip("_")

    return name


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(

            f"Dataset file not found: {FILE_PATH}"

        )


    data = pd.read_csv(FILE_PATH)


    if data.empty:

        raise ValueError(
            "The dataset is empty."
        )


    if TARGET not in data.columns:

        raise ValueError(

            f"Target column '{TARGET}' "
            f"was not found in the dataset."
        )


    return data


# =========================================================
# CREATE PREPROCESSOR
# =========================================================

def create_preprocessor(X):

    numerical_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )


    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )


    # =====================================================
    # NUMERICAL PIPELINE
    # =====================================================

    numerical_pipeline = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )

    ])


    # =====================================================
    # CATEGORICAL PIPELINE
    # =====================================================

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
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ])


    transformers = []


    if numerical_features:

        transformers.append(

            (
                "numerical",

                numerical_pipeline,

                numerical_features
            )

        )


    if categorical_features:

        transformers.append(

            (
                "categorical",

                categorical_pipeline,

                categorical_features
            )

        )


    if not transformers:

        raise ValueError(
            "No usable features were found."
        )


    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop"

    )


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


    # =====================================================
    # REMOVE UNNECESSARY COLUMNS
    # =====================================================

    remove_columns = [

        "StudentID",

        TARGET,

        "Salary Package",

        "IsAnomaly"

    ]


    remove_columns = [

        column

        for column in remove_columns

        if column in data.columns

    ]


    X = data.drop(

        columns=remove_columns,

        errors="ignore"

    )


    Y = data[TARGET].copy()


    # =====================================================
    # REMOVE MISSING TARGET VALUES
    # =====================================================

    valid_rows = Y.notna()


    X = X.loc[
        valid_rows
    ].copy()


    Y = Y.loc[
        valid_rows
    ].copy()


    if len(X) == 0:

        raise ValueError(
            "No valid rows available for training."
        )


    if Y.nunique() < 2:

        raise ValueError(
            "Target column must contain at least two classes."
        )


    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, Y_train, Y_test = (

        train_test_split(

            X,

            Y,

            test_size=0.30,

            random_state=42,

            stratify=Y

        )

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

    try:

        feature_names = (

            preprocessor
            .get_feature_names_out()
            .tolist()

        )


        cleaned_names = []


        for feature in feature_names:

            feature = str(feature)

            feature = feature.replace(
                "numerical__",
                ""
            )

            feature = feature.replace(
                "categorical__",
                ""
            )

            cleaned_names.append(
                feature
            )


        return cleaned_names


    except Exception:

        return []


# =========================================================
# GET ESTIMATOR
# =========================================================

def get_estimator(model):

    for step_name, estimator in reversed(model.steps):

        if step_name != "preprocessor":

            return estimator


    raise ValueError(
        "Could not find trained estimator."
    )


# =========================================================
# CONFUSION MATRIX CHART
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

        safe_filename(model_name)

        +

        "_confusion_matrix.png"

    )


    path = os.path.join(

        CHART_DIR,

        filename

    )


    classes = sorted(

        list(

            set(Y_test).union(
                set(Y_pred)
            )

        ),

        key=str

    )


    plt.figure(

        figsize=(7, 5)

    )


    plt.imshow(

        cm,

        interpolation="nearest"

    )


    plt.title(

        f"{model_name} - Confusion Matrix"

    )


    plt.colorbar()


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

        dpi=150,

        bbox_inches="tight"

    )


    plt.close()


    return (

        "/static/charts/"

        +

        filename

    )


# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================

def create_feature_importance_chart(

    model_name,

    model,

    preprocessor

):

    estimator = get_estimator(model)


    if not hasattr(
        estimator,
        "feature_importances_"
    ):

        return None


    importances = (

        estimator.feature_importances_

    )


    feature_names = (

        get_feature_names(
            preprocessor
        )

    )


    # =====================================================
    # FIX FEATURE NAME MISMATCH
    # =====================================================

    if len(feature_names) != len(importances):

        feature_names = [

            f"Feature {i + 1}"

            for i in range(
                len(importances)
            )

        ]


    importance_df = (

        pd.DataFrame({

            "feature":

            feature_names,

            "importance":

            importances

        })

        .sort_values(

            "importance",

            ascending=False

        )

        .head(15)

    )


    filename = (

        safe_filename(
            model_name
        )

        +

        "_feature_importance.png"

    )


    path = os.path.join(

        CHART_DIR,

        filename

    )


    plt.figure(

        figsize=(11, 7)

    )


    plt.barh(

        importance_df[
            "feature"
        ][::-1],

        importance_df[
            "importance"
        ][::-1]

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

        dpi=150,

        bbox_inches="tight"

    )


    plt.close()


    return (

        "/static/charts/"

        +

        filename

    )


# =========================================================
# DECISION TREE STRUCTURE CHART
# =========================================================

def create_decision_tree_chart(

    model,

    preprocessor

):

    # =====================================================
    # GET DECISION TREE
    # =====================================================

    estimator = model.named_steps.get(

        "decision_tree"

    )


    if estimator is None:

        estimator = get_estimator(model)


    if not isinstance(

        estimator,

        DecisionTreeClassifier

    ):

        return None


    # =====================================================
    # GET FEATURE NAMES
    # =====================================================

    feature_names = (

        get_feature_names(
            preprocessor
        )

    )


    # =====================================================
    # MAKE SURE LENGTH MATCHES
    # =====================================================

    n_features = estimator.n_features_in_


    if len(feature_names) != n_features:

        feature_names = [

            f"Feature {i + 1}"

            for i in range(
                n_features
            )

        ]


    filename = (

        "decision_tree_structure.png"

    )


    path = os.path.join(

        CHART_DIR,

        filename

    )


    plt.figure(

        figsize=(28, 14)

    )


    plot_tree(

        estimator,

        feature_names=
        feature_names,

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


    return (

        "/static/charts/"

        +

        filename

    )


# =========================================================
# PREDICTION DISTRIBUTION CHART
# =========================================================

def create_prediction_chart(

    model_name,

    Y_test,

    Y_pred

):

    actual_counts = (

        pd.Series(
            Y_test
        )

        .value_counts()

    )


    predicted_counts = (

        pd.Series(
            Y_pred
        )

        .value_counts()

    )


    classes = sorted(

        set(
            actual_counts.index
        )

        .union(

            set(
                predicted_counts.index
            )

        ),

        key=str

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

        safe_filename(
            model_name
        )

        +

        "_prediction_distribution.png"

    )


    path = os.path.join(

        CHART_DIR,

        filename

    )


    x = list(
        range(
            len(classes)
        )
    )


    width = 0.35


    plt.figure(

        figsize=(8, 5)

    )


    plt.bar(

        [

            i - width / 2

            for i in x

        ],

        actual_values,

        width=width,

        label="Actual"

    )


    plt.bar(

        [

            i + width / 2

            for i in x

        ],

        predicted_values,

        width=width,

        label="Predicted"

    )


    plt.xticks(

        x,

        [

            str(c)

            for c in classes

        ]

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

        dpi=150,

        bbox_inches="tight"

    )


    plt.close()


    return (

        "/static/charts/"

        +

        filename

    )


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

        output_dict=True,

        zero_division=0

    )


    precision = report[
        "weighted avg"
    ][
        "precision"
    ]


    recall = report[
        "weighted avg"
    ][
        "recall"
    ]


    f1 = report[
        "weighted avg"
    ][
        "f1-score"
    ]


    tn = 0
    fp = 0
    fn = 0
    tp = 0


    if cm.shape == (2, 2):

        tn = int(cm[0][0])

        fp = int(cm[0][1])

        fn = int(cm[1][0])

        tp = int(cm[1][1])


    # =====================================================
    # GENERATE CHARTS
    # =====================================================

    confusion_chart = (

        create_confusion_matrix_chart(

            model_name,

            Y_test,

            Y_pred

        )

    )


    prediction_chart = (

        create_prediction_chart(

            model_name,

            Y_test,

            Y_pred

        )

    )


    preprocessor = model.named_steps.get(

        "preprocessor"

    )


    feature_importance_chart = None

    decision_tree_chart = None


    if preprocessor is not None:

        feature_importance_chart = (

            create_feature_importance_chart(

                model_name,

                model,

                preprocessor

            )

        )


        if model_name == "Decision Tree":

            decision_tree_chart = (

                create_decision_tree_chart(

                    model,

                    preprocessor

                )

            )


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
# BUILD MODEL
# =========================================================

def build_and_run_model(

    model_name,

    estimator

):

    (

        data,

        X,

        Y,

        X_train,

        X_test,

        Y_train,

        Y_test

    ) = prepare_data()


    preprocessor, _, _ = (

        create_preprocessor(X)

    )


    model_key = (

        safe_filename(
            model_name
        )

    )


    model = Pipeline([

        (

            "preprocessor",

            preprocessor

        ),

        (

            model_key,

            estimator

        )

    ])


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(

        X_train,

        Y_train

    )


    # =====================================================
    # PREDICT
    # =====================================================

    Y_pred = (

        model.predict(
            X_test
        )

    )


    return (

        generate_results(

            model_name,

            model,

            Y_test,

            Y_pred,

            data,

            X_train,

            X_test

        )

    )


# =========================================================
# DECISION TREE
# =========================================================

def decision_tree_model():

    return build_and_run_model(

        "Decision Tree",

        DecisionTreeClassifier(

            criterion="gini",

            max_depth=5,

            min_samples_split=2,

            min_samples_leaf=1,

            random_state=42

        )

    )


# =========================================================
# RANDOM FOREST
# =========================================================

def random_forest_model():

    return build_and_run_model(

        "Random Forest",

        RandomForestClassifier(

            n_estimators=100,

            max_depth=10,

            random_state=42,

            n_jobs=-1

        )

    )


# =========================================================
# ADABOOST
# =========================================================

def adaboost_model():

    return build_and_run_model(

        "AdaBoost",

        AdaBoostClassifier(

            n_estimators=100,

            learning_rate=1.0,

            random_state=42

        )

    )


# =========================================================
# GRADIENT BOOSTING
# =========================================================

def gradient_boosting_model():

    return build_and_run_model(

        "Gradient Boosting",

        GradientBoostingClassifier(

            n_estimators=100,

            learning_rate=0.1,

            max_depth=3,

            random_state=42

        )

    )


# =========================================================
# XGBOOST
# =========================================================

def xgboost_model():

    if not XGBOOST_AVAILABLE:

        return {

            "error":

            "XGBoost is not installed. Run: pip install xgboost"

        }


    return build_and_run_model(

        "XGBoost",

        XGBClassifier(

            n_estimators=100,

            learning_rate=0.1,

            max_depth=3,

            random_state=42,

            eval_metric="logloss",

            n_jobs=-1

        )

    )


# =========================================================
# LIGHTGBM
# =========================================================

def lightgbm_model():

    if not LIGHTGBM_AVAILABLE:

        return {

            "error":

            "LightGBM is not installed. Run: pip install lightgbm"

        }


    return build_and_run_model(

        "LightGBM",

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


    class_counts = (

        Y.value_counts()

    )


    total = len(Y)


    gini = 1.0


    for count in class_counts:

        probability = count / total

        gini -= probability ** 2


    distribution = {}


    for class_value, count in (

        class_counts.items()

    ):

        percentage = (

            count / total

        ) * 100


        distribution[
            str(class_value)
        ] = {

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

    if not model_name:

        return {

            "error":

            "Please select a model."

        }


    model_name = (

        str(model_name)

        .strip()

        .lower()

    )


    try:

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


    except Exception as e:

        import traceback

        traceback.print_exc()


        return {

            "error":

            f"{type(e).__name__}: {str(e)}"

        }