import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler
)


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = r"C:\Users\Abhiii\Downloads\KL SEM-1\ml\PythonProject1\placement_predict_50k Dataset (3)(in).csv"

TARGET = "PlacementStatus"

MISSING_METHOD = "median"

IQR_COLUMN = "CodingTestScore"


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    return pd.read_csv(DATASET_PATH)


# =========================================================
# MISSING VALUE HANDLING
# =========================================================

def handle_missing_values(data, method="median"):

    df = data.copy()

    missing_before = int(
        df.isnull().sum().sum()
    )

    rows_deleted = []
    columns_deleted = []
    values_imputed = []

    # -----------------------------------------------------
    # ROW DELETION
    # -----------------------------------------------------

    if method == "row":

        old_index = df.index.tolist()

        df = df.dropna()

        new_index = df.index.tolist()

        rows_deleted = [
            i for i in old_index
            if i not in new_index
        ]

    # -----------------------------------------------------
    # COLUMN DELETION
    # -----------------------------------------------------

    elif method == "column":

        columns_deleted = (
            df.columns[
                df.isnull().any()
            ].tolist()
        )

        df = df.dropna(axis=1)

    # -----------------------------------------------------
    # MEAN IMPUTATION
    # -----------------------------------------------------

    elif method == "mean":

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numeric_columns:

            count = int(
                df[column].isnull().sum()
            )

            if count > 0:

                value = df[column].mean()

                df[column] = (
                    df[column].fillna(value)
                )

                values_imputed.append({
                    "column": column,
                    "count": count,
                    "method": "Mean",
                    "value": round(
                        float(value), 4
                    )
                })

        categorical_columns = (
            df.select_dtypes(
                exclude=np.number
            ).columns
        )

        for column in categorical_columns:

            count = int(
                df[column].isnull().sum()
            )

            if count > 0:

                mode = df[column].mode()

                if not mode.empty:

                    value = mode.iloc[0]

                    df[column] = (
                        df[column].fillna(value)
                    )

                    values_imputed.append({
                        "column": column,
                        "count": count,
                        "method": "Mode",
                        "value": str(value)
                    })

    # -----------------------------------------------------
    # MEDIAN IMPUTATION
    # -----------------------------------------------------

    elif method == "median":

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numeric_columns:

            count = int(
                df[column].isnull().sum()
            )

            if count > 0:

                value = df[column].median()

                df[column] = (
                    df[column].fillna(value)
                )

                values_imputed.append({
                    "column": column,
                    "count": count,
                    "method": "Median",
                    "value": round(
                        float(value), 4
                    )
                })

        categorical_columns = (
            df.select_dtypes(
                exclude=np.number
            ).columns
        )

        for column in categorical_columns:

            count = int(
                df[column].isnull().sum()
            )

            if count > 0:

                mode = df[column].mode()

                if not mode.empty:

                    value = mode.iloc[0]

                    df[column] = (
                        df[column].fillna(value)
                    )

                    values_imputed.append({
                        "column": column,
                        "count": count,
                        "method": "Mode",
                        "value": str(value)
                    })

    else:

        raise ValueError(
            "Invalid missing value method"
        )

    missing_after = int(
        df.isnull().sum().sum()
    )

    # -----------------------------------------------------
    # MISSING VALUE TABLE
    # -----------------------------------------------------

    missing_table = []

    for column in data.columns:

        before = int(
            data[column].isnull().sum()
        )

        if column in df.columns:

            after = int(
                df[column].isnull().sum()
            )

        else:

            after = "Column Deleted"

        missing_table.append({
            "column": column,
            "before": before,
            "after": after
        })

    return (
        df,
        missing_before,
        missing_after,
        rows_deleted,
        columns_deleted,
        values_imputed,
        missing_table
    )


# =========================================================
# IQR OUTLIER DETECTION
# =========================================================

def detect_iqr_outliers(df, column):

    if column not in df.columns:

        return None, df

    df = df.copy()

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    n_outliers = int(
        outlier_mask.sum()
    )

    min_before = series.min()
    max_before = series.max()

    clipped = series.clip(
        lower=lower_bound,
        upper=upper_bound
    )

    df[column + "_clipped"] = clipped

    result = {

        "column":
            column,

        "q1":
            round(float(q1), 4),

        "q3":
            round(float(q3), 4),

        "iqr":
            round(float(iqr), 4),

        "lower_bound":
            round(float(lower_bound), 4),

        "upper_bound":
            round(float(upper_bound), 4),

        "n_outliers":
            n_outliers,

        "min_before":
            round(float(min_before), 4),

        "max_before":
            round(float(max_before), 4),

        "min_after":
            round(float(clipped.min()), 4),

        "max_after":
            round(float(clipped.max()), 4)
    }

    return result, df


# =========================================================
# FEATURE ENCODING
# =========================================================

def feature_encoding(df):

    data = df.copy()

    categorical_cols = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "ExtraCurricular"
    ]

    categorical_cols = [
        col for col in categorical_cols
        if col in data.columns
    ]

    # =====================================================
    # ONE HOT ENCODING
    # =====================================================

    onehot_columns = []
    onehot_preview = []

    onehot_cols = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    onehot_cols = [
        col for col in onehot_cols
        if col in data.columns
    ]

    if onehot_cols:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        encoded = encoder.fit_transform(
            data[onehot_cols]
        )

        names = (
            encoder
            .get_feature_names_out(
                onehot_cols
            )
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=names,
            index=data.index
        )

        data = pd.concat(
            [
                data.drop(
                    columns=onehot_cols
                ),
                encoded_df
            ],
            axis=1
        )

        onehot_columns = list(names)

        onehot_preview = (
            encoded_df
            .head(5)
            .round(4)
            .to_dict(
                orient="records"
            )
        )

    # =====================================================
    # ORDINAL ENCODING
    # =====================================================

    ordinal_columns = []
    ordinal_preview = []

    ordinal_mapping = {

        "CollegeTier": [
            "Tier3",
            "Tier2",
            "Tier1"
        ],

        "CGPA_Tier": [
            "Low",
            "Mid",
            "High"
        ]

    }

    for column, categories in ordinal_mapping.items():

        if column in data.columns:

            encoder = OrdinalEncoder(
                categories=[categories],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

            data[[column]] = (
                encoder.fit_transform(
                    data[[column]]
                )
            )

            ordinal_columns.append(
                column
            )

    if ordinal_columns:

        ordinal_preview = (
            data[ordinal_columns]
            .head(5)
            .round(4)
            .to_dict(
                orient="records"
            )
        )

    # =====================================================
    # TARGET ENCODING
    # =====================================================

    target_columns = []
    target_preview = []
    global_target_mean = None

    if TARGET in data.columns:

        if data[TARGET].dtype == "object":

            values = (
                data[TARGET]
                .dropna()
                .unique()
            )

            target_mapping = {
                value: index
                for index, value
                in enumerate(values)
            }

            data[TARGET] = (
                data[TARGET]
                .map(target_mapping)
            )

            target_columns.append(TARGET)

        global_target_mean = round(
            float(data[TARGET].mean()),
            4
        )

        target_preview = (
            data[[TARGET]]
            .head(5)
            .round(4)
            .to_dict(
                orient="records"
            )
        )

    # =====================================================
    # EMBEDDING STYLE ENCODING
    # =====================================================

    embedding_columns = []
    embedding_preview = []

    embedding_candidates = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "ExtraCurricular"
    ]

    for column in embedding_candidates:

        if column in data.columns:

            if data[column].dtype == "object":

                data[column + "_ID"] = (
                    data[column]
                    .astype("category")
                    .cat.codes
                )

                embedding_columns.append(
                    column + "_ID"
                )

    if embedding_columns:

        embedding_preview = (
            data[embedding_columns]
            .head(5)
            .to_dict(
                orient="records"
            )
        )

    return (
        data,
        categorical_cols,
        onehot_columns,
        onehot_preview,
        ordinal_columns,
        ordinal_preview,
        target_columns,
        target_preview,
        global_target_mean,
        embedding_columns,
        embedding_preview
    )


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

def split_data(df):

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================================
# SCALING
# =========================================================

def scale_features(X_train, X_test):

    numeric_columns = (
        X_train
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    standard_scaler = StandardScaler()

    minmax_scaler = MinMaxScaler()

    X_train_standard = X_train.copy()
    X_test_standard = X_test.copy()

    X_train_minmax = X_train.copy()

    if len(numeric_columns) > 0:

        X_train_standard[
            numeric_columns
        ] = standard_scaler.fit_transform(
            X_train[numeric_columns]
        )

        X_test_standard[
            numeric_columns
        ] = standard_scaler.transform(
            X_test[numeric_columns]
        )

        X_train_minmax[
            numeric_columns
        ] = minmax_scaler.fit_transform(
            X_train[numeric_columns]
        )

    return (
        X_train_standard,
        X_test_standard,
        X_train_minmax,
        standard_scaler,
        minmax_scaler
    )


# =========================================================
# MAIN FUNCTION
# =========================================================

def run_feature_engineering(
    missing_method=MISSING_METHOD
):

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    original_data = load_dataset()

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    (
        cleaned_data,
        missing_before,
        missing_after,
        rows_deleted,
        columns_deleted,
        values_imputed,
        missing_table
    ) = handle_missing_values(
        original_data,
        missing_method
    )

    # -----------------------------------------------------
    # IQR
    # -----------------------------------------------------

    iqr_result, cleaned_data = (
        detect_iqr_outliers(
            cleaned_data,
            IQR_COLUMN
        )
    )

    # -----------------------------------------------------
    # ENCODING
    # -----------------------------------------------------

    (
        encoded_data,
        categorical_cols,
        onehot_columns,
        onehot_preview,
        ordinal_columns,
        ordinal_preview,
        target_columns,
        target_preview,
        global_target_mean,
        embedding_columns,
        embedding_preview
    ) = feature_encoding(
        cleaned_data
    )

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        encoded_data
    )

    # -----------------------------------------------------
    # NUMERIC COLUMNS
    # -----------------------------------------------------

    numeric_cols = (
        X_train
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    # -----------------------------------------------------
    # BEFORE SCALING
    # -----------------------------------------------------

    before_scaling = (
        X_train[numeric_cols]
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # -----------------------------------------------------
    # SCALING
    # -----------------------------------------------------

    (
        X_train_scaled,
        X_test_scaled,
        X_train_minmax,
        standard_scaler,
        minmax_scaler
    ) = scale_features(
        X_train,
        X_test
    )

    # -----------------------------------------------------
    # PREVIEWS
    # -----------------------------------------------------

    standard_preview = (
        X_train_scaled[numeric_cols]
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    minmax_preview = (
        X_train_minmax[numeric_cols]
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # =====================================================
    # FINAL DICTIONARY
    # =====================================================

    preprocessing = {

        "n_rows":
            int(original_data.shape[0]),

        "n_columns":
            int(original_data.shape[1]),

        "missing_values": {

            "method":
                missing_method,

            "method_name":
                {
                    "row":
                        "Row-wise Deletion",

                    "column":
                        "Column-wise Deletion",

                    "mean":
                        "Mean Imputation",

                    "median":
                        "Median Imputation"
                }.get(
                    missing_method
                ),

            "missing_before":
                missing_before,

            "missing_after":
                missing_after,

            "rows_deleted":
                len(rows_deleted),

            "columns_deleted":
                len(columns_deleted)
        },

        "changes": {

            "rows_deleted":
                rows_deleted,

            "columns_deleted":
                columns_deleted,

            "values_imputed":
                values_imputed
        },

        "missing_table":
            missing_table,

        "iqr":
            iqr_result,

        "categorical_cols":
            categorical_cols,

        "onehot_columns":
            onehot_columns,

        "onehot_preview":
            onehot_preview,

        "ordinal_columns":
            ordinal_columns,

        "ordinal_preview":
            ordinal_preview,

        "target_columns":
            target_columns,

        "target_preview":
            target_preview,

        "global_target_mean":
            global_target_mean,

        "embedding_columns":
            embedding_columns,

        "embedding_preview":
            embedding_preview,

        "numeric_cols":
            numeric_cols,

        "before_scaling":
            before_scaling,

        "minmax_preview":
            minmax_preview,

        "standard_preview":
            standard_preview,

        "split": {

            "total":
                int(
                    X_train.shape[0]
                    + X_test.shape[0]
                ),

            "train_rows":
                int(X_train.shape[0]),

            "test_rows":
                int(X_test.shape[0]),

            "train_columns":
                int(X_train.shape[1]),

            "test_columns":
                int(X_test.shape[1])
        }
    }

    return {

        "data":
            encoded_data,

        "X_train":
            X_train,

        "X_test":
            X_test,

        "y_train":
            y_train,

        "y_test":
            y_test,

        "X_train_scaled":
            X_train_scaled,

        "X_test_scaled":
            X_test_scaled,

        "preprocessing":
            preprocessing
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = run_feature_engineering()

    print("\n======================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("======================================")

    print(
        "Rows:",
        result["preprocessing"]["n_rows"]
    )

    print(
        "Columns:",
        result["preprocessing"]["n_columns"]
    )

    print(
        "Training:",
        result["preprocessing"]["split"]["train_rows"]
    )

    print(
        "Testing:",
        result["preprocessing"]["split"]["test_rows"]
    )