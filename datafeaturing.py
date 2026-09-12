from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# ORIGINAL DATASET
# ALWAYS USED AS INPUT
# =========================================================

DATASET_PATH = (
    BASE_DIR /
    "placement_predict_50k Dataset (3)(in).csv"
)


# =========================================================
# CLEANED DATASET
# ONLY GENERATED AS OUTPUT
# NEVER USED AS INPUT
# =========================================================

CLEANED_DATASET_PATH = (
    BASE_DIR /
    "cleaned_placement_dataset.csv"
)


TARGET = "PlacementStatus"

MISSING_METHOD = "median"

IQR_COLUMN = "CodingTestScore"


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(

            f"Original dataset not found.\n"
            f"Expected location:\n"
            f"{DATASET_PATH}"

        )

    try:

        df = pd.read_csv(
            DATASET_PATH
        )

    except Exception as e:

        raise RuntimeError(

            f"Unable to read original dataset: {e}"

        )

    if df.empty:

        raise ValueError(

            "The original dataset is empty."

        )

    return df


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(
    data,
    method="median"
):

    df = data.copy()

    missing_before = int(
        df.isnull().sum().sum()
    )

    rows_deleted = []

    columns_deleted = []

    values_imputed = []


    # =====================================================
    # ROW-WISE DELETION
    # =====================================================

    if method == "row":

        rows_with_missing = (

            df[
                df.isnull().any(axis=1)
            ]

            .index

            .tolist()

        )

        rows_deleted = [

            int(index)

            for index in rows_with_missing

        ]

        df = df.dropna().copy()


    # =====================================================
    # COLUMN-WISE DELETION
    # =====================================================

    elif method == "column":

        columns_deleted = (

            df.columns[
                df.isnull().any()
            ]

            .tolist()

        )

        df = df.drop(

            columns=columns_deleted

        ).copy()


    # =====================================================
    # MEAN IMPUTATION
    # =====================================================

    elif method == "mean":

        numeric_columns = (

            df.select_dtypes(
                include=np.number
            )

            .columns

            .tolist()

        )

        for column in numeric_columns:

            count = int(

                df[column]

                .isnull()

                .sum()

            )

            if count > 0:

                value = (

                    df[column]

                    .mean()

                )

                df[column] = (

                    df[column]

                    .fillna(value)

                )

                values_imputed.append({

                    "column":
                        column,

                    "count":
                        count,

                    "method":
                        "Mean",

                    "value":
                        round(
                            float(value),
                            4
                        )

                })


        # Categorical columns use mode

        categorical_columns = (

            df.select_dtypes(
                exclude=np.number
            )

            .columns

            .tolist()

        )

        for column in categorical_columns:

            count = int(

                df[column]

                .isnull()

                .sum()

            )

            if count > 0:

                mode = (

                    df[column]

                    .mode()

                )

                if not mode.empty:

                    value = mode.iloc[0]

                    df[column] = (

                        df[column]

                        .fillna(value)

                    )

                    values_imputed.append({

                        "column":
                            column,

                        "count":
                            count,

                        "method":
                            "Mode",

                        "value":
                            str(value)

                    })


    # =====================================================
    # MEDIAN IMPUTATION
    # =====================================================

    elif method == "median":

        numeric_columns = (

            df.select_dtypes(
                include=np.number
            )

            .columns

            .tolist()

        )

        for column in numeric_columns:

            count = int(

                df[column]

                .isnull()

                .sum()

            )

            if count > 0:

                value = (

                    df[column]

                    .median()

                )

                df[column] = (

                    df[column]

                    .fillna(value)

                )

                values_imputed.append({

                    "column":
                        column,

                    "count":
                        count,

                    "method":
                        "Median",

                    "value":
                        round(
                            float(value),
                            4
                        )

                })


        # Categorical columns use mode

        categorical_columns = (

            df.select_dtypes(
                exclude=np.number
            )

            .columns

            .tolist()

        )

        for column in categorical_columns:

            count = int(

                df[column]

                .isnull()

                .sum()

            )

            if count > 0:

                mode = (

                    df[column]

                    .mode()

                )

                if not mode.empty:

                    value = mode.iloc[0]

                    df[column] = (

                        df[column]

                        .fillna(value)

                    )

                    values_imputed.append({

                        "column":
                            column,

                        "count":
                            count,

                        "method":
                            "Mode",

                        "value":
                            str(value)

                    })


    else:

        raise ValueError(

            "Invalid missing value method. "
            "Choose row, column, mean, or median."

        )


    # =====================================================
    # MISSING VALUES AFTER PROCESSING
    # =====================================================

    missing_after = int(

        df.isnull()

        .sum()

        .sum()

    )


    # =====================================================
    # CREATE MISSING VALUE TABLE
    # =====================================================

    missing_table = []


    for column in data.columns:

        before = int(

            data[column]

            .isnull()

            .sum()

        )

        if column in df.columns:

            after = int(

                df[column]

                .isnull()

                .sum()

            )

        else:

            after = "Column Deleted"


        missing_table.append({

            "column":
                str(column),

            "before":
                before,

            "after":
                after

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
# SAVE CLEANED DATASET
# =========================================================

def save_cleaned_dataset(df):

    try:

        df.to_csv(

            CLEANED_DATASET_PATH,

            index=False,

            encoding="utf-8"

        )

    except OSError as e:

        raise RuntimeError(

            f"Unable to save cleaned dataset.\n"
            f"Path: {CLEANED_DATASET_PATH}\n"
            f"Error: {e}"

        )

    return str(
        CLEANED_DATASET_PATH
    )


# =========================================================
# IQR OUTLIER DETECTION
# =========================================================

def detect_iqr_outliers(
    df,
    column
):

    if column not in df.columns:

        return None, df.copy()


    data = df.copy()


    series = pd.to_numeric(

        data[column],

        errors="coerce"

    )


    # If column contains no numeric values

    if series.notna().sum() == 0:

        return None, data


    q1 = series.quantile(
        0.25
    )

    q3 = series.quantile(
        0.75
    )


    iqr = q3 - q1


    lower_bound = (

        q1 -

        1.5 * iqr

    )


    upper_bound = (

        q3 +

        1.5 * iqr

    )


    outlier_mask = (

        (series < lower_bound)

        |

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


    # Create clipped column

    data[
        f"{column}_clipped"
    ] = clipped


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


    return (

        result,

        data

    )


# =========================================================
# FEATURE ENCODING
# =========================================================

def feature_encoding(df):

    data = df.copy()


    # =====================================================
    # FIND CATEGORICAL COLUMNS
    # =====================================================

    categorical_cols = (

        data.select_dtypes(

            include=[
                "object",
                "string",
                "category"
            ]

        )

        .columns

        .tolist()

    )


    # =====================================================
    # ONE-HOT ENCODING
    # =====================================================

    onehot_columns = []

    onehot_preview = []


    onehot_cols = [

        column

        for column in categorical_cols

        if column != TARGET

        and column not in [

            "CollegeTier",

            "CGPA_Tier"

        ]

    ]


    if onehot_cols:

        try:

            encoder = OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            )

        except TypeError:

            encoder = OneHotEncoder(

                handle_unknown="ignore",

                sparse=False

            )


        encoded = encoder.fit_transform(

            data[onehot_cols]

        )


        names = encoder.get_feature_names_out(

            onehot_cols

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


        onehot_columns = names.tolist()


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

                categories=[
                    categories
                ],

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

        if not pd.api.types.is_numeric_dtype(

            data[TARGET]

        ):

            # Explicit mapping is safer

            unique_values = (

                data[TARGET]

                .dropna()

                .unique()

                .tolist()

            )


            target_mapping = {

                value: index

                for index, value

                in enumerate(
                    unique_values
                )

            }


            data[TARGET] = (

                data[TARGET]

                .map(
                    target_mapping
                )

            )


            target_columns.append(
                TARGET
            )


        global_target_mean = round(

            float(
                data[TARGET].mean()
            ),

            4

        )


        target_preview = (

            data[[TARGET]]

            .head(5)

            .to_dict(
                orient="records"
            )

        )


    # =====================================================
    # EMBEDDING STYLE ENCODING
    # =====================================================

    embedding_columns = []

    embedding_preview = []


    remaining_categorical = (

        data.select_dtypes(

            include=[
                "object",
                "string",
                "category"
            ]

        )

        .columns

        .tolist()

    )


    for column in remaining_categorical:

        if column != TARGET:

            new_column = (

                f"{column}_ID"

            )


            data[new_column] = (

                data[column]

                .astype("category")

                .cat.codes

            )


            embedding_columns.append(
                new_column
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


    try:

        return train_test_split(

            X,

            y,

            test_size=0.30,

            random_state=42,

            stratify=y

        )


    except ValueError:

        return train_test_split(

            X,

            y,

            test_size=0.30,

            random_state=42

        )


# =========================================================
# FEATURE SCALING
# =========================================================

def scale_features(
    X_train,
    X_test
):

    numeric_columns = (

        X_train.select_dtypes(

            include=np.number

        )

        .columns

        .tolist()

    )


    standard_scaler = StandardScaler()

    minmax_scaler = MinMaxScaler()


    X_train_standard = X_train.copy()

    X_test_standard = X_test.copy()

    X_train_minmax = X_train.copy()

    X_test_minmax = X_test.copy()


    if numeric_columns:

        # Standard Scaling

        X_train_standard[numeric_columns] = (

            standard_scaler.fit_transform(

                X_train[
                    numeric_columns
                ]

            )

        )


        X_test_standard[numeric_columns] = (

            standard_scaler.transform(

                X_test[
                    numeric_columns
                ]

            )

        )


        # Min-Max Scaling

        X_train_minmax[numeric_columns] = (

            minmax_scaler.fit_transform(

                X_train[
                    numeric_columns
                ]

            )

        )


        X_test_minmax[numeric_columns] = (

            minmax_scaler.transform(

                X_test[
                    numeric_columns
                ]

            )

        )


    return (

        X_train_standard,

        X_test_standard,

        X_train_minmax,

        X_test_minmax,

        standard_scaler,

        minmax_scaler

    )


# =========================================================
# MAIN FEATURE ENGINEERING
# =========================================================

def run_feature_engineering(

    missing_method=MISSING_METHOD

):


    # =====================================================
    # STEP 1: LOAD ORIGINAL DATASET
    # =====================================================

    original_data = load_dataset()


    # =====================================================
    # STEP 2: HANDLE MISSING VALUES
    # =====================================================

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


    # =====================================================
    # STEP 3: SAVE CLEANED DATASET
    # =====================================================

    cleaned_dataset_file = (

        save_cleaned_dataset(

            cleaned_data

        )

    )


    # =====================================================
    # STEP 4: OUTLIER DETECTION
    # =====================================================

    (

        iqr_result,

        processed_data

    ) = detect_iqr_outliers(

        cleaned_data,

        IQR_COLUMN

    )


    # =====================================================
    # STEP 5: FEATURE ENCODING
    # =====================================================

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

        processed_data

    )


    # =====================================================
    # STEP 6: TRAIN TEST SPLIT
    # =====================================================

    (

        X_train,

        X_test,

        y_train,

        y_test

    ) = split_data(

        encoded_data

    )


    # =====================================================
    # STEP 7: NUMERIC COLUMNS
    # =====================================================

    numeric_cols = (

        X_train.select_dtypes(

            include=np.number

        )

        .columns

        .tolist()

    )


    # =====================================================
    # BEFORE SCALING PREVIEW
    # =====================================================

    if numeric_cols:

        before_scaling = (

            X_train[numeric_cols]

            .head(5)

            .round(4)

            .to_dict(
                orient="records"
            )

        )

    else:

        before_scaling = []


    # =====================================================
    # STEP 8: SCALING
    # =====================================================

    (

        X_train_scaled,

        X_test_scaled,

        X_train_minmax,

        X_test_minmax,

        standard_scaler,

        minmax_scaler

    ) = scale_features(

        X_train,

        X_test

    )


    # =====================================================
    # PREVIEWS
    # =====================================================

    if numeric_cols:

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

    else:

        standard_preview = []

        minmax_preview = []


    # =====================================================
    # PREPROCESSING RESULTS
    # =====================================================

    preprocessing = {


        "n_rows":

            int(
                original_data.shape[0]
            ),


        "n_columns":

            int(
                original_data.shape[1]
            ),


        "original_dataset_path":

            str(
                DATASET_PATH
            ),


        "cleaned_dataset_path":

            cleaned_dataset_file,


        "cleaned_rows":

            int(
                cleaned_data.shape[0]
            ),


        "cleaned_columns":

            int(
                cleaned_data.shape[1]
            ),


        # Missing values

        "missing_values": {

            "method":

                missing_method,


            "method_name": {

                "row":
                    "Row-wise Deletion",

                "column":
                    "Column-wise Deletion",

                "mean":
                    "Mean Imputation",

                "median":
                    "Median Imputation"

            }.get(

                missing_method,

                missing_method

            ),


            "missing_before":

                missing_before,


            "missing_after":

                missing_after,


            "rows_deleted":

                len(
                    rows_deleted
                ),


            "columns_deleted":

                len(
                    columns_deleted
                )

        },


        # Changes

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


        # Outliers

        "iqr":

            iqr_result,


        # Encoding

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


        # Scaling

        "numeric_cols":

            numeric_cols,


        "before_scaling":

            before_scaling,


        "minmax_preview":

            minmax_preview,


        "standard_preview":

            standard_preview,


        # Split

        "split": {

            "total":

                int(
                    X_train.shape[0]
                    +
                    X_test.shape[0]
                ),


            "train_rows":

                int(
                    X_train.shape[0]
                ),


            "test_rows":

                int(
                    X_test.shape[0]
                ),


            "train_columns":

                int(
                    X_train.shape[1]
                ),


            "test_columns":

                int(
                    X_test.shape[1]
                )

        }

    }


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "original_data":

            original_data,


        "cleaned_data":

            cleaned_data,


        "processed_data":

            processed_data,


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


        "X_train_minmax":

            X_train_minmax,


        "X_test_minmax":

            X_test_minmax,


        "preprocessing":

            preprocessing

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    try:

        result = run_feature_engineering()

        print(
            "\nFEATURE ENGINEERING COMPLETED"
        )

        print(
            "\nOriginal Dataset:"
        )

        print(
            result["preprocessing"]
            ["original_dataset_path"]
        )

        print(
            "\nCleaned Dataset Generated:"
        )

        print(
            result["preprocessing"]
            ["cleaned_dataset_path"]
        )

        print(
            "\nOriginal Rows:",
            result["preprocessing"]["n_rows"]
        )

        print(
            "Cleaned Rows:",
            result["preprocessing"]["cleaned_rows"]
        )

    except Exception as e:

        import traceback

        print("\nERROR OCCURRED:\n")

        traceback.print_exc()