import os
import pandas as pd


# =========================================================
# DATASET PATH
# =========================================================

DATA_PATH = r"C:\Users\Abhiii\Downloads\PythonProject3\placement_predict_50k Dataset (3)(in).csv"


# =========================================================
# LOAD DATASET
# =========================================================

def load_data(path=DATA_PATH):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    try:

        df = pd.read_csv(path)

    except Exception as e:

        raise Exception(
            f"Error reading CSV file: {e}"
        )

    return df


# =========================================================
# DATASET SUMMARY
# =========================================================

def get_data_summary():

    df = load_data()

    numerical_df = df.select_dtypes(
        include="number"
    )

    categorical_df = df.select_dtypes(
        include="object"
    )

    summary = {

        "n_rows": int(
            df.shape[0]
        ),

        "n_columns": int(
            df.shape[1]
        ),

        "total_cells": int(
            df.size
        ),

        "memory_usage": round(
            df.memory_usage(
                deep=True
            ).sum() / 1024,
            2
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "columns": list(
            df.columns
        ),

        "dtypes": {

            col: str(dtype)

            for col, dtype
            in df.dtypes.items()

        },

        "missing_counts": {

            col: int(
                df[col].isnull().sum()
            )

            for col in df.columns

        },

        "missing_percentages": {

            col: round(
                (
                    df[col].isnull().sum()
                    / len(df)
                ) * 100,
                2
            )

            for col in df.columns

        },

        "unique_values": {

            col: int(
                df[col].nunique()
            )

            for col in df.columns

        },

        "numerical_summary": (

            numerical_df
            .describe()
            .round(2)
            .to_dict()

        ),

        "categorical_summary": (

            categorical_df
            .describe()
            .to_dict()

        ),

        "correlation": (

            numerical_df
            .corr()
            .round(2)
            .to_dict()

        )

    }

    return summary


# =========================================================
# GET DATASET PAGE
# =========================================================

def get_data_page(
    page=1,
    per_page=20
):

    df = load_data()

    total_rows = len(df)

    # -----------------------------------------------------
    # Calculate total number of pages
    # -----------------------------------------------------

    total_pages = (
        total_rows + per_page - 1
    ) // per_page


    # -----------------------------------------------------
    # Make sure page is valid
    # -----------------------------------------------------

    if page < 1:

        page = 1


    if page > total_pages:

        page = total_pages


    # -----------------------------------------------------
    # Calculate row positions
    # -----------------------------------------------------

    start = (
        page - 1
    ) * per_page

    end = start + per_page


    # -----------------------------------------------------
    # Get required rows
    # -----------------------------------------------------

    page_df = df.iloc[
        start:end
    ].copy()


    # =====================================================
    # IMPORTANT JSON FIX
    #
    # Pandas uses NaN for missing values.
    # NaN is NOT valid JSON.
    #
    # Convert:
    #
    # NaN       -> None
    # Infinity  -> None
    # -Infinity -> None
    #
    # JavaScript will receive these as null.
    # =====================================================

    page_df = page_df.astype(
        object
    )


    page_df = page_df.where(
        pd.notna(page_df),
        None
    )


    # -----------------------------------------------------
    # Convert infinity values
    # -----------------------------------------------------

    page_df = page_df.replace(
        [
            float("inf"),
            float("-inf")
        ],
        None
    )


    # -----------------------------------------------------
    # Convert dataframe into JSON-safe records
    # -----------------------------------------------------

    records = page_df.to_dict(
        orient="records"
    )


    # -----------------------------------------------------
    # Return dataset information
    # -----------------------------------------------------

    return {

        "columns": list(
            df.columns
        ),

        "data": records,

        "page": page,

        "per_page": per_page,

        "total_rows": total_rows,

        "total_pages": total_pages,

        "start_row": start + 1,

        "end_row": min(
            end,
            total_rows
        )

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)


    summary = get_data_summary()


    print(
        "Rows:",
        summary["n_rows"]
    )


    print(
        "Columns:",
        summary["n_columns"]
    )


    print(
        "Total Cells:",
        summary["total_cells"]
    )


    print(
        "Duplicate Rows:",
        summary["duplicate_rows"]
    )


    print(
        "Memory Usage:",
        summary["memory_usage"],
        "KB"
    )


    print("=" * 70)


    # Test dataset page

    data = get_data_page(
        page=1,
        per_page=20
    )


    print(
        "Showing rows:",
        data["start_row"],
        "to",
        data["end_row"]
    )


    print(
        "Total Pages:",
        data["total_pages"]
    )


    print("=" * 70)