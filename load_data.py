import os
import pandas as pd

DATA_PATH = r"C:\Users\Abhiii\Downloads\PythonProject3\placement_predict_50k Dataset (3)(in).csv"


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def get_data_summary():
    df = load_data()

    summary = {

        # Dataset information
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "total_cells": df.size,
        "memory_usage": round(df.memory_usage(deep=True).sum() / 1024, 2),   # KB
        "duplicate_rows": int(df.duplicated().sum()),

        # Column information
        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        # Missing values
        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },

        "missing_percentages": {
            col: round((df[col].isnull().sum()/len(df))*100,2)
            for col in df.columns
        },

        # Unique values
        "unique_values": {
            col: int(df[col].nunique())
            for col in df.columns
        },

        # Numerical statistics
        "numerical_summary":
            df.describe().round(2).to_dict(),

        # Categorical statistics
        "categorical_summary":
            df.describe(include=["object"]).to_dict(),

        # Correlation matrix
        "correlation":
            df.select_dtypes(include="number")
              .corr()
              .round(2)
              .to_dict(),

        # First 10 rows
        "preview":
            df.head(10).to_dict(orient="records")
    }

    return summary


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_data_summary())