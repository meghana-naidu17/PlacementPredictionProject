import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

sns.set_style("whitegrid")
sns.set_context("paper")

CHART_DIR = os.path.join("static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, filename), dpi=120)
    plt.close()


def run_eda():

    data = load_data()

    results = {}

    charts = []

    # ============================================================
    # 1. DATA LOADING
    # ============================================================

    results["shape"] = data.shape
    results["head"] = data.head().to_dict(orient="records")

    # ============================================================
    # 2. BASIC INFORMATION
    # ============================================================

    results["dtypes"] = data.dtypes.astype(str).to_dict()

    results["describe"] = (
        data.describe(include="all")
        .fillna("")
        .to_dict()
    )

    # ============================================================
    # 3. MISSING VALUES
    # ============================================================

    missing = data.isnull().sum()

    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({

        "Missing Count": missing,

        "Missing %": missing_pct

    })

    results["missing"] = missing_df.to_dict()

    if missing.sum() > 0:

        plt.figure(figsize=(10,5))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["Missing %"]
        )

        plt.xticks(rotation=45)

        plt.title("Missing Values")

        save_chart("missing_values.png")

        charts.append("missing_values.png")


        plt.figure(figsize=(12,6))

        sns.heatmap(
            data.isnull(),
            cmap="viridis",
            cbar=False
        )

        plt.title("Missing Value Heatmap")

        save_chart("missing_heatmap.png")

        charts.append("missing_heatmap.png")

    # ============================================================
    # 4. DUPLICATES
    # ============================================================

    results["duplicates"] = int(data.duplicated().sum())

    # ============================================================
    # 5. TARGET VARIABLE
    # ============================================================

    if "PlacementStatus" in data.columns:
        results["target_counts"] = (
            data["PlacementStatus"]
            .value_counts()
            .to_dict()
        )

        plt.figure(figsize=(6, 4))

        sns.countplot(
            x="PlacementStatus",
            data=data
        )

        plt.title("Placement Status")

        save_chart("placement_status.png")

        charts.append("placement_status.png")
    # ============================================================
    # 6. NUMERIC FEATURE DISTRIBUTION
    # ============================================================

    numeric_cols = data.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        try:
            plt.figure(figsize=(7, 4))

            sns.histplot(data[col], bins=20, kde=True)

            plt.title(col)

            safe_col = (
                col.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("%", "Percent")
            )

            filename = f"hist_{safe_col}.png"

            save_chart(filename)

            charts.append(filename)

        except Exception as e:
            print(f"Histogram Error ({col}): {e}")

    # ============================================================
    # 7. BOXPLOTS
    # ============================================================

    for col in numeric_cols:

        plt.figure(figsize=(7,4))

        sns.boxplot(
            x=data[col],
            color="skyblue"
        )

        plt.title(f"Boxplot - {col}")

        safe_col = (
            col.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("%", "Percent")
        )

        filename = f"box_{safe_col}.png"

        save_chart(filename)

        charts.append(filename)

    # ============================================================
    # 8. CORRELATION
    # ============================================================

    corr = data.select_dtypes(include=np.number).corr()

    results["correlation"] = corr.round(2).to_dict()

    plt.figure(figsize=(14,10))

    sns.heatmap(

        corr,

        annot=True,

        fmt=".2f",

        cmap="PuOr"

    )

    plt.title("Correlation Heatmap")

    save_chart("correlation_heatmap.png")

    charts.append("correlation_heatmap.png")
    # ============================================================
    # 9. RELATIONSHIP PLOTS
    # ============================================================

    for col in numeric_cols:

        if col == "PlacementStatus":
            continue

        plt.figure(figsize=(7,5))

        sns.boxplot(
            data=data,
            x="PlacementStatus",
            y=col,
            palette="viridis"
        )

        plt.title(f"{col} vs Placement Status")

        safe_col = (
            col.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("%", "Percent")
        )

        filename = f"relationship_{safe_col}.png"

        save_chart(filename)

        charts.append(filename)

    # ============================================================
    # 10. CATEGORICAL FEATURE COUNTS
    # ============================================================

    categorical_cols = [

        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"

    ]

    categorical_cols = [

        col for col in categorical_cols

        if col in data.columns

    ]

    for col in categorical_cols:

        results[f"{col}_counts"] = data[col].value_counts().to_dict()

        plt.figure(figsize=(8,5))

        sns.countplot(
            data=data,
            x=col,
            color="skyblue"
        )

        plt.xticks(rotation=45)

        plt.title(f"{col} Count")

        safe_col = (
            col.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("%", "Percent")
        )

        filename = f"count_{safe_col}.png"

        save_chart(filename)

        charts.append(filename)

    # ============================================================
    # 11. GENDER VS PLACEMENT
    # ============================================================

    if "Gender" in data.columns:

        plt.figure(figsize=(7,5))

        sns.countplot(
            data=data,
            x="Gender",
            hue="PlacementStatus"
        )

        plt.title("Gender vs Placement Status")

        save_chart("gender_vs_placement.png")

        charts.append("gender_vs_placement.png")

    # ============================================================
    # 12. COLLEGE TIER & STREAM VS PLACEMENT
    # ============================================================

    if "CollegeTier" in data.columns:

        plt.figure(figsize=(7,5))

        sns.countplot(
            data=data,
            x="CollegeTier",
            hue="PlacementStatus"
        )

        plt.title("College Tier vs Placement")

        save_chart("tier_vs_placement.png")

        charts.append("tier_vs_placement.png")


    if "Stream" in data.columns:

        plt.figure(figsize=(10,5))

        sns.countplot(
            data=data,
            x="Stream",
            hue="PlacementStatus"
        )

        plt.xticks(rotation=45)

        plt.title("Stream vs Placement")

        save_chart("stream_vs_placement.png")

        charts.append("stream_vs_placement.png")

    # ============================================================
    # 13. SGPA TREND
    # ============================================================

    sgpa_cols = [

        f"SGPA_Sem{i}"

        for i in range(1,9)

        if f"SGPA_Sem{i}" in data.columns

    ]

    if sgpa_cols:

        avg_sgpa = data[sgpa_cols].mean()

        plt.figure(figsize=(8,5))

        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o",
            linewidth=2
        )

        plt.grid(True)

        plt.title("Average SGPA Across Semesters")

        plt.xlabel("Semester")

        plt.ylabel("Average SGPA")

        save_chart("sgpa_trend.png")

        charts.append("sgpa_trend.png")

    # ============================================================
    # 14. SALARY PACKAGE ANALYSIS
    # ============================================================

    if "Salary Package" in data.columns and "PlacementStatus" in data.columns:

        placed = data[data["PlacementStatus"] == 1]

        plt.figure(figsize=(8,5))

        sns.histplot(
            data=placed,
            x="Salary Package",
            bins=20,
            kde=True,
            color="green"
        )

        plt.title("Salary Distribution (Placed Students)")

        save_chart("salary_distribution.png")

        charts.append("salary_distribution.png")

        if "CollegeTier" in placed.columns:

            plt.figure(figsize=(8,5))

            sns.boxplot(
                data=placed,
                x="CollegeTier",
                y="Salary Package",
                palette="Set2"
            )

            plt.title("Salary by College Tier")

            save_chart("salary_vs_collegetier.png")

            charts.append("salary_vs_collegetier.png")

    # ============================================================
    # 15. PAIRPLOT
    # ============================================================

    pair_cols = [

        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "PlacementStatus"

    ]

    pair_cols = [

        col for col in pair_cols

        if col in data.columns

    ]

    if len(pair_cols) >= 2:

        g = sns.pairplot(
            data=data[pair_cols],
            hue="PlacementStatus",
            diag_kind="hist",
            corner=True,
            palette="Set1"
        )

        g.figure.suptitle(
            "Pairwise Relationships",
            y=1.02
        )

        g.savefig(os.path.join(CHART_DIR, "pairplot.png"))

        plt.close()

        charts.append("pairplot.png")

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    results["charts"] = charts

    results["rows"] = data.shape[0]
    results["columns"] = data.shape[1]

    results["missing"] = data.isnull().sum().to_dict()

    results["duplicates"] = int(data.duplicated().sum())

    if "PlacementStatus" in data.columns:
        results["target_counts"] = (
            data["PlacementStatus"]
            .value_counts()
            .to_dict()
        )

    results["charts"] = charts
    return results