print("CORRECT APP.PY IS RUNNING")
from flask import Flask, render_template, request, jsonify

from load_data import (
    get_data_summary,
    get_data_page
)

from placement_eda import run_eda

from datafeaturing import run_feature_engineering

from linear_regression import run_linear_regression

from logistic_regression import run_logistic_regression

from treebased import run_tree_based

from kmeans import run_kmeans


app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active="none"
    )


@app.route("/index")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    try:

        summary = get_data_summary()

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# DATASET API
# =========================================================

@app.route("/api/dataset")
def dataset_api():

    try:

        page = int(
            request.args.get(
                "page",
                1
            )
        )

        per_page = int(
            request.args.get(
                "per_page",
                20
            )
        )

        page = max(page, 1)
        per_page = max(per_page, 1)
        per_page = min(per_page, 100)

        data = get_data_page(
            page=page,
            per_page=per_page
        )

        return jsonify(data)

    except Exception as e:

        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}"
        }), 500


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda_page():

    try:

        results = run_eda()

        return render_template(
            "eda.html",
            active="eda",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "eda.html",
            active="eda",
            results=None,
            error=f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route(
    "/preprocessing",
    methods=["GET", "POST"]
)
def preprocessing_page():

    missing_method = "median"

    if request.method == "POST":

        missing_method = request.form.get(
            "missing_method",
            "median"
        )

    try:

        print("STEP 1: Starting feature engineering")

        result = run_feature_engineering(
            missing_method=missing_method
        )

        print("STEP 2: Feature engineering completed")

        preprocessing_result = result.get(
            "preprocessing",
            {}
        )

        print("STEP 3: Rendering template")

        return render_template(
            "datafeaturing.html",
            active="preprocessing",
            preprocessing=preprocessing_result,
            error=None
        )

    except Exception as e:

        import traceback

        print("\nPREPROCESSING ERROR:")
        traceback.print_exc()

        return render_template(
            "datafeaturing.html",
            active="preprocessing",
            preprocessing=None,
            error=f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route(
    "/linear-regression",
    methods=["GET", "POST"]
)
def linear_regression_page():

    results = None
    error = None
    selected_regularization = "none"

    if request.method == "POST":

        selected_regularization = request.form.get(
            "regularization",
            "none"
        )

        try:

            results = run_linear_regression(
                regularization=selected_regularization
            )

            if not results:

                error = "No result was returned."

            elif results.get("error"):

                error = results["error"]
                results = None

        except Exception as e:

            error = f"{type(e).__name__}: {str(e)}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        selected_regularization=selected_regularization,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route(
    "/logistic-regression",
    methods=["GET", "POST"]
)
def logistic_regression_page():

    results = None
    error = None
    selected_regularization = "none"

    if request.method == "POST":

        selected_regularization = request.form.get(
            "regularization",
            "none"
        )

        try:

            results = run_logistic_regression(
                regularization=selected_regularization
            )

            if not results:

                error = "No result was returned."

            elif results.get("error"):

                error = results["error"]
                results = None

        except Exception as e:

            error = f"{type(e).__name__}: {str(e)}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        logistic=results,
        selected_regularization=selected_regularization,
        error=error
    )


# =========================================================
# MACHINE LEARNING
# =========================================================

@app.route("/machine-learning")
def machine_learning_page():

    return render_template(
        "index.html",
        active="machine-learning"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/prediction")
def prediction_page():

    return render_template(
        "index.html",
        active="prediction"
    )


# =========================================================
# TREE BASED MODELS
# =========================================================

@app.route(
    "/tree-based",
    methods=["GET", "POST"]
)
def tree_based_page():

    results = None
    error = None
    selected_model = None

    if request.method == "POST":

        selected_model = request.form.get(
            "model"
        )

        if not selected_model:

            error = (
                "Please select a tree-based algorithm."
            )

        else:

            try:

                results = run_tree_based(
                    selected_model
                )

                if results and results.get("error"):

                    error = results["error"]
                    results = None

            except Exception as e:

                error = (
                    f"{type(e).__name__}: {str(e)}"
                )

    return render_template(
        "treebased.html",
        active="tree-based",
        results=results,
        selected_model=selected_model,
        error=error
    )

# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route(
    "/kmeans",
    methods=["GET", "POST"]
)
def kmeans_page():

    results = None
    error = None

    selected_method = None
    selected_k = None

    if request.method == "POST":

        selected_method = request.form.get(
            "method"
        )

        # -------------------------------------------------
        # ELBOW GRAPH
        # -------------------------------------------------

        if selected_method == "elbow_graph":

            try:

                results = run_kmeans(
                    method="elbow",
                    k=None
                )

                if results.get("error"):

                    error = results["error"]
                    results = None

                else:

                    selected_method = "elbow"

            except Exception as e:

                error = (
                    f"{type(e).__name__}: {str(e)}"
                )


        # -------------------------------------------------
        # ELBOW FINAL CALCULATION
        # -------------------------------------------------

        elif selected_method == "elbow_calculate":

            k_value = request.form.get(
                "k",
                ""
            ).strip()

            if not k_value:

                error = (
                    "Please enter the K value "
                    "after observing the Elbow graph."
                )

            else:

                try:

                    selected_k = int(
                        k_value
                    )

                    if selected_k < 2:

                        error = (
                            "K must be at least 2."
                        )

                    else:

                        results = run_kmeans(
                            method="elbow",
                            k=selected_k
                        )

                        if results.get("error"):

                            error = results["error"]
                            results = None

                        selected_method = "elbow"

                except ValueError:

                    error = (
                        "Please enter a valid integer K."
                    )


        # -------------------------------------------------
        # MANUAL K-MEANS
        # -------------------------------------------------

        elif selected_method == "manual":

            k_value = request.form.get(
                "k",
                ""
            ).strip()

            if not k_value:

                error = (
                    "Please enter the number of clusters K."
                )

            else:

                try:

                    selected_k = int(
                        k_value
                    )

                    results = run_kmeans(
                        method="manual",
                        k=selected_k
                    )

                    if results.get("error"):

                        error = results["error"]
                        results = None

                except ValueError:

                    error = (
                        "Please enter a valid integer K."
                    )


        # -------------------------------------------------
        # SILHOUETTE
        # -------------------------------------------------

        elif selected_method == "silhouette":

            try:

                results = run_kmeans(
                    method="silhouette",
                    k=None
                )

                if results.get("error"):

                    error = results["error"]
                    results = None

            except Exception as e:

                error = (
                    f"{type(e).__name__}: {str(e)}"
                )


        else:

            error = (
                "Please select a K-Means method."
            )


    return render_template(
        "kmeans.html",
        active="kmeans",
        results=results,
        selected_method=selected_method,
        selected_k=selected_k,
        error=error
    )

# =========================================================
# RUN APPLICATION
# =========================================================
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
        use_reloader=False
    )