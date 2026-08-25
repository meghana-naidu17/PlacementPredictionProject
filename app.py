from flask import Flask, render_template, request, jsonify

from load_data import get_data_summary, get_data_page
from placement_eda import run_eda
from datafeaturing import run_feature_engineering
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression


app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
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
            error=str(e)
        )


# =========================================================
# DATASET API
# =========================================================

@app.route("/api/dataset")
def dataset_api():

    try:

        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))

        if page < 1:
            page = 1

        if per_page < 1:
            per_page = 20

        if per_page > 100:
            per_page = 100

        data = get_data_page(
            page=page,
            per_page=per_page
        )

        return jsonify(data)

    except Exception as e:

        return jsonify({
            "error": str(e)
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
            error=str(e)
        )


# =========================================================
# DATA PREPROCESSING
# =========================================================

@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing_page():

    try:

        missing_method = "median"

        if request.method == "POST":

            missing_method = request.form.get(
                "missing_method",
                "median"
            )

        result = run_feature_engineering(
            missing_method=missing_method
        )

        return render_template(
            "datafeaturing.html",
            active="preprocessing",
            preprocessing=result["preprocessing"],
            error=None
        )

    except Exception as e:

        return render_template(
            "datafeaturing.html",
            active="preprocessing",
            preprocessing=None,
            error=str(e)
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression_page():

    try:

        regression = run_linear_regression()

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=regression,
            error=None
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=None,
            error=str(e)
        )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    try:

        results = run_logistic_regression()

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=None,
            error=str(e)
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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )