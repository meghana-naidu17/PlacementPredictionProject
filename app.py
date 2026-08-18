from flask import Flask, render_template, request

from load_data import get_data_summary
from placement_eda import run_eda
from datafeaturing import run_preprocessing


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"


    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"


    return render_template(
        "eda.html",
        active="eda",
        results=results,
        error=error
    )


# ============================================================
# DATA PREPROCESSING
# ============================================================

@app.route(
    "/preprocessing",
    methods=["GET", "POST"]
)
def preprocessing_page():

    error = None
    preprocessing = None


    try:

        # ----------------------------------------------------
        # DEFAULT METHOD
        # ----------------------------------------------------

        missing_method = "median"


        # ----------------------------------------------------
        # GET SELECTED METHOD FROM HTML
        # ----------------------------------------------------

        if request.method == "POST":

            missing_method = request.form.get(
                "missing_method",
                "median"
            )


        # ----------------------------------------------------
        # RUN PREPROCESSING
        # ----------------------------------------------------

        preprocessing = run_preprocessing(
            missing_method
        )


        print("\n" + "=" * 80)

        print("PREPROCESSING RESULTS")

        print("=" * 80)

        print(
            "Selected Method:",
            missing_method
        )

        print(
            preprocessing.keys()
        )


    except FileNotFoundError as e:

        error = str(e)

        print(
            "FILE ERROR:",
            error
        )


    except Exception as e:

        error = f"Unexpected error: {e}"

        print(
            "PREPROCESSING ERROR:",
            error
        )


    # --------------------------------------------------------
    # RETURN PAGE
    # --------------------------------------------------------

    return render_template(

        "datafeaturing.html",

        active="preprocessing",

        preprocessing=preprocessing,

        error=error

    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )