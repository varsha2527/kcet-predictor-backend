
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the cutoff dataset
DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "kcet_cutoffs.csv"
)

df = pd.read_csv(DATA_FILE)

categories = [
    "1G", "1K", "1R",
    "2AG", "2AK", "2AR",
    "2BG", "2BK", "2BR",
    "3AG", "3AK", "3AR",
    "3BG", "3BK", "3BR",
    "GM", "GMK", "GMR",
    "SCG", "SCK", "SCR",
    "STG", "STK", "STR"
]

# Check that the dataset has the required columns
required_columns = ["college", "course"] + categories

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )

# Convert cutoff columns to numeric values
for category in categories:
    df[category] = pd.to_numeric(
        df[category], errors="coerce"
    )


def predict_colleges(
    rank,
    category,
    branch=None,
    college_search=None
):
    category = category.upper().strip()

    if category not in categories:
        raise ValueError("Invalid reservation category.")

    if rank <= 0:
        raise ValueError("Rank must be a positive number.")

    results = df[df[category].notna()].copy()

    # Lower numerical rank is better.
    # A historical closing rank >= the student's rank
    # indicates a potentially eligible option.
    results = results[results[category] >= rank]

    if branch and branch.upper() != "ALL BRANCHES":
        results = results[
            results["course"].str.contains(
                branch, case=False, na=False, regex=False
            )
        ]

    if college_search:
        results = results[
            results["college"].str.contains(
                college_search,
                case=False,
                na=False,
                regex=False
            )
        ]

    results = results[
        ["college", "course", category]
    ].rename(columns={category: "closing_rank"})

    results = results.sort_values(
        "closing_rank"
    ).reset_index(drop=True)

    return results


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "KCET College Predictor API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        rank = int(data.get("rank", 0))
        category = str(data.get("category", "GM"))
        branch = data.get("branch")
        college = data.get("college", "")

        results = predict_colleges(
            rank=rank,
            category=category,
            branch=branch or None,
            college_search=college or None
        )

        return jsonify({
            "count": len(results),
            "results": results.to_dict(orient="records")
        })

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

