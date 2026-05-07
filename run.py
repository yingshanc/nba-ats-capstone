import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import create_features
from src.model import get_model
from src.evaluation import evaluate_model


DATA_PATH = "data/raw/nba_2008-2025.csv"
RESULTS_PATH = Path("experiments/results.tsv")
RESULT_COLUMNS = [
    "timestamp",
    "description",
    "accuracy",
    "brier_score",
    "log_loss",
    "model_summary",
    "feature_summary",
    "changed_files",
    "notes",
]


def calculate_ats_cover(row):
    margin_home = row["score_home"] - row["score_away"]
    margin_away = row["score_away"] - row["score_home"]

    if row["whos_favored"] == "home":
        return 1 if margin_home > row["spread"] else 0
    else:
        return 1 if margin_away > row["spread"] else 0


def summarize_model(model):
    return model.__class__.__name__


def summarize_features(features):
    return ",".join(features)


def summarize_changed_files():
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

    files = []
    for line in status:
        if line.strip():
            files.append(line[3:].strip())
    return ",".join(files)


def log_experiment(description, results, model, features):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": description,
        "accuracy": f"{results['accuracy']:.6f}",
        "brier_score": f"{results['brier_score']:.6f}",
        "log_loss": f"{results['log_loss']:.6f}",
        "model_summary": summarize_model(model),
        "feature_summary": summarize_features(features),
        "changed_files": summarize_changed_files(),
        "notes": "",
    }

    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    description = " ".join(sys.argv[1:]).strip() or "no description provided"

    df = pd.read_csv(DATA_PATH)

    df["fav_covered"] = df.apply(calculate_ats_cover, axis=1)

    df = create_features(df)

    features = ["spread_abs", "fav_home", "fav_rolling_5_point_diff"]
    print(f"Final feature list: {features}")

    X = df[features]
    y = df["fav_covered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = get_model()
    print(f"Model used: {type(model).__name__}")

    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)

    print("Experiment Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    log_experiment(description, results, model, features)
    print("Logged experiment to experiments/results.tsv")


if __name__ == "__main__":
    main()
