
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 23


def load_adult(path: str | None) -> pd.DataFrame:
    if path:
        frame = pd.read_csv(path)
        required = {"income", "sex", "race"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        return frame
    try:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    except TypeError:
        data = fetch_openml("adult", version=2, as_frame=True)
    frame = data.data.copy()
    frame["income"] = data.target.astype(str)
    return frame


def fairness_summary(y_true: pd.Series, y_pred: np.ndarray, sensitive: pd.Series) -> dict[str, float]:
    table = pd.DataFrame({"truth": np.asarray(y_true), "prediction": y_pred, "group": np.asarray(sensitive)})
    rates = table.groupby("group", observed=False).prediction.mean()
    positive_groups = table[table.truth == 1].groupby("group", observed=False).prediction.mean()
    dp_diff = float(rates.max() - rates.min()) if len(rates) else 0.0
    dp_ratio = float(rates.min() / rates.max()) if len(rates) and rates.max() > 0 else 0.0
    eo_diff = float(positive_groups.max() - positive_groups.min()) if positive_groups.notna().any() else 0.0
    return {"demographic_parity_difference": dp_diff, "disparate_impact_ratio": dp_ratio,
            "equal_opportunity_difference": eo_diff}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Local Adult-style CSV")
    parser.add_argument("--output-dir", default="./practical8_output")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = load_adult(args.csv).replace("?", np.nan).dropna(subset=["income", "sex", "race"])
    target_text = frame.pop("income").astype(str).str.strip().str.rstrip(".")
    y = target_text.str.lower().isin([">50k", "1", "true", "yes"]).astype(int)
    sex = frame["sex"].astype(str)
    race = frame["race"].astype(str)
    features = frame.drop(columns=["sex", "race"], errors="ignore")
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    classifier = Pipeline([("prepare", transformer), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))])
    x_fit, x_test, y_fit, y_test, sex_fit, sex_test, race_fit, race_test = train_test_split(
        features, y, sex, race, test_size=0.25, random_state=SEED, stratify=y)
    classifier.fit(x_fit, y_fit)
    baseline = classifier.predict(x_test)
    predictions = {"baseline": baseline}
    fairlearn_available = True
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
        for constraint, key in (("demographic_parity", "threshold_demographic_parity"),
                                ("equalized_odds", "threshold_equalized_odds")):
            mitigator = ThresholdOptimizer(estimator=classifier, constraints=constraint,
                                           objective="balanced_accuracy_score", predict_method="predict_proba",
                                           prefit=True)
            mitigator.fit(x_fit, y_fit, sensitive_features=sex_fit)
            predictions[key] = mitigator.predict(x_test, sensitive_features=sex_test)
    except ImportError:
        fairlearn_available = False
        print("Fairlearn is not installed; reporting baseline fairness only. Install it to run mitigation.")

    rows = []
    for name, pred in predictions.items():
        metrics = fairness_summary(y_test, pred, sex_test)
        metrics.update({"model": name, "accuracy": accuracy_score(y_test, pred),
                        "precision": precision_score(y_test, pred, zero_division=0),
                        "recall": recall_score(y_test, pred, zero_division=0),
                        "f1": f1_score(y_test, pred, zero_division=0)})
        rows.append(metrics)
    summary = pd.DataFrame(rows).set_index("model")
    # Supplementary race audit on the original model, separate from the sex mitigation.
    race_metrics = fairness_summary(y_test, baseline, race_test)
    summary.to_csv(out / "sex_fairness_comparison.csv")
    pd.DataFrame([{"sensitive_attribute": "race", **race_metrics}]).to_csv(out / "race_fairness_audit.csv", index=False)
    summary.to_json(out / "sex_fairness_comparison.json", indent=2)
    print("ADULT INCOME FAIRNESS STUDY | Student ID: 23AIML014")
    print(f"Rows: {len(frame)} | Fairlearn mitigation available: {fairlearn_available}")
    print(summary.round(4).to_string())
    print("\nRace audit of baseline:", race_metrics)
    group_table = pd.DataFrame({"sex": sex_test.to_numpy(), "truth": y_test.to_numpy(), "prediction": baseline})
    rates = group_table.groupby("sex").prediction.mean()
    ax = rates.plot(kind="bar", color=["#3976a8", "#d17a32"][:len(rates)], figsize=(6, 4), rot=0)
    ax.set(ylabel="Selection rate", title="Baseline positive prediction rate by sex", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(out / "baseline_selection_rates.png", dpi=180); plt.close()
    if fairlearn_available:
        frame_metrics = MetricFrame(metrics={"selection_rate": selection_rate, "true_positive_rate": true_positive_rate},
                                   y_true=y_test, y_pred=baseline, sensitive_features=sex_test)
        frame_metrics.by_group.to_csv(out / "baseline_group_metrics.csv")
    print(f"Saved fairness reports and chart under {out.resolve()}")


if __name__ == "__main__":
    main()
