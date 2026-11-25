
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 23


def load_data(csv_path: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path:
        frame = pd.read_csv(csv_path)
        target = next((col for col in ("class", "target", "credit_risk") if col in frame.columns), None)
        if target is None:
            raise ValueError("CSV needs a target column named class, target, or credit_risk")
        labels = frame.pop(target)
        return frame, labels
    try:
        bunch = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    except TypeError:  # Compatibility with older scikit-learn releases.
        bunch = fetch_openml(name="credit-g", version=1, as_frame=True)
    return bunch.data, bunch.target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Optional local credit dataset CSV")
    parser.add_argument("--output-dir", default="./practical6_output")
    parser.add_argument("--trees", type=int, default=250)
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    x, y_raw = load_data(args.csv)
    y = (pd.Series(y_raw).astype(str).str.lower().isin(["good", "1", "yes", "true"])).astype(int)
    x = x.replace("?", np.nan).copy()
    categorical = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = x.select_dtypes(include=["number"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median"))]), numeric),
    ], remainder="drop")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y)
    model = Pipeline([("preprocess", preprocess),
                      ("forest", RandomForestClassifier(n_estimators=args.trees, max_depth=8,
                                                        min_samples_leaf=2, class_weight="balanced",
                                                        random_state=SEED, n_jobs=-1))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    print("CREDIT RISK MODEL | Student ID: 23AIML014")
    print(f"Rows: {len(x)} | Features: {x.shape[1]} | Positive label: creditworthy/good")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f} | ROC AUC: {roc_auc_score(y_test, probability):.4f}")
    print(classification_report(y_test, pred, target_names=["Higher risk", "Creditworthy"], zero_division=0))
    names = model.named_steps["preprocess"].get_feature_names_out()
    transformed_test = model.named_steps["preprocess"].transform(x_test)
    forest = model.named_steps["forest"]
    importances = pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(out / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    shown = importances.head(15).sort_values("importance")
    ax.barh(shown.feature.str.replace("categorical__", "", regex=False).str.replace("numeric__", "", regex=False),
            shown.importance, color="#3572a5")
    ax.set(xlabel="Random forest feature importance", title="Global credit model drivers")
    fig.tight_layout(); fig.savefig(out / "global_feature_importance.png", dpi=180); plt.close(fig)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["actual_risk", "actual_good"], columns=["predicted_risk", "predicted_good"]).to_csv(out / "confusion_matrix.csv")

    # Explain one held-out applicant with local model-agnostic perturbations.
    row = x_test.iloc[[0]].copy()
    rng = np.random.default_rng(SEED)
    neighborhood = pd.concat([row] * 800, ignore_index=True)
    for column in x.columns:
        if column in categorical:
            values = x_train[column].dropna().unique()
            if len(values):
                neighborhood[column] = rng.choice(values, size=len(neighborhood))
        else:
            vals = pd.to_numeric(x_train[column], errors="coerce").dropna().to_numpy()
            if len(vals):
                neighborhood[column] = rng.choice(vals, size=len(neighborhood))
    dist = np.zeros(len(neighborhood))
    for col in x.columns:
        reference = row.iloc[0][col]
        if pd.isna(reference):
            if col in categorical:
                modes = x_train[col].mode(dropna=True)
                reference = modes.iloc[0] if len(modes) else ""
            else:
                reference = pd.to_numeric(x_train[col], errors="coerce").median()
        if col in categorical:
            dist += (neighborhood[col].to_numpy() != reference).astype(float)
        else:
            vals = pd.to_numeric(x_train[col], errors="coerce").dropna()
            scale = float(vals.std()) if len(vals) else 1.0
            scale = scale or 1.0
            dist += np.abs(pd.to_numeric(neighborhood[col], errors="coerce").to_numpy() - float(reference)) / scale
    weights = np.exp(-dist / max(np.sqrt(x.shape[1]), 1))
    local_prob = model.predict_proba(neighborhood)[:, 1]
    local_effect = []
    for col in x.columns:
        perturbed = neighborhood.copy()
        reference = row.iloc[0][col]
        if pd.isna(reference):
            if col in categorical:
                modes = x_train[col].mode(dropna=True)
                reference = modes.iloc[0] if len(modes) else ""
            else:
                reference = pd.to_numeric(x_train[col], errors="coerce").median()
        perturbed[col] = reference
        local_effect.append((col, float(np.average(local_prob - model.predict_proba(perturbed)[:, 1], weights=weights))))
    local_df = pd.DataFrame(local_effect, columns=["feature", "local_effect"]).sort_values("local_effect")
    local_df.to_csv(out / "local_explanation.csv", index=False)
    # Optional SHAP and LIME paths make the report comparable with the practical
    # handout while preserving a dependency-light local explanation fallback.
    try:
        import shap
        transformed_row = model.named_steps["preprocess"].transform(row)
        if hasattr(transformed_row, "toarray"):
            transformed_row = transformed_row.toarray()
        shap_explainer = shap.TreeExplainer(forest)
        shap_result = shap_explainer.shap_values(transformed_row)
        if isinstance(shap_result, list):
            shap_vector = np.asarray(shap_result[1])[0]
        else:
            shap_array = np.asarray(shap_result)
            shap_vector = shap_array[0, :, 1] if shap_array.ndim == 3 else shap_array[0]
        shap_df = pd.DataFrame({"feature": names, "shap_value": shap_vector,
                                "mean_absolute_shap": np.abs(shap_vector)})
        shap_df.sort_values("mean_absolute_shap", ascending=False).to_csv(out / "shap_local_explanation.csv", index=False)
        print("SHAP values saved to shap_local_explanation.csv")
    except ImportError:
        print("Optional SHAP package not installed; permutation importance and local perturbation results are included.")
    try:
        from lime.lime_tabular import LimeTabularExplainer
        train_matrix = model.named_steps["preprocess"].transform(x_train)
        test_vector = model.named_steps["preprocess"].transform(row)
        if hasattr(train_matrix, "toarray"):
            train_matrix = train_matrix.toarray()
            test_vector = test_vector.toarray()
        lime_explainer = LimeTabularExplainer(
            training_data=np.asarray(train_matrix), feature_names=names.tolist(),
            class_names=["Higher risk", "Creditworthy"], mode="classification", random_state=SEED)
        lime_result = lime_explainer.explain_instance(
            np.asarray(test_vector)[0], forest.predict_proba, num_features=min(10, len(names)))
        pd.DataFrame(lime_result.as_list(label=1), columns=["feature_rule", "lime_weight"]).to_csv(
            out / "lime_local_explanation.csv", index=False)
        print("LIME values saved to lime_local_explanation.csv")
    except ImportError:
        print("Optional LIME package not installed; local_explanation.csv contains a model-agnostic substitute.")
    print(f"Applicant #0: true label={int(y_test.iloc[0])}, predicted={int(pred[0])}, P(good)={probability[0]:.3f}")
    print("Largest absolute local effects:")
    print(local_df.iloc[local_df.local_effect.abs().sort_values(ascending=False).index].head(8).to_string(index=False))
    print(f"\nSaved figures and explanation tables under {out.resolve()}")


if __name__ == "__main__":
    main()
