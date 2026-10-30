"""Rebuild the CSV/JSON inputs used by dashboard.py from the project splits.

Run after installing requirements.txt:
    python export_dashboard_data.py

The script reproduces the notebook's fixed 5x3 nested CV, final 3-fold
selection, benchmark predictions, and competition scores. It writes only to
dashboard_data/ and does not overwrite the fitted deployment artifacts.
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.decision_tree import build_decision_tree
from models.dummy import build_dummy
from models.extra_trees import build_extra_trees
from models.gaussian_nb import build_gaussian_nb
from models.gradient_boosting import build_gradient_boosting
from models.knn import build_knn
from models.logistic import build_logistic
from models.random_forest import build_random_forest
from models.rbf_svm import build_rbf_svm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "data_splits_colab" / "data_split"
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "dashboard_data"
NUMERIC = ["duration", "amount", "age"]
FEATURE_COUNTS = {"Top_20": 20, "Top_18": 18, "Top_16": 16, "Top_14": 14}
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
EXPECTED = {"full20_nested_cost": 391, "topk_nested_cost": 372,
            "final_selection_cost": 325, "benchmark_cost": 99}

RENAME = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history",
    "verw": "purpose", "hoehe": "amount", "sparkont": "savings",
    "beszeit": "employment_duration", "rate": "installment_rate",
    "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable",
    "telef": "telephone", "gastarb": "foreign_worker",
}


def model_candidates():
    return [
        {"model_name": "Dummy", "model_builder": build_dummy, "configs": [{}]},
        {"model_name": "Logistic Regression", "model_builder": build_logistic,
         "configs": [{"C": c, "class_weight": w}
                     for c, w in product([0.1, 1, 10], [None, "balanced"])]},
        {"model_name": "Gaussian Naive Bayes", "model_builder": build_gaussian_nb,
         "configs": [{"var_smoothing": v} for v in [1e-9, 1e-3]]},
        {"model_name": "KNN", "model_builder": build_knn,
         "configs": [{"n_neighbors": k, "weights": w}
                     for k, w in product([5, 15, 31], ["uniform", "distance"])]},
        {"model_name": "Decision Tree", "model_builder": build_decision_tree,
         "configs": [{"max_depth": d, "min_samples_leaf": leaf}
                     for d, leaf in product([3, 6], [5, 15])]},
        {"model_name": "Random Forest", "model_builder": build_random_forest,
         "configs": [{"max_depth": d, "min_samples_leaf": leaf}
                     for d, leaf in product([5, None], [3, 10])]},
        {"model_name": "Extra Trees", "model_builder": build_extra_trees,
         "configs": [{"max_depth": d, "min_samples_leaf": leaf}
                     for d, leaf in product([5, None], [3, 10])]},
        {"model_name": "RBF SVM", "model_builder": build_rbf_svm,
         "configs": [{"C": c, "gamma": g}
                     for c, g in product([0.1, 1, 10], ["scale", 0.01])]},
        {"model_name": "Gradient Boosting", "model_builder": build_gradient_boosting,
         "configs": [{"learning_rate": lr, "max_depth": d}
                     for lr, d in product([0.05, 0.1], [1, 2])]},
    ]


def build_preprocessor(features, categorical_features):
    nums = [f for f in features if f in NUMERIC]
    cats = [f for f in features if f in categorical_features]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), nums),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats),
    ])


def feature_associations(X_train, y_train):
    rows = []
    for feature in X_train.columns:
        if feature in NUMERIC:
            score = abs(pointbiserialr(y_train, X_train[feature])[0])
        else:
            table = pd.crosstab(X_train[feature], y_train)
            chi2 = chi2_contingency(table)[0]
            denominator = min(table.shape[0] - 1, table.shape[1] - 1)
            score = np.sqrt((chi2 / table.to_numpy().sum()) / denominator) if denominator else 0.0
        rows.append((feature, float(score)))
    return [r[0] for r in sorted(rows, key=lambda r: r[1], reverse=True)]


def find_threshold(y_true, probabilities):
    best = None
    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        row = {"Threshold": float(threshold), "TN": int(tn), "FP": int(fp),
               "FN": int(fn), "TP": int(tp), "Cost": int(5 * fn + fp)}
        if best is None or (row["Cost"], row["Threshold"]) < (best["Cost"], best["Threshold"]):
            best = row
    return best


def make_folds(dev):
    """Mirror main_v2.ipynb: outer and inner StratifiedKFold, seed 42."""
    folds = {}
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(dev, dev["bad_credit"]), 1):
        train = dev.iloc[train_idx].copy()
        val = dev.iloc[val_idx].copy()
        inner_assignment = {}
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for inner_fold, (_, inner_val_idx) in enumerate(inner_cv.split(train, train["bad_credit"]), 1):
            for applicant_id in train.iloc[inner_val_idx]["Id"]:
                inner_assignment[int(applicant_id)] = inner_fold
        folds[outer_fold] = (train, val, inner_assignment)
    return folds


def inner_oof(train, inner_assignment, builder, params, all_features,
              categorical_features, n_features=None):
    oof = pd.Series(index=train.index, dtype=float)
    for inner_fold in (1, 2, 3):
        val_ids = {i for i, fold in inner_assignment.items() if fold == inner_fold}
        val_mask = train["Id"].isin(val_ids)
        tr, va = train.loc[~val_mask], train.loc[val_mask]
        Xtr, ytr = tr[all_features], tr["bad_credit"]
        Xva = va[all_features]
        features = (feature_associations(Xtr, ytr)[:n_features]
                    if n_features is not None else all_features)
        model = builder(build_preprocessor(features, categorical_features), **params)
        model.fit(Xtr[features], ytr)
        oof.loc[va.index] = model.predict_proba(Xva[features])[:, 1]
    return oof.to_numpy()


def search_candidates(train, inner_assignment, candidates, all_features,
                      categorical_features, feature_counts=None):
    rows = []
    strategies = feature_counts or {"Full_20": None}
    for strategy, n_features in strategies.items():
        for candidate in candidates:
            for params in candidate["configs"]:
                probabilities = inner_oof(
                    train, inner_assignment, candidate["model_builder"], params,
                    all_features, categorical_features, n_features,
                )
                best = find_threshold(train["bad_credit"].to_numpy(), probabilities)
                rows.append({
                    "Feature Strategy": strategy,
                    "Number of Features": n_features if n_features is not None else len(all_features),
                    "Model": candidate["model_name"], "Params": params, **best,
                })
    return pd.DataFrame(rows).sort_values(
        ["Cost", "Threshold", "Number of Features"], ignore_index=True
    )


def rank_for_training(train, all_features, y=None, n_features=None):
    selected = feature_associations(train[all_features], train["bad_credit"])
    return selected if n_features is None else selected[:n_features]


def evaluate_outer(fold, train, val, winner, all_features, categorical_features, candidates):
    n_features = int(winner["Number of Features"])
    if winner["Feature Strategy"] == "Full_20":
        features = all_features
    else:
        features = rank_for_training(train, all_features, n_features=n_features)
    candidate = next(c for c in candidates if c["model_name"] == winner["Model"])
    model = candidate["model_builder"](
        build_preprocessor(features, categorical_features), **winner["Params"]
    )
    model.fit(train[features], train["bad_credit"])
    probability = model.predict_proba(val[features])[:, 1]
    threshold = float(winner["Threshold"])
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(val["bad_credit"], prediction, labels=[0, 1]).ravel()
    result = {
        "Outer Fold": fold, "Feature Strategy": winner["Feature Strategy"],
        "Number of Features": n_features, "Selected Features": features,
        "Selected Model": winner["Model"], "Params": winner["Params"],
        "Threshold": threshold, "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "TP": int(tp), "Cost": int(5 * fn + fp),
        "Average Cost": float((5 * fn + fp) / len(val)),
        "Accuracy": float(accuracy_score(val["bad_credit"], prediction)),
        "Precision Bad": float(precision_score(val["bad_credit"], prediction, zero_division=0)),
        "Recall Bad": float(recall_score(val["bad_credit"], prediction)),
        "ROC-AUC": float(roc_auc_score(val["bad_credit"], probability)),
    }
    predictions = pd.DataFrame({
        "Id": val["Id"].to_numpy(), "Outer Fold": fold,
        "Actual": val["bad_credit"].to_numpy(), "P(Bad)": probability,
        "Prediction": prediction,
    })
    return result, predictions


def final_oof(X, y, all_features, categorical_features, candidate, params, n_features=None):
    oof = np.empty(len(X), dtype=float)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in cv.split(X, y):
        Xtr_all, ytr = X.iloc[train_idx][all_features], y.iloc[train_idx]
        Xva_all = X.iloc[val_idx][all_features]
        features = (feature_associations(Xtr_all, ytr)[:n_features]
                    if n_features is not None else all_features)
        model = candidate["model_builder"](
            build_preprocessor(features, categorical_features), **params
        )
        model.fit(Xtr_all[features], ytr)
        oof[val_idx] = model.predict_proba(Xva_all[features])[:, 1]
    return oof


def save_csv(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(SPLIT / "development_640.csv").rename(columns=RENAME)
    ben = pd.read_csv(SPLIT / "historical_benchmark_160.csv").rename(columns=RENAME)
    comp = pd.read_csv(SPLIT / "competition_200.csv").rename(columns=RENAME)
    all_features = [c for c in dev.columns if c not in ("Id", "bad_credit")]
    categorical_features = [c for c in all_features if c not in NUMERIC]
    candidates = model_candidates()
    folds = make_folds(dev)

    # Re-run the two nested-CV procedures that are displayed in the dashboard.
    topk_results, topk_predictions, full_results, full_predictions = [], [], [], []
    for fold, (train, val, inner_assignment) in folds.items():
        print(f"Nested CV export: outer fold {fold}/5")
        full_search = search_candidates(train, inner_assignment, candidates,
                                        all_features, categorical_features)
        full_result, full_pred = evaluate_outer(
            fold, train, val, full_search.iloc[0], all_features,
            categorical_features, candidates,
        )
        full_results.append(full_result)
        full_predictions.append(full_pred)

        topk_search = search_candidates(train, inner_assignment, candidates,
                                        all_features, categorical_features, FEATURE_COUNTS)
        topk_result, topk_pred = evaluate_outer(
            fold, train, val, topk_search.iloc[0], all_features,
            categorical_features, candidates,
        )
        topk_results.append(topk_result)
        topk_predictions.append(topk_pred)

    full_results = pd.DataFrame(full_results)
    topk_results = pd.DataFrame(topk_results)
    full_predictions = pd.concat(full_predictions, ignore_index=True).sort_values("Id")
    topk_predictions = pd.concat(topk_predictions, ignore_index=True).sort_values("Id")
    full_cost = int(full_results["Cost"].sum())
    topk_cost = int(topk_results["Cost"].sum())
    save_csv(full_results, "nested_full20_results")
    save_csv(full_predictions, "nested_full20_predictions")
    save_csv(topk_results, "nested_topk_results")
    save_csv(topk_predictions, "nested_topk_predictions")

    # Reproduce the final 3-fold Development searches shown on the model-selection page.
    X, y = dev[all_features], dev["bad_credit"]
    final_full_rows, final_topk_rows = [], []
    for candidate in candidates:
        for params in candidate["configs"]:
            p = final_oof(X, y, all_features, categorical_features, candidate, params)
            row = {"Feature Strategy": "Full_20", "Number of Features": 20,
                   "Model": candidate["model_name"], "Params": params,
                   **find_threshold(y.to_numpy(), p)}
            final_full_rows.append(row)
            for strategy, k in FEATURE_COUNTS.items():
                p = final_oof(X, y, all_features, categorical_features, candidate, params, k)
                row = {"Feature Strategy": strategy, "Number of Features": k,
                       "Model": candidate["model_name"], "Params": params,
                       **find_threshold(y.to_numpy(), p)}
                final_topk_rows.append(row)
    full_search = pd.DataFrame(final_full_rows).sort_values(
        ["Cost", "Threshold"], ignore_index=True
    )
    topk_search = pd.DataFrame(final_topk_rows).sort_values(
        ["Cost", "Threshold", "Number of Features"], ignore_index=True
    )
    save_csv(full_search, "search_full20_final")
    save_csv(topk_search, "search_topk_final")

    final = topk_search.iloc[0]
    final_strategy = final["Feature Strategy"]
    final_k = int(final["Number of Features"])
    final_candidate = next(c for c in candidates if c["model_name"] == final["Model"])
    final_params = final["Params"]
    final_threshold = float(final["Threshold"])
    features_640 = rank_for_training(dev, all_features, n_features=final_k)

    # This is the frozen rule trained on Development, used for the historical diagnostic benchmark.
    development_model = final_candidate["model_builder"](
        build_preprocessor(features_640, categorical_features), **final_params
    )
    development_model.fit(dev[features_640], dev["bad_credit"])
    benchmark_prob = development_model.predict_proba(ben[features_640])[:, 1]
    benchmark_pred = (benchmark_prob >= final_threshold).astype(int)
    benchmark_export = pd.DataFrame({
        "Id": ben["Id"].to_numpy(), "actual": ben["bad_credit"].to_numpy(),
        "prob_bad": benchmark_prob, "prediction": benchmark_pred,
    })
    save_csv(benchmark_export, "benchmark_predictions")
    benchmark_cost = int(5 * ((ben["bad_credit"].to_numpy() == 1) & (benchmark_pred == 0)).sum()
                        + ((ben["bad_credit"].to_numpy() == 0) & (benchmark_pred == 1)).sum())

    # Use the already-fitted all-800 deployment artifact for unlabeled Competition scores.
    model = joblib.load(ARTIFACTS / "final_model.joblib")
    metadata = joblib.load(ARTIFACTS / "model_metadata.joblib")
    deployment_features = metadata["features"]
    comp_prob = model.predict_proba(comp[deployment_features])[:, 1]
    comp_export = pd.DataFrame({"Id": comp["Id"].to_numpy(), "prob_bad": comp_prob})
    comp_export["prediction_bad"] = (comp_prob >= float(metadata["threshold"])).astype(int)
    save_csv(comp_export, "competition_predictions")

    summary = {
        "n_development": int(len(dev)), "n_benchmark": int(len(ben)),
        "n_competition": int(len(comp)), "full20_nested_cost": full_cost,
        "topk_nested_cost": topk_cost, "final_strategy": str(final_strategy),
        "final_model_name": str(final["Model"]), "final_params": final_params,
        "final_threshold": final_threshold, "final_selection_cost": int(final["Cost"]),
        "features_640": features_640,
        "deployment_features_800": deployment_features,
        "benchmark_cost": benchmark_cost,
        "deployment_threshold": float(metadata["threshold"]),
        "provenance": "Regenerated from fixed project splits, notebook CV design, and saved deployment artifact.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDashboard exports written to:", OUT.relative_to(ROOT))
    print(f"Full-20 nested cost: {full_cost} (expected {EXPECTED['full20_nested_cost']})")
    print(f"Top-k nested cost: {topk_cost} (expected {EXPECTED['topk_nested_cost']})")
    print(f"Final selection: {final_strategy} / {final['Model']} / threshold {final_threshold:.2f}; "
          f"cost {int(final['Cost'])} (expected {EXPECTED['final_selection_cost']})")
    print(f"Benchmark cost: {benchmark_cost} (expected {EXPECTED['benchmark_cost']})")
    if (full_cost, topk_cost, int(final["Cost"]), benchmark_cost) != (
        EXPECTED["full20_nested_cost"], EXPECTED["topk_nested_cost"],
        EXPECTED["final_selection_cost"], EXPECTED["benchmark_cost"],
    ):
        print("WARNING: regenerated results differ from the saved notebook outputs; review versions/data splits.")


if __name__ == "__main__":
    main()
