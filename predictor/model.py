"""
Modeltraining en voorspelling.

Twee modellen, getraind op dezelfde features:
  1. Multinomiale classifier -> P(thuiswinst), P(gelijk), P(uitwinst)
  2. Twee Poisson-regressies -> verwacht aantal doelpunten thuis/uit
     (voor een concrete score-suggestie, los van de W/D/L-kans)

Met een relatief kleine dataset (Nations League heeft weinig wedstrijden per
land per jaar) kiezen we bewust voor eenvoudige, sterk geregulariseerde
modellen i.p.v. een zware gradient boosting — dat overfit snel op zo'n kleine
sample.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from predictor import config
from predictor.features import FEATURE_COLUMNS


def _build_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def train_models(training_df: pd.DataFrame) -> dict:
    played = training_df.dropna(subset=["result"]).copy()
    played = played.dropna(subset=FEATURE_COLUMNS, how="all")

    X = played[FEATURE_COLUMNS]
    y_result = played["result"]

    # C=0.05: modelvergelijking (verschillende classifiers, feature-sets en
    # datasetgroottes, zie README) wees dit uit als de beste combinatie op
    # deze kleine dataset — sterkere regularisatie dan sklearns default
    # generaliseert merkbaar beter over meerdere cross-validatiesplits.
    clf = _build_pipeline(LogisticRegression(max_iter=2000, C=0.05))
    clf.fit(X, y_result)

    # Poisson-regressies vereisen volledige (niet-NaN) rijen; imputer in de
    # pipeline lost dat al op.
    poisson_home = _build_pipeline(PoissonRegressor(alpha=1.0, max_iter=500))
    poisson_home.fit(X, played["home_goals"])

    poisson_away = _build_pipeline(PoissonRegressor(alpha=1.0, max_iter=500))
    poisson_away.fit(X, played["away_goals"])

    # Simpele cross-validatie als sanity check, wordt meegenomen in de mail.
    cv_scores = None
    if len(played) >= 15:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y_result, cv=cv, scoring="accuracy")

    models = {
        "classifier": clf,
        "poisson_home": poisson_home,
        "poisson_away": poisson_away,
        "cv_accuracy_mean": float(np.mean(cv_scores)) if cv_scores is not None else None,
        "n_training_matches": len(played),
    }
    joblib.dump(models, config.MODEL_FILE)
    return models


def load_models() -> dict:
    return joblib.load(config.MODEL_FILE)


def predict_matches(models: dict, upcoming_df: pd.DataFrame) -> pd.DataFrame:
    X = upcoming_df[FEATURE_COLUMNS]

    proba = models["classifier"].predict_proba(X)
    classes = list(models["classifier"].named_steps["model"].classes_)

    out = upcoming_df[["date", "home_team", "away_team", "group"]].copy()
    out["prob_home_win"] = proba[:, classes.index("H")]
    out["prob_draw"] = proba[:, classes.index("D")]
    out["prob_away_win"] = proba[:, classes.index("A")]

    out["exp_goals_home"] = models["poisson_home"].predict(X)
    out["exp_goals_away"] = models["poisson_away"].predict(X)
    out["predicted_score"] = (
        out["exp_goals_home"].round().astype(int).astype(str) + "-" +
        out["exp_goals_away"].round().astype(int).astype(str)
    )

    def _pick_outcome(row):
        probs = {"Thuiswinst": row["prob_home_win"], "Gelijkspel": row["prob_draw"], "Uitwinst": row["prob_away_win"]}
        return max(probs, key=probs.get)

    out["meest_waarschijnlijk"] = out.apply(_pick_outcome, axis=1)
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(3)
    return out
