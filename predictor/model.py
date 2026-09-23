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

De twee modellen zijn onafhankelijk getraind, dus zonder correctie kán de
"meest waarschijnlijke uitslag" van de classifier een andere kant op wijzen
dan de score die je krijgt door de twee Poisson-verwachtingen los af te
ronden (bv. classifier zegt thuiswinst, afgeronde Poisson-scores geven 1-1).
`predict_matches` kiest daarom de meest waarschijnlijke score ÔNDER DE
VOORWAARDE dat die bij de classifier-uitslag past (zie `_best_score_for_outcome`),
zodat score en uitslag altijd met elkaar overeenkomen.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from predictor import config
from predictor.features import FEATURE_COLUMNS

MAX_GOALS = 6  # voldoende marge; internationale wedstrijden met >6 doelpunten zijn zeldzaam
OUTCOME_LABELS = {"H": "Thuiswinst", "D": "Gelijkspel", "A": "Uitwinst"}


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

    # C=0.1: modelvergelijking (verschillende classifiers, feature-sets en
    # datasetgroottes, zie README) wees dit uit als de beste combinatie op
    # deze kleine dataset — sterkere regularisatie dan sklearns default
    # generaliseert merkbaar beter over meerdere cross-validatiesplits.
    clf = _build_pipeline(LogisticRegression(max_iter=2000, C=0.1))
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


def _best_score_for_outcome(lam_home: float, lam_away: float, outcome: str) -> tuple[int, int]:
    """Kiest, gegeven de verwachte doelpuntenaantallen (Poisson-gemiddelden),
    de score (i, j) met de hoogste gezamenlijke Poisson-kans die nog steeds
    bij `outcome` (H/D/A) past. Zo wijst de weergegeven score nooit een
    andere uitslag aan dan de kans die de classifier het hoogst inschat."""
    best_score = (1, 1)
    best_p = -1.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            if outcome == "H" and i <= j:
                continue
            if outcome == "D" and i != j:
                continue
            if outcome == "A" and i >= j:
                continue
            p = poisson_dist.pmf(i, lam_home) * poisson_dist.pmf(j, lam_away)
            if p > best_p:
                best_p, best_score = p, (i, j)
    return best_score


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

    outcome_codes = out[["prob_home_win", "prob_draw", "prob_away_win"]].idxmax(axis=1).map(
        {"prob_home_win": "H", "prob_draw": "D", "prob_away_win": "A"}
    )
    scores = [
        _best_score_for_outcome(lam_h, lam_a, outcome)
        for lam_h, lam_a, outcome in zip(out["exp_goals_home"], out["exp_goals_away"], outcome_codes)
    ]
    out["predicted_score"] = [f"{h}-{a}" for h, a in scores]
    out["meest_waarschijnlijk"] = outcome_codes.map(OUTCOME_LABELS)

    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(3)
    return out
