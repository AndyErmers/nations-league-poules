"""
Bouwt de feature-tabel op basis van:
  - matches_raw.csv (Sofascore, groepsfase Nations League)
  - fifa_rankings_history.csv (opgebouwde historische FIFA-ranglijst-snapshots)

Features per wedstrijd (vanuit het perspectief van de thuisploeg):
  - rolling_points_home / rolling_points_away   : vorm laatste N interlands
  - rolling_gd_home / rolling_gd_away           : voortschrijdend doelsaldo
  - fifa_home_rank / fifa_away_rank / fifa_rank_diff
  - fifa_home_points / fifa_away_points / fifa_points_diff
  - rest_days_home / rest_days_away             : dagen sinds vorige interland
  - elo_home / elo_away / elo_diff              : zelf bijgehouden Elo-rating

FEATURE_COLUMNS (wat het model daadwerkelijk gebruikt) bevat bewust niet
alle bovenstaande kolommen — modelvergelijking (zie git-historie/README) wees
uit dat op deze kleine dataset (~600 wedstrijden) een kleine set
verschil-features (fifa_rank_diff, fifa_points_diff, elo_diff) met een sterk
geregulariseerde lineaire classifier beter generaliseert dan absolute
waarden, vormfeatures of complexere modellen (random forest/boosting/SVM):
die overfitten op zoveel features met zo weinig wedstrijden per land per
jaar. De overige kolommen blijven wel in matches_clean.csv staan, voor
eventuele toekomstige experimenten.

Target (alleen voor gespeelde wedstrijden):
  - result: "H" (thuiswinst), "D" (gelijk), "A" (uitwinst)
  - home_goals, away_goals (voor de score-regressie)

Team­namen komen bij matches én rankings allebei van Sofascore, dus die
matchen vrijwel altijd direct; de alias-dict is een vangnet voor incidentele
afwijkingen.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from predictor import config

TEAM_NAME_ALIASES: dict[str, str] = {
    # Vul aan als de pipeline een team niet kan koppelen (zie logging).
}


def _normalize_name(name: str) -> str:
    name = TEAM_NAME_ALIASES.get(name, name)
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _merge_ranking(df: pd.DataFrame, rankings_history: pd.DataFrame, team_col: str, date_col: str, prefix: str) -> pd.DataFrame:
    norm_col = f"__norm_{team_col}"
    df[norm_col] = df[team_col].map(_normalize_name)

    merged_rows = []
    for norm_team, group in df.groupby(norm_col):
        team_hist = rankings_history[rankings_history["norm_team"] == norm_team]
        if team_hist.empty:
            group[f"{prefix}_rank"] = np.nan
            group[f"{prefix}_points"] = np.nan
            merged_rows.append(group)
            continue
        g = group.sort_values(date_col)
        h = team_hist.sort_values("snapshot_date")
        m = pd.merge_asof(
            g, h[["snapshot_date", "rank", "points"]],
            left_on=date_col, right_on="snapshot_date",
            direction="backward",
        )
        if m["rank"].isna().any():
            fallback = h.iloc[0]
            m["rank"] = m["rank"].fillna(fallback["rank"])
            m["points"] = m["points"].fillna(fallback["points"])
        m = m.rename(columns={"rank": f"{prefix}_rank", "points": f"{prefix}_points"})
        merged_rows.append(m.drop(columns=["snapshot_date"]))

    out = pd.concat(merged_rows).sort_index()
    return out.drop(columns=[norm_col])


def _add_rolling_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Bouwt per team een chronologische geschiedenis op en berekent
    voortschrijdend gemiddelde punten/doelsaldo over de laatste N interlands,
    én het aantal rustdagen sinds de vorige interland."""
    played = matches[matches["status_type"] == "finished"].copy()

    long_rows = []
    for _, row in played.iterrows():
        long_rows.append({
            "team": row["home_team"], "date": row["date"], "fixture_id": row["fixture_id"],
            "gf": row["home_goals"], "ga": row["away_goals"],
            "points": 3 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0),
        })
        long_rows.append({
            "team": row["away_team"], "date": row["date"], "fixture_id": row["fixture_id"],
            "gf": row["away_goals"], "ga": row["home_goals"],
            "points": 3 if row["away_goals"] > row["home_goals"] else (1 if row["away_goals"] == row["home_goals"] else 0),
        })
    long_df = pd.DataFrame(long_rows).sort_values(["team", "date"])
    long_df["goal_diff"] = long_df["gf"] - long_df["ga"]
    long_df["rolling_points"] = (
        long_df.groupby("team")["points"].transform(lambda s: s.shift(1).rolling(config.ROLLING_WINDOW, min_periods=1).mean())
    )
    long_df["rolling_goal_diff"] = (
        long_df.groupby("team")["goal_diff"].transform(lambda s: s.shift(1).rolling(config.ROLLING_WINDOW, min_periods=1).mean())
    )
    long_df["prev_match_date"] = long_df.groupby("team")["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_match_date"]).dt.days

    return long_df.set_index(["fixture_id", "team"])[["rolling_points", "rolling_goal_diff", "rest_days"]]


def _add_elo_ratings(matches: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Bouwt per team een Elo-rating op vanaf config.ELO_INITIAL, chronologisch
    bijgewerkt na elke gespeelde wedstrijd (World-Football-Elo-stijl: extra
    K-multiplier naar gelang het doelsaldo, plus een vast thuisvoordeel in de
    verwachte-uitslagberekening). Voor nog niet gespeelde wedstrijden krijgt
    elk team zijn laatst bekende rating (of het startgetal, als het nog nooit
    speelde in de meegegeven data).

    Retourneert een dict fixture_id -> (elo_home, elo_away) met de rating van
    vóór de wedstrijd (voor gespeelde duels) resp. de actuele rating (voor
    aankomende duels) — dus zonder look-ahead naar de eigen uitslag."""
    elo: dict[str, float] = {}
    result: dict[int, tuple[float, float]] = {}

    played = matches[matches["status_type"] == "finished"].sort_values("date")
    for row in played.itertuples():
        home_elo = elo.get(row.home_team, config.ELO_INITIAL)
        away_elo = elo.get(row.away_team, config.ELO_INITIAL)
        result[row.fixture_id] = (home_elo, away_elo)

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + config.ELO_HOME_ADVANTAGE)) / 400.0))
        if row.home_goals > row.away_goals:
            actual_home = 1.0
        elif row.home_goals == row.away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0
        goal_diff = abs(row.home_goals - row.away_goals)
        goal_multiplier = 1.0 if goal_diff <= 1 else (1.5 if goal_diff == 2 else (11 + goal_diff) / 8.0)
        delta = config.ELO_K * goal_multiplier * (actual_home - expected_home)

        elo[row.home_team] = home_elo + delta
        elo[row.away_team] = away_elo - delta

    upcoming = matches[matches["status_type"] != "finished"]
    for row in upcoming.itertuples():
        result[row.fixture_id] = (
            elo.get(row.home_team, config.ELO_INITIAL),
            elo.get(row.away_team, config.ELO_INITIAL),
        )
    return result


def build_feature_table(matches: pd.DataFrame, rankings_history: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    rankings_history = rankings_history.copy()
    rankings_history["norm_team"] = rankings_history["team"].map(_normalize_name)

    df = _merge_ranking(df, rankings_history, "home_team", "date", "fifa_home")
    df = _merge_ranking(df, rankings_history, "away_team", "date", "fifa_away")
    df["fifa_rank_diff"] = df["fifa_away_rank"] - df["fifa_home_rank"]  # positief = thuisploeg beter
    df["fifa_points_diff"] = df["fifa_home_points"] - df["fifa_away_points"]

    form_lookup = _add_rolling_form(matches)
    df["rolling_points_home"] = df.apply(lambda r: form_lookup["rolling_points"].get((r["fixture_id"], r["home_team"]), np.nan), axis=1)
    df["rolling_points_away"] = df.apply(lambda r: form_lookup["rolling_points"].get((r["fixture_id"], r["away_team"]), np.nan), axis=1)
    df["rolling_gd_home"] = df.apply(lambda r: form_lookup["rolling_goal_diff"].get((r["fixture_id"], r["home_team"]), np.nan), axis=1)
    df["rolling_gd_away"] = df.apply(lambda r: form_lookup["rolling_goal_diff"].get((r["fixture_id"], r["away_team"]), np.nan), axis=1)
    df["rest_days_home"] = df.apply(lambda r: form_lookup["rest_days"].get((r["fixture_id"], r["home_team"]), np.nan), axis=1)
    df["rest_days_away"] = df.apply(lambda r: form_lookup["rest_days"].get((r["fixture_id"], r["away_team"]), np.nan), axis=1)

    elo_lookup = _add_elo_ratings(matches)
    df["elo_home"] = df["fixture_id"].map(lambda fid: elo_lookup.get(fid, (config.ELO_INITIAL, config.ELO_INITIAL))[0])
    df["elo_away"] = df["fixture_id"].map(lambda fid: elo_lookup.get(fid, (config.ELO_INITIAL, config.ELO_INITIAL))[1])
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    played_mask = df["status_type"] == "finished"
    df.loc[played_mask, "result"] = np.select(
        [df.loc[played_mask, "home_goals"] > df.loc[played_mask, "away_goals"],
         df.loc[played_mask, "home_goals"] == df.loc[played_mask, "away_goals"]],
        ["H", "D"], default="A",
    )

    df.to_csv(config.MATCHES_CLEAN_CSV, index=False)
    return df


FEATURE_COLUMNS = ["fifa_rank_diff", "fifa_points_diff", "elo_diff"]
