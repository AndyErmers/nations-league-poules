"""
Bouwt de feature-tabel op basis van:
  - matches_raw.csv (Sofascore, groepsfase Nations League)
  - fifa_rankings_history.csv (opgebouwde historische FIFA-ranglijst-snapshots)
  - eloratings_history.csv (eloratings.net — Elo per land over ál hun
    interlands, niet alleen Nations League; zie predictor/eloratings.py)

Features per wedstrijd (vanuit het perspectief van de thuisploeg):
  - rolling_points_home / rolling_points_away   : vorm laatste N interlands
  - rolling_gd_home / rolling_gd_away           : voortschrijdend doelsaldo
  - fifa_home_rank / fifa_away_rank / fifa_rank_diff
  - fifa_home_points / fifa_away_points / fifa_points_diff
  - rest_days_home / rest_days_away             : dagen sinds vorige interland
  - real_elo_home / real_elo_away / real_elo_diff : Elo-rating (eloratings.net)

FEATURE_COLUMNS (wat het model daadwerkelijk gebruikt) bevat bewust niet
alle bovenstaande kolommen. Modelvergelijking (zie README) doorliep twee
ronden:
  1. Een zelf berekende Elo (alleen op Nations League-wedstrijden) met een
     kleine verschil-feature-set haalde ~57,5% cross-validated accuracy,
     tegenover 54,4% met alleen FIFA-ranking.
  2. Vervangen van die zelfgebouwde Elo door eloratings.net's Elo — gebaseerd
     op ál iemands interlands, dus veel minder ruisgevoelig dan een rating
     die maar ~10 wedstrijden per land per jaar ziet — bracht dat naar
     ~62,6%. Extra features (vorm, competitietier, head-to-head, de eigen
     Elo ernaast) bleken binnen ruis van elkaar te liggen; de eenvoudigste
     variant wint dus volgens Occam's scheermes. De overige kolommen blijven
     wel in matches_clean.csv staan, voor eventuele toekomstige experimenten.

Target (alleen voor gespeelde wedstrijden):
  - result: "H" (thuiswinst), "D" (gelijk), "A" (uitwinst)
  - home_goals, away_goals (voor de score-regressie)

Team­namen komen bij matches én rankings allebei van Sofascore, dus die
matchen vrijwel altijd direct; de alias-dict is een vangnet voor incidentele
afwijkingen. eloratings.net gebruikt een eigen naamgeving — zie
predictor/eloratings.TEAM_SLUGS.
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


def _merge_eloratings(df: pd.DataFrame, eloratings_history: pd.DataFrame, team_col: str, date_col: str, prefix: str) -> pd.DataFrame:
    """Koppelt aan elke wedstrijd de Elo-rating van het team zoals die gold
    vlak vóór de matchdatum (`merge_asof`, net als bij de FIFA-ranking)."""
    merged_rows = []
    for team, group in df.groupby(team_col):
        team_hist = eloratings_history[eloratings_history["team"] == team]
        if team_hist.empty:
            group = group.copy()
            group[prefix] = np.nan
            merged_rows.append(group)
            continue
        g = group.sort_values(date_col)
        h = team_hist.sort_values("date").rename(columns={"date": "elo_date"})
        m = pd.merge_asof(g, h[["elo_date", "elo"]], left_on=date_col, right_on="elo_date", direction="backward")
        if m["elo"].isna().any():
            m["elo"] = m["elo"].fillna(h.iloc[0]["elo"])
        m = m.rename(columns={"elo": prefix}).drop(columns=["elo_date"])
        merged_rows.append(m)

    return pd.concat(merged_rows).sort_index()


def build_feature_table(matches: pd.DataFrame, rankings_history: pd.DataFrame, eloratings_history: pd.DataFrame) -> pd.DataFrame:
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

    df = _merge_eloratings(df, eloratings_history, "home_team", "date", "real_elo_home")
    df = _merge_eloratings(df, eloratings_history, "away_team", "date", "real_elo_away")
    df["real_elo_diff"] = df["real_elo_home"] - df["real_elo_away"]

    played_mask = df["status_type"] == "finished"
    df.loc[played_mask, "result"] = np.select(
        [df.loc[played_mask, "home_goals"] > df.loc[played_mask, "away_goals"],
         df.loc[played_mask, "home_goals"] == df.loc[played_mask, "away_goals"]],
        ["H", "D"], default="A",
    )

    df.to_csv(config.MATCHES_CLEAN_CSV, index=False)
    return df


FEATURE_COLUMNS = ["fifa_rank_diff", "fifa_points_diff", "real_elo_home", "real_elo_away", "real_elo_diff"]
