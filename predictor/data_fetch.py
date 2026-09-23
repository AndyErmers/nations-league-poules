"""
Ophalen van ruwe data:

  1. UEFA Nations League: groepsfase-wedstrijden (gespeeld + programma) van
     de laatste N seizoenen — Sofascore's publieke JSON-API.
  2. FIFA-wereldranglijst (mannen) — ook Sofascore.
  3. Elo-ratings per land, gebaseerd op ál hun interlands — eloratings.net.

Beide bronnen zijn niet-officieel en ongedocumenteerd, en volledig gratis
(geen account of key nodig). De gebruikte Sofascore-endpoints zijn op
22-09-2026 handmatig geverifieerd (zie README). Bij fouten loggen we
duidelijk wat er misging zodat de scheduled task dit in de foutmelding-mail
kan meenemen.
"""
from __future__ import annotations

import sys
import time
import requests
import pandas as pd

from predictor import config, eloratings


class DataFetchError(Exception):
    pass


def _get(url: str) -> dict:
    resp = requests.get(url, headers=config.UA_HEADER, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _recent_season_ids() -> list[dict]:
    data = _get(f"{config.SOFASCORE_BASE}/unique-tournament/{config.NL_TOURNAMENT_ID}/seasons")
    seasons = data.get("seasons", [])
    if not seasons:
        raise DataFetchError("Geen Nations League-seizoenen gevonden op Sofascore.")
    # Sofascore levert seasons nieuwste-eerst.
    return seasons[: config.N_HISTORICAL_SEASONS]


def fetch_nations_league_fixtures() -> pd.DataFrame:
    """Haalt groepsfase-wedstrijden (rondes 1-6) op van de laatste
    N_HISTORICAL_SEASONS seizoenen, inclusief het huidige (voor de
    aankomende wedstrijden)."""
    seasons = _recent_season_ids()

    rows = []
    for season in seasons:
        season_id = season["id"]
        season_label = season.get("year", str(season_id))
        for round_no in config.GROUP_STAGE_ROUNDS:
            url = f"{config.SOFASCORE_BASE}/unique-tournament/{config.NL_TOURNAMENT_ID}/season/{season_id}/events/round/{round_no}"
            try:
                data = _get(url)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue  # deze ronde bestaat niet (nog) voor dit seizoen
                raise DataFetchError(f"Sofascore-fout bij {url}: {e}") from e

            for ev in data.get("events", []):
                status_type = ev.get("status", {}).get("type")  # "finished" | "notstarted" | "inprogress" | ...
                rows.append(
                    {
                        "fixture_id": ev["id"],
                        "season": season_label,
                        "round": round_no,
                        "date": pd.to_datetime(ev["startTimestamp"], unit="s", utc=True),
                        "status_type": status_type,
                        "home_team": ev["homeTeam"]["name"],
                        "away_team": ev["awayTeam"]["name"],
                        "home_goals": ev.get("homeScore", {}).get("current"),
                        "away_goals": ev.get("awayScore", {}).get("current"),
                        "group": ev.get("tournament", {}).get("name"),  # bv. "... League C, Gr. 1"
                    }
                )
            time.sleep(0.2)  # vriendelijk zijn voor het (ongedocumenteerde) endpoint

    df = pd.DataFrame(rows)
    if df.empty:
        raise DataFetchError(
            "Geen Nations League-wedstrijden opgehaald van Sofascore — "
            "controleer NL_TOURNAMENT_ID in config.py (kan gewijzigd zijn)."
        )
    return df.sort_values("date").reset_index(drop=True)


def fetch_fifa_rankings() -> pd.DataFrame:
    """Haalt de actuele FIFA-mannenranglijst op."""
    data = _get(f"{config.SOFASCORE_BASE}/rankings/type/{config.FIFA_RANKING_TYPE}")

    rows = []
    for entry in data.get("rankings", []):
        team = entry.get("team", {})
        rows.append(
            {
                "team": team.get("name"),
                "country_code": team.get("country", {}).get("alpha3"),
                "rank": entry.get("ranking"),
                "points": entry.get("points"),
                "previous_rank": entry.get("previousRanking"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise DataFetchError(
            "Geen FIFA-ranglijst data van Sofascore — endpoint is mogelijk "
            "gewijzigd (config.FIFA_RANKING_TYPE)."
        )
    df["fetched_at"] = pd.Timestamp.utcnow()
    return df


def refresh_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Haalt alle bronnen op en schrijft ze weg als ruwe CSV's."""
    matches = fetch_nations_league_fixtures()
    matches.to_csv(config.MATCHES_RAW_CSV, index=False)

    rankings = fetch_fifa_rankings()
    rankings.to_csv(config.RANKINGS_CSV, index=False)

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    try:
        elo_history = eloratings.fetch_history(teams)
    except (eloratings.EloFetchError, requests.RequestException) as e:
        raise DataFetchError(f"Fout bij ophalen Elo-historie van eloratings.net: {e}") from e
    elo_history.to_csv(config.ELORATINGS_HISTORY_CSV, index=False)

    # Bouw een historische reeks op door elke run toe te voegen i.p.v. te
    # overschrijven, zodat features de ranking-op-matchdatum kunnen gebruiken
    # i.p.v. altijd de nieuwste ranking op oude wedstrijden te plakken.
    snapshot_date = pd.Timestamp.utcnow().normalize()
    rankings_snapshot = rankings.copy()
    rankings_snapshot["snapshot_date"] = snapshot_date

    if config.RANKINGS_HISTORY_CSV.exists():
        history = pd.read_csv(config.RANKINGS_HISTORY_CSV, parse_dates=["snapshot_date"])
        already_today = (history["snapshot_date"] == snapshot_date).any()
        if not already_today:
            history = pd.concat([history, rankings_snapshot], ignore_index=True)
    else:
        history = rankings_snapshot

    history.to_csv(config.RANKINGS_HISTORY_CSV, index=False)

    return matches, rankings


if __name__ == "__main__":
    try:
        m, r = refresh_raw_data()
        print(f"OK: {len(m)} wedstrijden, {len(r)} landen in ranglijst opgehaald.")
    except DataFetchError as e:
        print(f"FOUT bij ophalen data: {e}", file=sys.stderr)
        sys.exit(1)
