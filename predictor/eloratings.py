"""Haalt Elo-ratings op van eloratings.net (World Football Elo Ratings) —
gebaseerd op ÁLLE interlands van een land (WK/EK-kwalificatie, vriendschappelijk,
Nations League, eindtoernooien), niet alleen de Nations League-wedstrijden in
matches_raw.csv. Dat maakt het een veel stabielere ratingbron dan een Elo die
we zelf zouden opbouwen uit alleen de ~10 Nations League-duels per land per
jaar: modelvergelijking (zie README) toonde een sprong van 57,5% naar ~62,6%
cross-validated accuracy door deze bron te gebruiken i.p.v. een zelf berekende
Elo op alleen NL-wedstrijden.

Niet-officieel, ongedocumenteerd net als Sofascore — maar wél gewoon bereikbaar
vanaf een normale server-IP (in tegenstelling tot Sofascore, dat cloud/
datacenter-IP's blokkeert).
"""
from __future__ import annotations

import sys
import pandas as pd
import requests

UA_HEADER = {"User-Agent": "Mozilla/5.0 (compatible; nations-league-poules/1.0)"}
BASE = "https://www.eloratings.net"

# Ruim voor onze vroegste Nations League-wedstrijd (2018) — we hoeven niet
# verder terug, en heel oude rijen bevatten soms een onvolledige datum
# (bv. "1956-11-00", dag onbekend) die pd.to_datetime laat struikelen.
HISTORY_START = "2015-01-01"

# Sofascore-teamnaam -> eloratings.net URL-slug. Vul aan als de pipeline een
# team niet kan vinden (zie de foutmelding die fetch_history() dan geeft).
TEAM_SLUGS: dict[str, str] = {
    "Albania": "Albania", "Andorra": "Andorra", "Armenia": "Armenia", "Austria": "Austria",
    "Azerbaijan": "Azerbaijan", "Belarus": "Belarus", "Belgium": "Belgium",
    "Bosnia & Herzegovina": "Bosnia_and_Herzegovina", "Bulgaria": "Bulgaria", "Croatia": "Croatia",
    "Cyprus": "Cyprus", "Czechia": "Czechia", "Denmark": "Denmark", "England": "England",
    "Estonia": "Estonia", "Faroe Islands": "Faroe_Islands", "Finland": "Finland", "France": "France",
    "Georgia": "Georgia", "Germany": "Germany", "Gibraltar": "Gibraltar", "Greece": "Greece",
    "Hungary": "Hungary", "Iceland": "Iceland", "Ireland": "Ireland", "Israel": "Israel",
    "Italy": "Italy", "Kazakhstan": "Kazakhstan", "Kosovo": "Kosovo", "Latvia": "Latvia",
    "Liechtenstein": "Liechtenstein", "Lithuania": "Lithuania", "Luxembourg": "Luxembourg",
    "Malta": "Malta", "Moldova": "Moldova", "Montenegro": "Montenegro", "Netherlands": "Netherlands",
    "North Macedonia": "North_Macedonia", "Northern Ireland": "Northern_Ireland", "Norway": "Norway",
    "Poland": "Poland", "Portugal": "Portugal", "Romania": "Romania", "Russia": "Russia",
    "San Marino": "San_Marino", "Scotland": "Scotland", "Serbia": "Serbia", "Slovakia": "Slovakia",
    "Slovenia": "Slovenia", "Spain": "Spain", "Sweden": "Sweden", "Switzerland": "Switzerland",
    "Türkiye": "Turkey", "Ukraine": "Ukraine", "Wales": "Wales",
}

# Elk team kan van landcode zijn veranderd (bv. Noord-Macedonië: MK -> NM in
# 2019) — beide accepteren voorkomt valse "misses".
TEAM_CODES: dict[str, list[str]] = {
    "Albania": ["AL"], "Andorra": ["AD"], "Armenia": ["AM"], "Austria": ["AT"], "Azerbaijan": ["AZ"],
    "Belarus": ["BY"], "Belgium": ["BE"], "Bosnia & Herzegovina": ["BA"], "Bulgaria": ["BG"],
    "Croatia": ["HR"], "Cyprus": ["CY"], "Czechia": ["CZ"], "Denmark": ["DK"], "England": ["EN"],
    "Estonia": ["EE"], "Faroe Islands": ["FO"], "Finland": ["FI"], "France": ["FR"], "Georgia": ["GE"],
    "Germany": ["DE"], "Gibraltar": ["GI"], "Greece": ["GR"], "Hungary": ["HU"], "Iceland": ["IS"],
    "Ireland": ["IE"], "Israel": ["IL"], "Italy": ["IT"], "Kazakhstan": ["KZ"], "Kosovo": ["KO"],
    "Latvia": ["LV"], "Liechtenstein": ["LI"], "Lithuania": ["LT"], "Luxembourg": ["LU"], "Malta": ["MT"],
    "Moldova": ["MD"], "Montenegro": ["ME"], "Netherlands": ["NL"], "North Macedonia": ["NM", "MK"],
    "Northern Ireland": ["EI"], "Norway": ["NO"], "Poland": ["PL"], "Portugal": ["PT"], "Romania": ["RO"],
    "Russia": ["RU"], "San Marino": ["SM"], "Scotland": ["SQ"], "Serbia": ["RS"], "Slovakia": ["SK"],
    "Slovenia": ["SI"], "Spain": ["ES"], "Sweden": ["SE"], "Switzerland": ["CH"], "Türkiye": ["TR"],
    "Ukraine": ["UA"], "Wales": ["WA"],
}


class EloFetchError(Exception):
    pass


def fetch_history(teams: list[str]) -> pd.DataFrame:
    """Haalt voor elk opgegeven team (Sofascore-naam) de wedstrijdhistorie
    sinds HISTORY_START op van eloratings.net en zet 'm om naar een lange
    tabel (team, date, elo) met de rating van dát team ná elke wedstrijd.
    Wedstrijden op of na vandaag worden overgeslagen — eloratings.net toont
    voor lopende toernooien gesimuleerde toekomstige uitslagen, die we niet
    als trainingsdata willen."""
    today = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    unknown = [t for t in teams if t not in TEAM_SLUGS]
    if unknown:
        raise EloFetchError(
            f"Onbekende team(s) voor eloratings.net, voeg toe aan TEAM_SLUGS/TEAM_CODES: {unknown}"
        )

    rows = []
    for team in teams:
        slug = TEAM_SLUGS[team]
        codes = TEAM_CODES[team]
        resp = requests.get(f"{BASE}/{slug}.tsv", headers=UA_HEADER, timeout=30)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            c = line.split("\t")
            date_str = f"{c[0]}-{c[1]}-{c[2]}"
            if date_str < HISTORY_START or date_str >= today:
                continue
            home_code, away_code = c[3], c[4]
            if home_code in codes:
                elo = c[10]
            elif away_code in codes:
                elo = c[11]
            else:
                continue  # ander team met toevallig dezelfde datum-lijn; niet mogelijk binnen 1 bestand, maar defensief
            rows.append({"team": team, "date": date_str, "elo": float(elo)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise EloFetchError("Geen Elo-historie opgehaald van eloratings.net.")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values(["team", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    try:
        df = fetch_history(list(TEAM_SLUGS))
        print(f"OK: {len(df)} rijen Elo-historie voor {df['team'].nunique()} landen.")
    except EloFetchError as e:
        print(f"FOUT bij ophalen Elo-historie: {e}", file=sys.stderr)
        sys.exit(1)
