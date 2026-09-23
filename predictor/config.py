"""Centrale paden en instellingen voor de Nations League predictor.

Volledig gratis: alle data komt van Sofascore's publieke JSON-API, geen
account of API-key nodig.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "predictor" / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

MATCHES_RAW_CSV = DATA_DIR / "matches_raw.csv"
MATCHES_CLEAN_CSV = DATA_DIR / "matches_clean.csv"
RANKINGS_CSV = DATA_DIR / "fifa_rankings.csv"
RANKINGS_HISTORY_CSV = DATA_DIR / "fifa_rankings_history.csv"
ELORATINGS_HISTORY_CSV = DATA_DIR / "eloratings_history.csv"
TRAINING_CSV = DATA_DIR / "training_data.csv"

NEXT_PREDICTIONS_CSV = ARTIFACTS_DIR / "next_predictions.csv"
WEEKLY_EMAIL_TXT = ARTIFACTS_DIR / "weekly_email.txt"
MODEL_FILE = ARTIFACTS_DIR / "model.joblib"

# --- Sofascore (publiek, geen key nodig) ---
SOFASCORE_BASE = "https://api.sofascore.com/api/v1"
UA_HEADER = {"User-Agent": "Mozilla/5.0 (compatible; nations-league-poules/1.0)"}

# UEFA Nations League op Sofascore (geverifieerd 22-09-2026)
NL_TOURNAMENT_ID = 10783

# type=2 in /rankings/type/{n} is de FIFA-mannenranglijst (geverifieerd:
# Spanje #1, Argentinië #2). type=1 is UEFA-clubcoëfficiënten per land — dus
# NIET de FIFA-ranking, ondanks dat dat de eerste gok leek.
FIFA_RANKING_TYPE = 2

# Aantal seizoenen (incl. het huidige) waarvan de groepsfase wordt gebruikt
# als trainingsdata. De knock-outfase (halve finales/finale, alleen League A)
# zit achter een ander endpoint dat niet via de simpele round-index werkt en
# wordt bewust overgeslagen — een kleine, acceptabele beperking t.o.v. de
# volledige groepsfase-dataset.
N_HISTORICAL_SEASONS = 5
GROUP_STAGE_ROUNDS = range(1, 7)  # rondes 1 t/m 6

N_UPCOMING_MATCHES = 10
ROLLING_WINDOW = 5  # aantal laatste interlands voor vormfeatures
