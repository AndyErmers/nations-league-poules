"""
Hoofdscript, analoog aan `predictor.weekly_pipeline` in de KKD-repo:

  1. Ververs ruwe data (gespeelde Nations League-wedstrijden + FIFA-ranking snapshot)
  2. Herbouw de feature-/trainingstabel
  3. Train de modellen opnieuw
  4. Voorspel de komende N Nations League-wedstrijden
  5. Schrijf next_predictions.csv en weekly_email.txt naar predictor/artifacts/

Wordt aangeroepen als: python -m predictor.weekly_pipeline
"""
from __future__ import annotations

import sys
import pandas as pd

from predictor import config
from predictor.data_fetch import refresh_raw_data, DataFetchError
from predictor.features import build_feature_table
from predictor.model import train_models, predict_matches


def run() -> None:
    matches, _rankings = refresh_raw_data()
    rankings_history = pd.read_csv(config.RANKINGS_HISTORY_CSV, parse_dates=["snapshot_date"])

    feature_table = build_feature_table(matches, rankings_history)
    feature_table.to_csv(config.TRAINING_CSV, index=False)

    models = train_models(feature_table)

    upcoming = feature_table[feature_table["status_type"] == "notstarted"].sort_values("date").head(config.N_UPCOMING_MATCHES)
    if upcoming.empty:
        raise RuntimeError(
            "Geen aankomende Nations League-wedstrijden gevonden in het "
            "geconfigureerde seizoen — controleer API_FOOTBALL_SEASON in .env."
        )

    predictions = predict_matches(models, upcoming)
    predictions.to_csv(config.NEXT_PREDICTIONS_CSV, index=False)

    _write_weekly_email(predictions, models)


def _write_weekly_email(predictions: pd.DataFrame, models: dict) -> None:
    lines = []
    lines.append("Nations League voorspellingen — komende " f"{len(predictions)} wedstrijden")
    lines.append("=" * 60)
    if models.get("cv_accuracy_mean") is not None:
        lines.append(f"Model cross-val nauwkeurigheid (5-fold): {models['cv_accuracy_mean']:.1%}")
    lines.append(f"Getraind op {models['n_training_matches']} gespeelde wedstrijden.")
    lines.append("")

    for _, row in predictions.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%d-%m-%Y")
        lines.append(f"{date_str}  {row['home_team']} - {row['away_team']}  ({row['group']})")
        lines.append(
            f"  Kans: thuis {row['prob_home_win']:.0%} | gelijk {row['prob_draw']:.0%} | uit {row['prob_away_win']:.0%}"
        )
        lines.append(f"  Verwachte score: {row['predicted_score']}  ->  {row['meest_waarschijnlijk']}")
        lines.append("")

    config.WEEKLY_EMAIL_TXT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        run()
        print("Pipeline succesvol afgerond.")
    except DataFetchError as e:
        print(f"FOUT (data ophalen): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"FOUT (pipeline): {e}", file=sys.stderr)
        sys.exit(1)
