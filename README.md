# nations-league-poules

Voorspelt de uitslagen van de komende UEFA Nations League-wedstrijden, elke
dag opnieuw getraind op de laatste data. **Volledig gratis: geen API-key,
geen account, geen betaalplan.** Alle data komt van Sofascore's publieke
JSON-API.

## Waarom niet api-football

Eerste opzet gebruikte api-football, maar het gratis plan van dat account
bleek alleen toegang te geven tot seizoenen 2022-2024 — niet het lopende
2026-seizoen. Een Pro-abonnement ($19/maand) lost dat op, maar dat is niet
gratis. Sofascore's (ongedocumenteerde) publieke API geeft zowel historische
als actuele Nations League-data én de FIFA-ranglijst, zonder key — dat is nu
de enige databron.

## Wat dit toevoegt t.o.v. de KKD-versie

- **FIFA-ranking als feature**: rankingpositie én -punten van beide landen,
  plus het verschil tussen thuis- en uitploeg.
- **Echte historische ranking i.p.v. één snapshot**: elke pipeline-run voegt
  de actuele FIFA-ranking toe aan `data/fifa_rankings_history.csv`. Features
  gebruiken de ranking zoals die gold vóór de matchdatum (`merge_asof`),
  i.p.v. altijd de nieuwste ranking op oude wedstrijden te plakken.
- **Rustdagen** sinds de vorige interland per land.
- **Geregulariseerde modellen** (multinomiale logistische regressie +
  Poisson-regressie voor de score) i.p.v. een zware boosting-model: met een
  kleine dataset zoals Nations League overfit een complex model snel.
- **Cross-validatie-score** wordt meegenomen in de e-mail zodra er genoeg
  data is.

## Databron-details (belangrijk om te weten)

Sofascore's API is niet-officieel en ongedocumenteerd. Op 22-09-2026 zijn
deze endpoints handmatig geverifieerd (zie `predictor/config.py`):

- UEFA Nations League tournament-id: **10783**
- FIFA-mannenranglijst: `/rankings/type/2` (type 1 is UEFA-clubcoëfficiënten
  per land, dat is géén FIFA-ranking — let op als je dit ooit aanpast)
- Groepsfase = rondes 1 t/m 6. De knock-outfase (halve finales/finale, alleen
  relevant voor League A) zit achter een ander endpoint dat niet via de
  simpele round-index werkt en wordt bewust overgeslagen.

Omdat het ongedocumenteerd is, kan Sofascore de structuur wijzigen. Draai bij
twijfel `python -m predictor.data_fetch` los en controleer de output.

## Setup

```bash
python -m pip install -r requirements.txt
python -m predictor.weekly_pipeline
```

Dat is alles — geen `.env`, geen key, geen aanmelding.

Schrijft:
- `predictor/artifacts/next_predictions.csv`
- `predictor/artifacts/weekly_email.txt`

## Dagelijkse automation om 09:00 via Claude Code scheduled tasks

Ga naar `claude.ai/code/scheduled` (of `/schedule` in de Claude Code CLI) en
maak een taak met deze instructies, cron dagelijks 09:00:

```
Je werkt in de repo AndyErmers/nations-league-poules (branch main).

Doe precies dit:
1. Installeer dependencies indien nodig: `python -m pip install -r requirements.txt`
2. Draai de dagelijkse pipeline: `python -m predictor.weekly_pipeline`
   Dit vernieuwt gespeelde wedstrijden en de FIFA-ranking, herbouwt features,
   en voorspelt de komende 10 Nations League-wedstrijden.
3. Lees `predictor/artifacts/weekly_email.txt` en `predictor/artifacts/next_predictions.csv`.
4. Stuur een e-mail naar andy.ermers@gmail.com met:
   - Onderwerp: Nations League voorspellingen – komende 10 wedstrijden
   - Body: de volledige inhoud van weekly_email.txt
5. Commit en push bijgewerkte data-bestanden terug naar main (matches_raw.csv,
   matches_clean.csv, fifa_rankings.csv, fifa_rankings_history.csv,
   training_data.csv, next_predictions.csv, weekly_email.txt) met een korte
   commit message zoals "Daily Nations League data refresh and predictions".

Als iets faalt: mail alsnog een korte foutmelding naar andy.ermers@gmail.com
met wat er misging.
```

Zet bij het aanmaken van de taak "unrestricted branch pushes" aan voor deze
repo, en zorg dat de Gmail-connector aan die taak gekoppeld is.
