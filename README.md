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

## Dagelijkse automation om 09:00 via GitHub Actions

`.github/workflows/daily.yml` draait dagelijks om 07:00 UTC (09:00 CEST /
08:00 CET), volledig op GitHub's eigen infrastructuur — geen lokale pc of
Claude Code sessie nodig. De workflow:

1. Installeert dependencies en draait `python -m predictor.weekly_pipeline`.
2. Committet en pusht de bijgewerkte data-bestanden terug naar `main` (met
   het automatische `GITHUB_TOKEN`, geen handmatige credential nodig).
3. Mailt `predictor/artifacts/weekly_email.txt` naar andy.ermers@gmail.com.
4. Mailt bij een mislukte run een korte foutmelding met een link naar de logs.

**Eenmalige setup (via GitHub, Settings → Secrets and variables → Actions):**
maak twee *repository secrets* aan zodat de workflow via Gmail SMTP kan
mailen:

- `GMAIL_ADDRESS`: andy.ermers@gmail.com
- `GMAIL_APP_PASSWORD`: een Gmail
  [app-wachtwoord](https://myaccount.google.com/apppasswords) voor dat
  account (vereist 2FA op het account).

Test daarna de workflow eenmalig handmatig via het "Run workflow"-knopje op
het Actions-tabblad (`workflow_dispatch`), om te controleren of Sofascore
GitHub's runner-IP's niet blokkeert en of de mail aankomt.
