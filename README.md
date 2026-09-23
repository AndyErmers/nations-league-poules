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
- **Eigen Elo-rating** (`predictor.features._add_elo_ratings`): per team
  chronologisch bijgehouden vanaf 1500, bijgewerkt na elke gespeelde
  wedstrijd (World-Football-Elo-stijl: K-multiplier naar doelsaldo, vast
  thuisvoordeel). In tegenstelling tot de FIFA-ranking (waarvan we vóór
  vandaag geen echte historische snapshots hebben) is Elo nooit "stale" voor
  oude wedstrijden — puur berekend uit de uitslagen zelf.
- **Geregulariseerde modellen** (multinomiale logistische regressie +
  Poisson-regressie voor de score) i.p.v. een zware boosting-model: met een
  kleine dataset zoals Nations League overfit een complex model snel.
- **Cross-validatie-score** wordt meegenomen in de e-mail zodra er genoeg
  data is.

## Modelkeuze: waarom deze features en dit model

Op 23-09-2026 is systematisch uitgeprobeerd wat de cross-validated accuracy
(5-fold + 10-fold gemiddeld, `StratifiedKFold`) het meest verbetert:
verschillende classifiers (logistic regression, random forest, extra trees,
gradient boosting, HistGradientBoosting, XGBoost, LightGBM, SVM, KNN,
Gaussian NB, MLP, voting/stacking-ensembles), features (absolute
FIFA-rank/-punten vs. alleen het verschil, vormfeatures met en zonder
doelpunten los van doelsaldo, competitietier, head-to-head-geschiedenis,
Elo) en datasetgrootte (4 vs. 5 seizoenen historie).

Uitkomst:

- **Meer trainingsdata helpt merkbaar**: `N_HISTORICAL_SEASONS` ging van 4
  naar 5 (seizoen 18/19 erbij, 472 → 610 gespeelde wedstrijden), goed voor
  ~+2 procentpunt.
- **Een kleine set verschil-features wint van absolute waarden of een
  uitgebreide feature-set**: `fifa_rank_diff`, `fifa_points_diff` en
  `elo_diff` (3 kolommen) presteren beter dan varianten met ook de absolute
  FIFA-rank/Elo per land, vormfeatures, competitietier of
  head-to-head-historie erbij — die extra kolommen voegden ruis toe in
  plaats van signaal op deze kleine dataset.
- **Complexere modellen verliezen het van sterk geregulariseerde lineaire
  modellen**: random forest, gradient boosting (incl. XGBoost/LightGBM), SVM
  met RBF-kernel, KNN en neurale netjes scoorden allemaal lager dan
  logistische regressie met een kleine `C` (sterke regularisatie). Met ~600
  wedstrijden en maar een paar interlands per land per jaar overfitten de
  complexere modellen sneller dan ze leren.
- **Winnende combinatie**: `LogisticRegression(C=0.05)` op
  `[fifa_rank_diff, fifa_points_diff, elo_diff]` — 5-fold cross-validated
  accuracy **~57,5%** op 610 wedstrijden (was 54,4% op 472 wedstrijden vóór
  dit onderzoek), tegenover 33,3% puur gokken en ~41-43% als baseline voor
  "voorspel altijd de meest voorkomende uitkomst" (thuiswinst).

Voeg je later nieuwe features toe (bijv. head-to-head of transfermarkt-
waarde), test dan opnieuw met dezelfde cross-validatie-aanpak vóór je
`FEATURE_COLUMNS` aanpast — met deze datasetgrootte is het makkelijk om een
schijnbare verbetering te meten die eigenlijk ruis is.

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
