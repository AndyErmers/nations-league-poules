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
- **Elo-rating van eloratings.net** (`predictor/eloratings.py`): per land
  berekend over ál hun interlands (WK/EK-kwalificatie, vriendschappelijk,
  Nations League, eindtoernooien), niet alleen de ~10 Nations
  League-wedstrijden per jaar die in matches_raw.csv staan. Eerst zelf een
  Elo gebouwd (alleen op NL-wedstrijden) — dat werkte, maar eloratings.net's
  bredere basis bleek een veel sterkere feature (zie "Modelkeuze" hieronder).
  In tegenstelling tot de FIFA-ranking (waarvan we vóór vandaag geen echte
  historische snapshots hebben) is Elo nooit "stale" voor oude wedstrijden —
  puur berekend uit de uitslagen zelf.
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
- **Winnende combinatie (ronde 1)**: `LogisticRegression(C=0.05)` op
  `[fifa_rank_diff, fifa_points_diff, elo_diff]` (zelfgebouwde Elo, alleen op
  NL-wedstrijden) — 5-fold cross-validated accuracy **~57,5%** op 610
  wedstrijden (was 54,4% op 472 wedstrijden vóór dit onderzoek).

### Ronde 2 (23-09-2026): eloratings.net, walk-forward-validatie, Dixon-Coles

Verdere vraag: kan het nóg beter? Vier dingen uitgeprobeerd:

1. **Elo baseren op ál iemands interlands i.p.v. alleen Nations
   League** — via [eloratings.net](https://www.eloratings.net) (World
   Football Elo Ratings), dat per land een `.tsv` met de volledige
   wedstrijdhistorie-plus-Elo publiceert. Bleek, in tegenstelling tot
   Sofascore, gewoon bereikbaar vanaf een normale server-IP. Dit leverde
   verreweg de grootste sprong op: **~57,5% → ~62,6%** cross-validated
   accuracy (10-seed gemiddelde, zie hieronder voor de robuustheidscheck).
   Reden: de zelfgebouwde Elo zag maar ~10 wedstrijden per land per jaar en
   was daardoor te ruisgevoelig; eloratings.net's Elo ziet alles.
2. **Walk-forward-validatie** (trainen op alles vóór datum X, testen op wat
   erna komt, 5 opeenvolgende blokken) als eerlijkere schatting dan
   willekeurige k-fold: **~60,3%** gemiddeld — iets lager dan de
   random-CV-schatting (verwacht, met kleine testblokken van 61 wedstrijden
   en veel spreiding per blok: 51%–75%), maar bevestigt dat het model ook
   vooruit in de tijd generaliseert, niet alleen binnen dezelfde periode.
3. **Dixon-Coles-achtig doelpuntenmodel** (thuis/uit-Poisson-scores samen
   gebruiken om H/D/A-kansen wiskundig af te leiden, i.p.v. een aparte
   classifier): **60,2%**, iets lager dan de gewone multinomiale classifier
   (62,8% op dezelfde split). De classifier-aanpak blijft dus staan.
4. **Extra features naast de nieuwe Elo** (vorm, competitietier,
   head-to-head, de oude zelfgebouwde Elo ernaast) zaten over 10 random
   seeds allemaal binnen elkaars ruismarge (std ~0,5-0,7 procentpunt) — dus
   koos de eenvoudigste variant: `[fifa_rank_diff, fifa_points_diff,
   real_elo_home, real_elo_away, real_elo_diff]` met `LogisticRegression(C=0.1)`.

**Huidige winnaar**: 610 wedstrijden, 5 features hierboven, cross-validated
accuracy **~62,6%** (10-seed gemiddelde) / **60,3%** (walk-forward, eerlijker
maar met meer spreiding) — tegenover 33,3% puur gokken en ~43% voor "altijd
de meest voorkomende uitkomst voorspellen".

Voeg je later nieuwe features toe, test dan opnieuw met dezelfde
cross-validatie-aanpak (en idealiter ook walk-forward + meerdere seeds) vóór
je `FEATURE_COLUMNS` aanpast — met deze datasetgrootte is het makkelijk om
een schijnbare verbetering te meten die eigenlijk ruis is.

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

**Sofascore blokkeert cloud/datacenter-IP's** (bevestigd: 403 Forbidden op
alle endpoints, ook de homepage, vanaf zowel deze coding-omgeving als
GitHub-hosted Actions-runners). De dagelijkse workflow draait daarom op een
self-hosted runner (zie hieronder) — een gewone thuis-IP wordt niet
geblokkeerd.

`predictor/eloratings.py` haalt daarnaast per land de wedstrijdhistorie op
van [eloratings.net](https://www.eloratings.net) (`/<Team>.tsv`, teamnamen in
`TEAM_SLUGS`). Dat is, in tegenstelling tot Sofascore, wél gewoon bereikbaar
vanaf een normale server-IP.

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
08:00 CET) via GitHub Actions, op een **self-hosted runner** (een klein
achtergrondproces op een gewone pc/server) — niet op GitHub's standaard
`ubuntu-latest`-runners, want Sofascore blokkeert die IP-reeksen (zie
"Databron-details"). GitHub blijft de trigger/orkestratie en logs doen; de
pipeline zelf draait op een IP dat niet geblokkeerd is. Geen Claude Code
sessie nodig, wel moet de machine met de runner-service aanstaan. De workflow:

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
