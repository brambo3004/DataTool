\
# iASSET Advisor - refactor v0.2

Deze versie splitst de bestaande Streamlit proof-of-concept op in een onderhoudbare projectstructuur.

## Doel van deze refactor

Eerst structuur, daarna bugs en nieuwe functionaliteit.

De huidige functionaliteit blijft zoveel mogelijk gelijk:
- iASSET CSV's inlezen;
- WKT-geometrie omzetten naar GeoDataFrame;
- wegnummer selecteren;
- ruimtelijk netwerk bouwen;
- datakwaliteitsmeldingen tonen;
- projectadviesgroepen maken;
- kaart tonen;
- wijzigingen loggen;
- exporteren naar CSV/Excel.

## Nieuwe structuur

```text
.
├── app.py
├── requirements.txt
├── iasset_tool/
│   ├── config.py
│   ├── domain.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── geometry.py
│   ├── rules.py
│   ├── advisor.py
│   ├── pdok.py
│   ├── map_view.py
│   ├── changes.py
│   └── state.py
├── tests/
│   ├── test_utils.py
│   ├── test_data_loader.py
│   ├── test_domain.py
│   ├── test_rules.py
│   └── test_advisor.py
└── legacy/
    └── app_v_11_5_2026.py
```

## Installatie

Plaats de twee iASSET-exportbestanden naast `app.py`:

```text
N-allemaal-niet-rijstrook.csv
N-allemaal-alleen-rijstrook.csv
```

Installeer requirements:

```bash
pip install -r requirements.txt
```

Start de app:

```bash
streamlit run app.py
```


## Tests draaien

Optioneel voor ontwikkelaars:

```bash
pip install -r requirements-dev.txt
pytest
```

## Wijzigingen in v0.2

Deze versie bevat de eerste inhoudelijke bugfix na de structurele refactor:

- uitzonderingenlijst voor onderhoudsprojectplicht gelijkgezet met het werkproces Grijs;
- centrale module `iasset_tool/domain.py` toegevoegd voor domeinpredicaten;
- `NaN`, `None`, lege tekst en de tekst `"nan"` worden gelijk behandeld als lege onderhoudsprojectwaarde;
- objecten met een uitgezonderd subthema krijgen geen melding meer "Mist verplicht onderhoudsproject";
- uitgezonderde objecten die tóch een onderhoudsproject hebben, krijgen nu een waarschuwing;
- objecten met marker `Oorspronkelijke BGT-data` worden uitgezonderd, ook als die marker niet in `subthema` staat;
- Project Adviseur gebruikt dezelfde uitzonderingslogica als Data Kwaliteit;
- tests toegevoegd voor `domain.py`, `rules.py` en `advisor.py`.

Let op: `geleideconstructie` stond in de oude app als uitzondering, maar staat niet in het werkprocesdocument. Daarom is die in v0.2 niet opgenomen als uitzondering. Voeg deze alleen opnieuw toe aan `SUBTHEMA_EXCEPTIONS` als dat inhoudelijk wordt bevestigd.
