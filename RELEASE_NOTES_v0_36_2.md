# Release notes v0.36.2

## Focus

v0.36.2 maakt Project Adviseur minder afhankelijk van handmatige tabelcontrole.
De gebruiker krijgt na een run automatisch een databeheer-oordeel en een
downloadbaar runrapport.

De rekenmotor is niet inhoudelijk verbouwd. Deze release vertaalt bestaande
uitkomsten naar een duidelijker werkproces.

## Waarom

In v0.36.1 werd de werklijst bruikbaarder, maar de gebruiker moest nog steeds
zelf afleiden of een run als geheel bruikbaar was. Dat past niet goed bij het
doel van de tool als automatisering en werkvoorbereiding.

v0.36.2 beantwoordt daarom bovenaan Project Adviseur automatisch:

```text
Kan ik deze analyse gebruiken?
Wat vraagt actie?
Wat is de aanbevolen vervolgstap?
```

## Nieuw

- Automatisch run-oordeel bovenaan Project Adviseur.
- Nieuwe expander `Automatisch runrapport`.
- Nieuwe export:

```text
Projectadvies_Runrapport_<weg>.csv
```

- Het runrapport bevat:
  - run-oordeel;
  - bron/weg;
  - aantal projectvoorstellen;
  - voorstellen zonder directe actie;
  - werklijstregels;
  - verschillen met iASSET;
  - micro-/eindzonevoorstellen;
  - voorstellen buiten ijkbereik;
  - afwijkende hm-intervallen;
  - datakwaliteit;
  - grensmeldingen;
  - eindhandeling: geen automatische iASSET-mutatie.

## Bewust niet gewijzigd

- Geen nieuwe GIS-kniplogica.
- Geen wijziging aan `project_axis.py`.
- Geen automatische iASSET-mutaties.
- Geen Excel-eindrapport; dat blijft een volgende stap.
- De brede voorstellenlijst en werklijst blijven bestaan als onderbouwing.

## Gewijzigde bestanden

- `app.py`
- `README.md`
- `BACKLOG.md`
- `iasset_tool/__init__.py`
- `iasset_tool/config.py`
- `iasset_tool/project_advisor_v2.py`
- `tests/test_project_advisor_v2.py`

## Nieuwe bestanden

- `RELEASE_NOTES_v0_36_2.md`
- `OVERDRACHT_NA_v0_36_2.md`

## Verwijderde bestanden

Geen.

## Validatie

- `pytest -q`: 193 passed
- `python -m py_compile` op alle Python-bestanden

## Testadvies Streamlit

1. Open de app en controleer dat versie `v0.36.2` zichtbaar is.
2. Kies `N398`.
3. Upload `wegassen_paspoort.geojson`.
4. Klik op `🚦 Maak projectadvies`.
5. Controleer niet handmatig losse projectregels, maar download:
   - `Projectadvies_Runrapport_N398.csv`;
   - `Projectadvies_Voorstellen_N398.csv`;
   - `Projectadvies_Werklijst_N398.csv`.
6. Beoordeel de run primair via het runrapport.

Verwachting voor N398:
- de run wordt gepresenteerd als bruikbaar databeheeradvies met werklijst;
- `N398-HRB-01.6-04.6` hoeft niet handmatig te worden opgezocht om het algemene oordeel te bepalen;
- micro-/eindzonegevallen blijven als actiepunt zichtbaar via de werklijst en het runrapport.
