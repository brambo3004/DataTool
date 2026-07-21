# Release notes v0.36.1

## Focus

v0.36.1 maakt de Project Adviseur bruikbaarder als dagelijkse werklijst voor
databeheer. De bestaande v0.35.5/v0.36.0-rekenmotor is inhoudelijk niet
verbouwd. Deze release past de presentatielaag aan.

## Waarom

In de eerste N398-test draaide v0.36.0 technisch goed, maar de werklijst bevatte
bijna alle voorstellen. Daardoor voelde de werklijst als een tweede
diagnosetabel, niet als een actielijst.

v0.36.1 maakt daarom onderscheid tussen:

- voorstellen zonder directe actie;
- voorstellen die afwijken van bestaande iASSET-projecten;
- voorstellen bij begin/eind of buiten ijkbereik;
- micro-/eindzonevoorstellen;
- informatieve datakwaliteitssignalen.

## Nieuw

- Nieuwe kolommen in de Project Adviseur-voorstellen:
  - `werkadvies`
  - `werklijst_reden`
  - `voorstelcategorie`
  - `iassetvergelijking`
  - `in_werklijst`
- De werklijst gebruikt nu `werklijst_reden` als filter.
- Lokale datakwaliteit of ondersteunende besteksignalen zetten een voorstel niet
  automatisch in de werklijst als het voorstel verder overeenkomt met iASSET.
- Verschillen met bestaande iASSET-projecten blijven wel werklijstregels.
- Micro-/eindzonevoorstellen worden apart gemarkeerd:
  - `micro-/eindzonevoorstel`
  - eindadvies: `Niet als regulier onderhoudsproject gebruiken`
- De samenvatting is functioneler:
  - voorstellen;
  - geen directe actie;
  - in werklijst;
  - iASSET-verschillen;
  - micro/eindzone;
  - afwijkende hm-intervallen.

## Bewust niet gewijzigd

- Geen nieuwe GIS-kniplogica.
- Geen wijziging aan `project_axis.py`.
- Geen automatische iASSET-mutaties.
- Technische diagnose en brede exports blijven beschikbaar.

## Gewijzigde bestanden

- `app.py`
- `README.md`
- `BACKLOG.md`
- `iasset_tool/__init__.py`
- `iasset_tool/config.py`
- `iasset_tool/project_advisor_v2.py`
- `tests/test_project_advisor_v2.py`

## Nieuwe bestanden

- `RELEASE_NOTES_v0_36_1.md`
- `OVERDRACHT_NA_v0_36_1.md`

## Verwijderde bestanden

Geen.

## Validatie

- `pytest -q`: 191 passed
- `python -m py_compile` op alle Python-bestanden

## Testadvies Streamlit

1. Open de app en controleer dat de versie `v0.36.1` zichtbaar is.
2. Kies `N398`.
3. Upload `wegassen_paspoort.geojson`.
4. Klik op `🚦 Maak projectadvies`.
5. Controleer de samenvatting:
   - `Geen directe actie` is apart zichtbaar;
   - `In werklijst` is lager dan het totaal aantal voorstellen.
6. Controleer dat `N398-HRB-01.6-04.6` zichtbaar blijft als voorstel, maar niet
   in de werklijst komt door lokale datakwaliteit alleen.
7. Controleer dat iASSET-verschillen en micro-/eindzonevoorstellen wel in de
   werklijst staan.
