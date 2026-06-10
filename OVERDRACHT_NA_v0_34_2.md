# Overdracht na v0.34.2

## Samenvatting

v0.34.2 verfijnt de v0.34.1-stabilisatie van **Projectgrenzen op referentieas**.

De belangrijkste verbetering is dat gaten nu worden getoetst aan fysieke primaire objectaanwezigheid. Daardoor verdwijnen gaten die alleen ontstaan omdat een parallelweg of fietspad op een traject simpelweg niet bestaat.

## Bestanden met belangrijke wijzigingen

### `iasset_tool/project_axis.py`

Aangepast:

- `PROJECT_AXIS_SCHEMA_VERSION = "projectaxis-v0.34.2"`;
- `LR` toegevoegd als voorlopig toegestane gecombineerde situering;
- primaire objecten zonder onderhoudsproject worden geprojecteerd voor gatcontrole;
- gatcontrole gebruikt fysieke primaire objecten van hetzelfde spoor;
- gaten zonder fysieke objectaanwezigheid worden niet meer als controlepunt geëxporteerd.

Nieuwe helpers:

- `_subtheme_project_family()`
- `_situering_code()`
- `_object_project_type_from_row()`
- `_project_type_matches_gap()`
- `_primary_object_presence_in_gap()`

### `app.py`

Aangepast:

- UI-labels naar v0.34.2.

### `iasset_tool/config.py`

Aangepast:

- `APP_VERSION = "v0.34.2"`.

### `tests/test_project_axis.py`

Uitgebreid/aangepast:

- test voor gat zonder primaire objecten;
- test voor gat met primair object zonder onderhoudsproject;
- test voor `BBLR`.

## Verwacht effect bij N354

Na opnieuw draaien van N354 zouden de gaten in `Projectdekking_Referentieas_N354.csv` rustiger moeten zijn.

Interpretatie:

- `overlap` binnen hetzelfde projecttype blijft een sterk databeheer-signaal;
- `gat` verschijnt alleen nog als er fysieke primaire objecten in het gat liggen;
- onderbrekingen zonder objectaanwezigheid verdwijnen uit de controlelijst.

`N354-BBLR-15.9-16.0` zou niet langer een naamvalidatie-aandachtspunt moeten zijn.

## Testadvies

Draai opnieuw voor N398 en N354 en controleer vooral:

```text
Projectgrenzen_Referentieas_N398.csv
Projectdekking_Referentieas_N398.csv
Projectobjecten_Referentieas_N398.csv

Projectgrenzen_Referentieas_N354.csv
Projectdekking_Referentieas_N354.csv
Projectobjecten_Referentieas_N354.csv
```

Voor N398 verwacht:

- rustige controlecase;
- geen overlap;
- geen onnodige gatmeldingen.

Voor N354 verwacht:

- `BBLR` naamvalidatie ok;
- minder of geen valse gaten bij parallelwegen/fietspaden;
- echte overlapkandidaten blijven zichtbaar.

## Testresultaat

```text
174 passed
```

Tijdens de teststart verscheen opnieuw een externe spreadsheet-runtime-warmupmelding vanuit de omgeving. De pytest-run is daarna volledig geslaagd.

## Niet gedaan

Nog niet toegevoegd:

- automatische projectnaamvoorstellen;
- automatische beheerknippen;
- afronding naar 5-meter-systeem;
- mutaties of schrijfacties richting iASSET.

