# Overdracht na v0.34.1

## Samenvatting

v0.34.1 stabiliseert de v0.34.0-proef **Projectgrenzen op referentieas**.

De belangrijkste verbetering is ruisreductie: projectdekking, gaten en overlap worden nu per projecttype gecontroleerd. Hierdoor worden HRB, parallelwegen en fietspaden niet meer ten onrechte als onderlinge overlap gemeld.

## Bestanden met belangrijke wijzigingen

### `iasset_tool/project_axis.py`

Aangepast:

- `PROJECT_AXIS_SCHEMA_VERSION = "projectaxis-v0.34.1"`
- aparte projectnaamvalidatie toegevoegd;
- projecttype-splitsing toegevoegd (`HRB`, `PW`, `FP`, `BB`, `LBP` + situering);
- objectligging losgekoppeld van hoofdstatus;
- dekking/gaten/overlap per projecttype;
- begin/eind-gaten richting volledige ijkingsrange verwijderd om ruis bij parallelwegen/fietspaden te voorkomen.

Belangrijke nieuwe helpers:

- `_split_project_type()`
- `_validate_project_name()`
- `_format_name_rule_or_empty()`
- `_worst_status()`

### `app.py`

Aangepast:

- UI-label naar v0.34.1;
- preview van projectgrenzen toont nu de aparte statussen:
  - `naam_validatie_status`
  - `status_projectgrens`
  - `objectligging_status`
  - `status`
- preview van projectdekking toont nu `project_type` en `projectbereik_m`.

### `tests/test_project_axis.py`

Uitgebreid met acceptatietests voor v0.34.1.

## Gedrag in de app

Voor de gebruiker blijft de route hetzelfde:

1. Start app.
2. Kies `🧪 NWB referentieproef`.
3. Upload `wegassen_paspoort.geojson`.
4. Klik op `Haal NWB-wegvakken en hectopunten op`.
5. Controleer:
   - `Projectgrenzen_Referentieas_<weg>.csv`
   - `Projectdekking_Referentieas_<weg>.csv`

## Verwachte verbetering bij N354

De v0.34.0-export gaf veel overlapruis, omdat HRB, PWR, PWL, FPR en FPL op dezelfde referentieas als elkaar werden gezien.

In v0.34.1 moeten deze meldingen verdwijnen:

- HRB overlapt met FPR;
- HRB overlapt met PWR;
- PWL overlapt met FPL.

Deze meldingen moeten juist blijven:

- HRB overlapt met HRB;
- PWR overlapt met PWR;
- PWL overlapt met PWL;
- FPR/FPL overlapt met hetzelfde fietspadtype.

## Interpretatie nieuwe kolommen

### Projectgrenzen

- `status_projectnaam`: gaat alleen over de onderhoudsprojectnaam.
- `status_projectgrens`: gaat over ijking, afwijkingszones en lengteverschil met de geijkte as.
- `objectligging_status`: contextcontrole van fysieke objectprojecties.
- `status`: eindstatus op basis van projectnaam + projectgrens, niet op basis van objectligging.

### Projectdekking

- `project_type`: het spoor waarin dekking/gat/overlap is gecontroleerd.
- `projectbereik_m`: het bereik van de projecten binnen dat spoor.
- `dekking_pct`: unieke dekking binnen het projectbereik van dat spoor.

## Niet gedaan in v0.34.1

Nog niet toegevoegd:

- automatisch projectnamen voorstellen;
- beheergrensvoorstellen;
- afronding naar 5-meter-systeem voor beheerknippen;
- mutaties richting iASSET;
- objecten automatisch opnieuw toewijzen.

## Testresultaat

```text
172 passed
```

Er was een externe spreadsheet-runtime-warmup waarschuwing in de testomgeving. Die staat los van de iASSET-tool en de pytest-suite is volledig geslaagd.
