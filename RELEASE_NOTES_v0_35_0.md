# Release notes v0.35.0

## Doel

v0.35.0 voegt een eerste proef toe voor **Onderhoudsprojectvoorstellen vanaf nul**.

De uitgangspositie is nadrukkelijk gewijzigd ten opzichte van een simpele controle op bestaande onderhoudsprojecten:

- bestaande iASSET-onderhoudsprojecten zijn niet de basis;
- primaire iASSET-objecten en hun fysieke ligging op de geijkte referentieas zijn leidend;
- bestaande projectnamen worden alleen achteraf gebruikt om te vergelijken.

## Nieuw

### Groenveld-projectvoorstellen

Nieuwe tabel/export:

```text
Projectvoorstellen_Referentieas_<weg>.csv
```

Per voorstel toont de tool onder andere:

- `voorstel_id`
- `project_type`
- `fysiek_begin_m`
- `fysiek_eind_m`
- `fysiek_lengte_m`
- `naam_begin`
- `naam_eind`
- `onderhoudsproject_voorgesteld`
- `knipreden_begin`
- `knipreden_eind`
- `aantal_primaire_objecten`
- `bestaande_onderhoudsprojecten`
- `vergelijking_iasset_status`
- `status_voorstel`
- `hoofdmelding`
- `contextmelding`

### Objecttoewijzing bij voorstellen

Nieuwe tabel/export:

```text
Projectvoorstel_Objecten_<weg>.csv
```

Deze laat per primair object zien bij welk voorstel het object volgens de groenveldlogica hoort.

### Vergelijking met iASSET

Nieuwe tabel/export:

```text
Projectvoorstel_Vergelijking_iASSET_<weg>.csv
```

Deze vergelijkt achteraf:

- bestaande onderhoudsprojecten versus voorstellen;
- voorstellen versus bestaande onderhoudsprojecten.

Daarmee kunnen controles ontstaan zoals:

- bestaand project komt overeen;
- bestaande naam wijkt af van groenveldvoorstel;
- bestaand project splitst over meerdere voorstellen;
- voorstel bundelt meerdere bestaande projecten;
- nieuw voorstel zonder bestaande iASSET-naam.

## Kniplogica v0.35.0

De eerste versie knipt per spoor, bijvoorbeeld HRB, PWR, PWL, FPR, FPL en BBLR.

Een nieuw voorstel begint bij:

- een fysiek gat groter dan de ingestelde gat-tolerantie;
- een wijziging in één van de beheerkenmerken:
  - `Besteknummer`
  - `verhardingssoort`
  - `Soort verharding_N`
  - `Soort deklaag specifiek`
  - `Jaar aanleg`
  - `Jaar deklaag`
  - `Jaar conservering`
  - `Jaar herstrating`

## Naamregel

De voorgestelde onderhoudsprojectnaam gebruikt dezelfde regel als v0.34.3/v0.34.5:

- standaard snap-tolerantie naar hectometerpunt: 2,5 meter;
- daarna hectometer-naar-boven-regel;
- schrijfwijze met één cijfer na de punt, inclusief voorloopnul onder 10.

## Niet gewijzigd

De bestaande v0.34.5-diagnose blijft aanwezig:

- projectgrenzen;
- projectdekking/gaten/overlap;
- compacte controlelijst;
- objectprojecties.

## Niet toegevoegd

v0.35.0 doet nog niet:

- iASSET automatisch wijzigen;
- beheerknippen aanmaken;
- objecten definitief herverdelen;
- projecten automatisch samenvoegen of splitsen;
- een iASSET-importbestand maken.

## Validatie

```text
182 passed
python -m py_compile app.py iasset_tool/project_axis.py
```
