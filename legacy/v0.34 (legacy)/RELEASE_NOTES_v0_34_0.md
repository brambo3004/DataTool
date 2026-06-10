# Release notes v0.34.0

## Kern

v0.34.0 bouwt de eerste versie van **Projectgrenzen op referentieas**.

De richting is bewust omgedraaid ten opzichte van de oude referentieasproef:
de iASSET-wegas is leidend. NWB-hectopunten worden alleen gebruikt om die
iASSET-as naar hectometrering te ijken. De uitkomst is diagnose en controle;
de tool past niets in iASSET aan.

## Nieuw

- Nieuwe module `iasset_tool/project_axis.py`.
- Nieuwe schema-key `projectaxis-v0.34.0`.
- Nieuwe berekening in `🧪 NWB referentieproef` zodra een iASSET-wegassen-GeoJSON is geüpload.
- Nieuwe exports:
  - `Projectgrenzen_Referentieas_<weg>.csv`
  - `Projectdekking_Referentieas_<weg>.csv`
  - `NWB_Hectopunten_op_iASSET_Wegas_<weg>.csv`
  - `Projectobjecten_Referentieas_<weg>.csv`

## Wat doet v0.34?

1. Projecteert NWB-hectopunten op de iASSET-wegas.
2. Maakt ijkpunten: routepositie op de iASSET-as ↔ hectometrering.
3. Plaatst onderhoudsprojectnamen zoals `N354-HRB-11.5-12.8` op die geijkte as.
4. Waarschuwt als begin- of eindgrenzen in oranje/rode NWB-afwijkingszones liggen.
5. Vergelijkt projectnaamlengte met geijkte-as-lengte.
6. Projecteert objectgeometrie indicatief op de gekozen as en vergelijkt fysieke objectligging met projectnaamrange.
7. Signaleert projectdekking, gaten en overlap langs de geijkte as.

## Veiligheidskaders

- iASSET blijft de single source of truth.
- De NWB-ijking is alleen een hulpmiddel.
- Geen automatische beheerknippen.
- Geen automatische mutaties.
- Corrupte/lege geometrie, ontbrekende hectometrering en lege projectnamen worden gelogd in waarschuwingen in plaats van crashes.

## Tests

Toegevoegd:
- `tests/test_project_axis.py`

Volledige regressie lokaal:
- `168 passed`
