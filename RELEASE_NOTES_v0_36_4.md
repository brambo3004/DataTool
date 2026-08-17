# Release notes v0.36.4

## Doel

v0.36.4 corrigeert de concept-export in het format van het N-wegendocument.
De export gebruikt nu knip- en verhardingswaarden die vergelijkbaar zijn met de
handmatige N-wegendocumenttabbladen.

In v0.36.3 werden de kolommen `knip (begin)`, `knip (einde)`,
`verharding (begin)` en `verharding (einde)` gevuld met relatieve route-meters
op de projectas. Dat was technisch logisch voor de GIS-engine, maar niet bruikbaar
om naast het N-wegendocument te leggen.

## Nieuw / gewijzigd

- De N-wegendocument-export gebruikt nu primair:
  - `fysiek_begin_km`
  - `fysiek_eind_km`
- Deze waarden worden omgerekend naar hectometreringsmeters:
  - `25.800 km` → `25800`
  - `26.300 km` → `26300`
- Relatieve projectas-meters worden alleen nog gebruikt als fallback wanneer
  km-waarden ontbreken.
- De kolommen die hiermee zijn gecorrigeerd:
  - `knip (begin)`
  - `knip (einde)`
  - `verharding (begin)`
  - `verharding (einde)`

## Ongewijzigd

- Geen wijziging in de projectas-engine.
- Geen wijziging in de werklijstlogica.
- Geen wijziging in het runrapport.
- Geen automatische iASSET-mutatie.
- Geen overschrijving van het bestaande N-wegendocument.

## Waarom dit belangrijk is

Het concepttabblad moet naast het handmatige N-wegendocument gelegd kunnen worden.
Daarvoor moeten de knipwaarden dezelfde betekenis hebben. Een waarde zoals `524`
betekent in de engine een relatieve afstand op de gekozen wegas, maar in het
N-wegendocument verwacht je bij een traject `25.8–26.3` waarden rond `25800–26300`.

## Validatie

- `pytest -q`: 196 passed
- `python -m py_compile` op alle Python-bestanden
- Nieuwe regressietest toegevoegd:
  relatieve route-meters `0–524` met hectometrering `25.8–26.3` worden in de
  concept-export `25800–26300`.
- Proefexport voor N354 gecontroleerd op de eerste HRB-regels:
  `N354-HRB-08.6-11.0` krijgt nu `8600–10960` in plaats van relatieve meters.
