# Release notes v0.35.5

## Focus

v0.35.5 voegt een vaste hectometerintervalcontrole toe aan de referentieas- en
greenfield-projectvoorstellen. De kniplogica van v0.35.3/v0.35.4 blijft bewust
intact.

## Waarom

Bij de N398 bleek dat sommige opeenvolgende hectometerpunten fysiek geen
verwachte 100 meter uit elkaar liggen. Vooral het interval 6.2-6.3 is korter.
Zulke afwijkende intervallen verklaren waarom een grens in fysieke meters anders
kan lijken dan de geijkte hectometrering.

## Nieuw

- Nieuwe export:
  - `Hectometerintervallen_Referentieas_<weg>.csv`

- Nieuwe intervalkolommen:
  - `hm_van`
  - `hm_tot`
  - `hm_interval`
  - `route_m_van`
  - `route_m_tot`
  - `verwachte_lengte_m`
  - `gemeten_lengte_m`
  - `afwijking_m`
  - `afwijking_pct`
  - `interval_factor`
  - `status`
  - `melding`

- Nieuwe begin/eindvelden in `Projectvoorstellen_Referentieas_<weg>.csv` en
  waar relevant in projectgrensdiagnose:
  - `*_hm_interval_status`
  - `*_hm_interval_melding`
  - `*_hm_interval_afwijking_pct`
  - `*_hm_interval_factor`
  - `*_grens_buiten_ijkbereik`

- Het scherm toont in de ontwikkel-/diagnose-expander een samenvatting van
  intervalstatussen en een downloadknop voor de volledige intervaldiagnose.

## Interpretatie

Een afwijkend hectometerinterval is geen automatische fout. Het is een
betrouwbaarheidssignaal voor projectgrenzen. Een grens in een afwijkend interval
wordt beter uitlegbaar, bijvoorbeeld bij het einde van de N398.

## Validatie

- `pytest -q`: 186 passed
- `python -m py_compile app.py iasset_tool/project_axis.py iasset_tool/map_view.py`
