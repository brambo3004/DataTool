# Release notes v0.35.4

## Focus

v0.35.4 voegt grens- en hectometerintervaldiagnose toe aan de
greenfield-projectvoorstellen. De kniplogica van v0.35.3 blijft bewust intact.

## Waarom

Bij de N398 bleek dat het laatste hectometerinterval 6.2-6.3 fysiek veel korter
is dan 100 meter. Daardoor kan een fysieke grens die ongeveer 53 meter na hm 6.2
ligt toch geijkt worden als circa 6.273. Zonder extra diagnose lijkt dat een
onverklaarbare afwijking.

## Nieuw

- Nieuwe exportkolommen in `Projectvoorstellen_Referentieas_<weg>.csv`:
  - `begin_hm_interval`
  - `begin_hm_interval_lengte_m`
  - `begin_hm_interval_verwacht_m`
  - `begin_hm_interval_afwijking_m`
  - `begin_grenspositie_in_interval_m`
  - `begin_grenspositie_in_interval_pct`
  - `begin_grensdiagnose`
  - `eind_hm_interval`
  - `eind_hm_interval_lengte_m`
  - `eind_hm_interval_verwacht_m`
  - `eind_hm_interval_afwijking_m`
  - `eind_grenspositie_in_interval_m`
  - `eind_grenspositie_in_interval_pct`
  - `eind_grensdiagnose`
  - `grensdiagnose`

- De projectvoorstel-selectielijst toont nu naast fysieke route-meters ook
  geijkte hm/km-waarden.
- De detailtabel in de kaartinspectie toont de nieuwe grensdiagnosekolommen.
- Voorstellen met vrijwel nul fysieke lengte worden expliciet `controleer`.

## Validatie

- `python -m py_compile app.py iasset_tool/project_axis.py iasset_tool/map_view.py`
- `pytest -q`: 186 passed
