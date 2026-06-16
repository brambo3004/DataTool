# Release notes v0.35.3

## Doel

v0.35.3 verbetert de projectvoorstellen vanaf nul door de kniplogica meer te laten redeneren als een databeheerder: eerst stabiele reeksen herkennen, daarna pas knippen.

## Belangrijkste wijzigingen

- Greenfield-projectvoorstellen gebruiken nu technische reeksherkenning.
- Leidend technisch profiel:
  - `Soort verharding_N`
  - `Soort deklaag specifiek`
  - `Jaar aanleg`
  - `Jaar deklaag`
  - `Jaar conservering`
  - `Jaar herstrating`
- `Besteknummer` is ondersteunend: wijziging of ontbrekende waarde wordt gemeld, maar veroorzaakt niet zelfstandig een harde knip.
- `verhardingssoort` is uit de kniplogica gehaald.
- Lokale afwijking is vastgelegd als maximaal 2 objecten, korter dan 100 meter en links/rechts hetzelfde technische profiel.
- Lokale ontbrekende waarden worden datakwaliteit binnen het voorstel.
- Lokale echte technische afwijkingen krijgen status `controleer`, maar blijven ingesloten in het voorstel.
- Structurele technische profielwijzigingen blijven een projectknip.

## Nieuwe exportvelden

`Projectvoorstellen_Referentieas_<weg>.csv`:
- `technisch_profiel`
- `bestek_signalen`
- `datakwaliteit_signalen`
- `lokale_afwijkingen`
- `ingesloten_objecten`

`Projectvoorstel_Objecten_<weg>.csv`:
- `technisch_profiel`
- `besteknummer_norm`
- `lokale_afwijking_type`
- `ingesloten_in_voorstel`
- `object_kniprol`

## Validatie

- `python -m py_compile app.py iasset_tool/project_axis.py iasset_tool/map_view.py`
- `185 passed`

