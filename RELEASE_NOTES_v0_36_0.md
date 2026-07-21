# Release notes v0.36.0

## Focus

v0.36.0 maakt van de bestaande v0.35.5-referentieas/greenfield-engine een
functioneel hoofdscherm: **Project Adviseur 2.0**.

De rekenlogica in `iasset_tool/project_axis.py` is bewust niet inhoudelijk
aangepast. De wijziging zit in de presentatie, statusordening en werklijst.

## Waarom

Na v0.35.5 was de technische motor ver genoeg om projectvoorstellen te maken,
maar de bruikbaarheid zat nog te veel in technische tabellen, diagnoseblokken en
CSV's. Deze versie haalt de projectvoorstellen naar het hoofdscherm en maakt het
verschil zichtbaar tussen:

- inhoudelijk projectadvies;
- datakwaliteit;
- grensbetrouwbaarheid;
- referentieas/ijking;
- samengevat eindadvies.

Daarmee wordt voorkomen dat bijvoorbeeld een lokaal ontbrekend besteknummer het
hele projectvoorstel als inhoudelijk fout laat lijken.

## Nieuw

- `🏗️ Project Adviseur` is de eerste modus in de app.
- Nieuw Project Adviseur 2.0-hoofdscherm met:
  - directe berekening van projectadvies op basis van de bestaande NWB-/projectas-engine;
  - compacte samenvatting bovenaan;
  - voorstellenlijst met gescheiden statussen;
  - kaartinspectie voor geselecteerd voorstel;
  - detailpaneel met knipreden, profielsignalen, hm-intervaldiagnose en iASSET-vergelijking;
  - compacte werklijst;
  - exports voor voorstellen en werklijst.
- Nieuwe presentatiemodule:
  - `iasset_tool/project_advisor_v2.py`
- Nieuwe tests:
  - `tests/test_project_advisor_v2.py`
- Technische diagnose en brede exports zijn behouden, maar staan in een expander.
- De oude adviesgroepenworkflow uit v0.35.5 is behouden achter:
  - `Toon oude adviesgroepen uit v0.35.5`

## Bewust niet gewijzigd

- Geen nieuwe GIS-kniplogica.
- Geen wijziging in de greenfield-reeksherkenning.
- Geen wijziging in de beheerregels voor:
  - technisch profiel;
  - besteknummer als ondersteunend signaal;
  - lokale afwijkingen;
  - snap-tolerantie;
  - hectometer-naar-boven-regel.
- Geen automatische iASSET-mutaties.

## Gewijzigde bestanden

- `app.py`
- `README.md`
- `iasset_tool/__init__.py`
- `iasset_tool/config.py`

## Nieuwe bestanden

- `iasset_tool/project_advisor_v2.py`
- `tests/test_project_advisor_v2.py`
- `RELEASE_NOTES_v0_36_0.md`
- `OVERDRACHT_NA_v0_36_0.md`

## Verwijderde bestanden

Geen.

## Validatie

- `pytest -q`: 189 passed
- `python -m py_compile` op alle Python-bestanden

## Testadvies Streamlit

1. Open de app.
2. Controleer dat `🏗️ Project Adviseur` de eerste modus is.
3. Kies `N398`.
4. Upload `wegassen_paspoort.geojson`.
5. Klik op `🚦 Maak projectadvies`.
6. Controleer dat:
   - er een samenvatting verschijnt;
   - voorstellen gescheiden statussen hebben;
   - lokale datakwaliteit niet automatisch de adviesstatus vervuilt;
   - een gekozen voorstel links op de kaart uitgelicht kan worden;
   - technische diagnose onder de expander blijft.
7. Herhaal later met `N354` om te controleren of complexe gevallen uitlegbaar
   blijven zonder dat het scherm dichtslibt.
