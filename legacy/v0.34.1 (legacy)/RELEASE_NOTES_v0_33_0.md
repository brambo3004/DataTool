# Release notes v0.33.0

## Hoofdwijziging

v0.33.0 introduceert een nieuwe experimentele **NWB referentieproef**. Deze
proef onderzoekt de officiële NWB OGC API Features als bron voor wegvakken en
hectopunten. De bestaande Onderhoudscontrole, Overzicht en v0.31/v0.32
trajectlogica blijven ongemoeid.

## Nieuwe module

- `iasset_tool/nwb.py`
  - haalt NWB-wegvakken op via OGC API Features binnen de bbox van de geselecteerde iASSET-data;
  - filtert wegvakken op wegnummer/routenummer;
  - haalt NWB-hectopunten op en koppelt deze via `wvk_id`;
  - maakt een compacte bronsamenvatting;
  - kan een iASSET-wegassen-GeoJSON lezen;
  - vergelijkt optioneel de iASSET/Dielplak-wegas met NWB-wegvakken.

## Nieuwe UI

- Nieuw scherm: `🧪 NWB referentieproef`
- Instellingen:
  - NWB-opvraagbuffer rond iASSET-objecten;
  - maximale afstand tussen iASSET-wegas en NWB-wegvakken;
  - maximaal aantal features per NWB-pagina;
  - optionele upload van `wegassen_paspoort.geojson`.

## Exports

- `NWB_Bronsamenvatting_<weg>.csv`
- `NWB_Wegvakken_<weg>.csv`
- `NWB_Hectopunten_<weg>.csv`
- `NWB_Wegasvergelijking_<weg>.csv` als een wegassen-GeoJSON is geüpload.

## Veiligheidsregel

De NWB-proef is alleen diagnose. De tool signaleert en exporteert, maar voert
geen automatische wijzigingen door in iASSET. iASSET blijft de single source of
truth.
