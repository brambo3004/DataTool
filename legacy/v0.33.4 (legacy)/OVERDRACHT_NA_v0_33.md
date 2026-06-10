# Overdracht na v0.33.0

## Context

v0.33.0 verlegt de referentierichting van een zelfgebouwde PDOK-puntenas naar
een officiële NWB-bronverkenning. Aanleiding was de vondst dat het Nationaal
Georegister/PDOK niet alleen hectopunten aanbiedt, maar ook NWB-wegvakken via
OGC API Features.

## Belangrijkste besluit

De bestaande v0.32.1 `Referentieas / PDOK-proef` blijft beschikbaar als
experimenteel diagnosekanaal, maar is niet langer de voorkeursrichting voor
verdere verfijning. Voor vervolgwerk is de route:

1. NWB-wegvakken ophalen.
2. NWB-hectopunten via `wvk_id` koppelen.
3. iASSET/Dielplak-wegas optioneel naast NWB leggen.
4. Pas daarna iASSET-objecten en projectnamen inhoudelijk vergelijken.

## Nieuwe code

- `iasset_tool/nwb.py`
- `tests/test_nwb.py`
- Nieuw scherm in `app.py`: `🧪 NWB referentieproef`

## Wat v0.33.0 wel doet

- NWB-wegvakken ophalen via OGC API Features.
- NWB-wegvakken filteren op de geselecteerde weg.
- NWB-hectopunten koppelen via `wvk_id`.
- Bronsamenvatting tonen en exporteren.
- Optioneel iASSET-wegassen-GeoJSON vergelijken met NWB-wegvakken.
- Fouten/netwerkproblemen als waarschuwing tonen in plaats van crashen.

## Wat v0.33.0 bewust nog niet doet

- Geen beheerknippen voorstellen.
- Geen projectnamen automatisch voorstellen.
- Geen iASSET-mutaties.
- Geen vervanging van Onderhoudscontrole of Overzicht.
- Nog geen route-opbouw over NWB-wegvakken voor exacte objectbegin/eindmetrering.

## Vervolgadvies

Eerst met N354 en N398 testen:
- hoeveel NWB-wegvakken worden gevonden;
- hoeveel hectopunten via `wvk_id` meekomen;
- of de iASSET-wegas dicht genoeg op NWB ligt;
- waar meerdere assen, gescheiden rijbanen of parallelstructuren zichtbaar worden.

Pas daarna kan v0.34 onderzoeken hoe objectgrenzen betrouwbaar op NWB/iASSET-as
kunnen worden geprojecteerd.
