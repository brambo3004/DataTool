# Release notes v0.34.2

## Type release

Verfijningsversie van **Projectgrenzen op referentieas**.

Deze release blijft diagnose/proef. De tool doet geen automatische mutaties, maakt geen beheerknippen en schrijft niets terug naar iASSET.

## Belangrijkste wijzigingen

### 1. BBLR voorlopig toegestaan

De projectnaamvalidator accepteert nu `LR` als gecombineerde links/rechts-situering bij projecttypen waar normaal een situering nodig is.

Voorbeeld:

```text
N354-BBLR-15.9-16.0
```

Wordt nu niet meer automatisch als naamvalidatie-aandachtspunt gemarkeerd.

Waarom:
In de testdata kwam `BBLR` voor. De beheerregel is nog niet definitief bevestigd, maar voorlopig lijkt dit een toegestane gecombineerde situering. Daarom accepteert de tool `LR` diagnostisch, zonder projectnamen automatisch voor te stellen of te wijzigen.

### 2. Gaten slimmer beoordeeld met fysieke objectaanwezigheid

v0.34.1 meldde gaten tussen opeenvolgende projecten van hetzelfde projecttype. Bij parallelwegen en fietspaden gaf dat nog ruis: het spoor bestaat niet altijd over de hele weg.

v0.34.2 meldt een gat alleen als controlepunt wanneer in dat interval ook primaire objecten van hetzelfde spoor liggen.

Voorbeelden:

- gat tussen twee `HRB`-projecten + daar ligt een rijstrookobject zonder passend project: `controleer`;
- gat tussen twee `FPR`-projecten + daar ligt fysiek geen fietspadobject: geen gatmelding;
- gat tussen twee `PWR`-projecten + daar ligt een parallelwegobject: `controleer`.

### 3. Primaire objecten zonder onderhoudsproject meegenomen in objectprojectie

`Projectobjecten_Referentieas_<weg>.csv` kan nu ook primaire objecten zonder onderhoudsproject bevatten.

Waarom:
Deze objecten zijn nodig om gaten betrouwbaar te beoordelen. Als er fysiek areaal ligt maar geen projectdekking aanwezig is, moet de tool dat kunnen signaleren.

Secundaire objecten zonder onderhoudsproject worden niet toegevoegd, om de export leesbaar te houden.

### 4. Objectligging blijft detaildiagnose

Objectligging blijft zichtbaar in de projectgrens- en objectexport, maar blijft buiten de hoofdstatus van projectgrenzen. Dit voorkomt dat parallelwegen en fietspaden te snel geel/rood worden door afstand tot de hoofdas.

## Aangepaste module

- `iasset_tool/project_axis.py`

Belangrijke nieuwe helpers:

- `_subtheme_project_family()`
- `_situering_code()`
- `_object_project_type_from_row()`
- `_project_type_matches_gap()`
- `_primary_object_presence_in_gap()`

## Tests

De volledige testset is gedraaid:

```text
174 passed
```

Nieuwe/gewijzigde tests:

- gat zonder fysieke primaire objecten wordt niet meer als controlepunt gemeld;
- gat met primair object zonder onderhoudsproject blijft zichtbaar;
- `BBLR` wordt als gecombineerde situering geaccepteerd.

## Niet gedaan in v0.34.2

Nog niet toegevoegd:

- automatische projectnaamvoorstellen;
- beheergrensvoorstellen;
- afronden naar 5-meter-systeem;
- automatische objecttoewijzing;
- mutaties richting iASSET.

