# Overdracht na v0.34.0

## Waarom deze stap?

v0.33.4 maakte NWB-afwijkingszones zichtbaar op de iASSET-wegas. v0.34.0
gebruikt die basis om onderhoudsprojectgrenzen te controleren op een geijkte
iASSET-referentieas.

De referentieas blijft geen doel op zich. Het doel is dat databeheer Grijs
beter kan zien of projectgrenzen, projectdekking en onderhoudsprojectnamen
logisch liggen.

## Werkwijze in de app

1. Open `🧪 NWB referentieproef`.
2. Upload `wegassen_paspoort.geojson`.
3. Laat de tool NWB-wegvakken en hectopunten ophalen.
4. De tool:
   - vergelijkt iASSET-wegas met NWB;
   - maakt afwijkingszones;
   - projecteert NWB-hectopunten op de iASSET-wegas;
   - ijkt de iASSET-wegas naar hectometrering;
   - controleert onderhoudsprojectgrenzen en projectdekking.

## Belangrijkste exports

- `Projectgrenzen_Referentieas_<weg>.csv`
  - projectnaamrange versus geijkte as;
  - lengteverschil;
  - begin/eind in oranje/rode zone;
  - fysieke objectligging.
- `Projectdekking_Referentieas_<weg>.csv`
  - totale dekking;
  - gaten;
  - overlap.
- `NWB_Hectopunten_op_iASSET_Wegas_<weg>.csv`
  - ijkpunten die gebruikt zijn voor hm ↔ routepositie.
- `Projectobjecten_Referentieas_<weg>.csv`
  - objectgeometrieën indicatief op de geijkte as.

## Interpretatie

- `ok`: geen aandachtspunt gevonden binnen de gekozen toleranties.
- `aandacht`: controleer, bijvoorbeeld bij oranje zone of lengteverschil.
- `controleer`: duidelijke waarschuwing, bijvoorbeeld rode zone, buiten ijkbereik of geen bruikbare ijking.

## Niet doen op basis van v0.34

- Geen automatische beheerknippen maken.
- Geen onderhoudsprojecten automatisch aanpassen.
- Geen iASSET-velden automatisch wijzigen.

## Aanbevolen volgende stap

Controleer N398 als groene controlecase en N354 als praktijkcase met oranje begin-/rotonde-/ovondezones en rode eindzone. Let vooral op:
- projectgrenzen die precies in een afwijkingszone vallen;
- gaten/overlap tussen opeenvolgende HRB-projecten;
- projecten waarbij de fysieke objectligging sterk afwijkt van de projectnaamrange.
