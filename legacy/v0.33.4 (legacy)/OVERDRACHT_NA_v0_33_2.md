# Overdracht na v0.33.2

## Stand
v0.33.2 is een kleine patch op de NWB-referentieproef.

De app kan nu naast de samenvattende iASSET-wegas versus NWB-vergelijking ook een detail-export maken met samplepunten langs de iASSET-wegas. Dit is bedoeld om lokale uitschieters, zoals de grote afwijking op `WA-N354_1`, ruimtelijk terug te vinden.

## Belangrijk
De NWB-proef blijft experimenteel en alleen-lezen:
- geen automatische mutaties in iASSET;
- geen automatische beheerknippen;
- geen projectnaamvoorstellen;
- geen wijziging in bestaande Overzicht- of Onderhoudscontrole-schermen.

## Nieuwe export
`NWB_Wegasvergelijking_Detail_<weg>.csv`

Deze bevat per samplepunt:
- `afstand_langs_iasset_wegas_m`
- `x_rd`
- `y_rd`
- `afstand_tot_nwb_m`
- `dichtstbijzijnde_nwb_wvk_id`
- `status`
- `waarschuwing`

## Aanbevolen vervolgtest
1. Open v0.33.2.
2. Kies N354.
3. Upload `wegassen_paspoort.geojson`.
4. Draai `🧪 NWB referentieproef`.
5. Download:
   - `NWB_Wegasvergelijking_N354.csv`
   - `NWB_Wegasvergelijking_Detail_N354.csv`
6. Sorteer de detail-export aflopend op `afstand_tot_nwb_m` om de lokale uitschieter(s) op `WA-N354_1` te vinden.

## Validatie
`161 passed`
