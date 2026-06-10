# Release notes v0.33.2

## Doel
Kleine patch op de NWB-referentieproef. v0.33.1 liet zien dát een iASSET-wegas lokaal kan afwijken van NWB. v0.33.2 voegt detailpunten toe zodat zichtbaar wordt wáár langs de wegas die afwijking zit.

## Aangepast
- Nieuwe detailfunctie in `iasset_tool/nwb.py`:
  - `compare_iasset_wegassen_to_nwb_detail`
- Nieuwe export in het scherm `🧪 NWB referentieproef`:
  - `NWB_Wegasvergelijking_Detail_<weg>.csv`
- Nieuwe instelling:
  - detail-sampleafstand langs de iASSET-wegas, standaard 100 meter.
- De bestaande samenvattende wegasvergelijking blijft behouden.
- Overzicht, Onderhoudscontrole, Referentieas/PDOK-proef en trajectlengtekeuze zijn niet gewijzigd.

## Detail-export
Per samplepunt wordt vastgelegd:
- wegasnummer en naam;
- afstand langs de iASSET-wegas;
- RD-coördinaten van het samplepunt;
- afstand tot het dichtstbijzijnde NWB-wegvak;
- dichtstbijzijnde NWB-wvk_id;
- wegnummer/routenummer/begin-eindkm van dat NWB-wegvak;
- status en waarschuwing.

## Validatie
- `161 passed`
