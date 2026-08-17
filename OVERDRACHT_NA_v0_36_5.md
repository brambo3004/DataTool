# Overdracht na v0.36.5

## Status

Project Adviseur kan nu naast CSV's ook een concept-Excelbestand maken dat beter
aansluit op de werkvorm van het N-wegendocument.

## Belangrijkste wijziging

De N-wegendocument-export gebruikt vanaf v0.36.5 tabbladspecifieke zichtbare
layouts:

- HRB: statusblok met oude complexen links en nieuwe complexen in kolom C.
- PW: filterkolom + oud in kolom B + nieuw in kolom C.
- FP: compact fietspadformat zonder knipkolommen.

De volledige technische conceptregels blijven beschikbaar in `Conceptregels_data`.

## Aandachtspunt

De export is nog steeds een concept. Locatie, documentatie en menselijke
opmerkingen blijven handmatige velden. Micro-/eindzonevoorstellen en buiten
ijkbereik moeten niet automatisch worden overgenomen.

## Aanbevolen test

Draai N354 opnieuw en download:

- `Projectadvies_Runrapport_N354.csv`
- `Projectadvies_Voorstellen_N354.csv`
- `Projectadvies_Werklijst_N354.csv`
- `Projectadvies_Nwegendocument_N354.xlsx`

Controleer primair automatisch:
- of het FP-tabblad 12 zichtbare kolommen heeft;
- of PW oud/nieuw in kolom B/C staat;
- of de knipwaarden nog steeds in hectometreringsmeters staan;
- of de aantallen overeenkomen met het runrapport.
