# Release notes v0.33.4

## Kern

v0.33.4 voegt afwijkingszones toe aan de NWB-referentieproef.

De referentieas is hiermee nadrukkelijk geen doel op zich. De zone-uitvoer is
bedoeld als praktisch hulpmiddel voor latere beoordeling van onderhoudsproject-
grenzen, begin-/eindmetrering en beheerknippen.

## Nieuw

- Nieuwe functie `build_nwb_wegas_deviation_zones`.
- Nieuwe export:
  - `NWB_Wegas_Afwijkingszones_<weg>.csv`
- Nieuwe tabel in het scherm `🧪 NWB referentieproef`:
  - `Afwijkingszones voor projectgrenzen`

## Wat doet de zonelogica?

De detailpunten uit v0.33.3 worden geclusterd tot zones:

- oranje zones: lokaal aandachtspunt tussen 10 m en de ingestelde maximale afstand;
- rode zones: buiten de ingestelde maximale afstand;
- beginzone/eindzone: aandachtspunt ligt bij begin of einde van een iASSET-wegas;
- lokale afwijking: kort lokaal verschil, bijvoorbeeld bij rotonde/ovonde/aansluiting;
- langere afwijking: langer traject waar iASSET-wegas en NWB mogelijk een andere hoofdstructuur volgen.

## Beheergerichte kolommen

De nieuwe export bevat onder andere:

- `zone_id`
- `afstand_van_m`
- `afstand_tot_m`
- `lengte_zone_m`
- `max_afstand_tot_nwb_m`
- `kleurklasse`
- `zone_type`
- `advies`
- `relevantie_projectgrenzen`

## Belangrijk

De zones wijzigen geen iASSET-data. Ze zijn diagnose en dienen als hulpmiddel
voor gesprekken over projectgrenzen en latere verfijning van beheerknippen.

## Validatie

- `165 passed`
