# Overdracht na v0.33.4

## Waarom deze stap?

De NWB-referentieproef wordt gebruikt als instrument om later onderhoudsproject-
grenzen beter te beoordelen. De referentieas is dus geen doel op zich.

v0.33.3 maakte afwijkingen zichtbaar als losse detailpunten. v0.33.4 vat die
detailpunten samen tot afwijkingszones, zodat een databeheerder niet honderden
punten hoeft na te lopen.

## Nieuwe export

`NWB_Wegas_Afwijkingszones_<weg>.csv`

Deze export geeft per zone aan:
- waar de zone langs de iASSET-wegas ligt;
- hoe groot de afwijking tot NWB is;
- of de zone oranje of rood is;
- of het een beginzone, eindzone, lokale afwijking of langere afwijking is;
- welke relevantie dit heeft voor onderhoudsprojectgrenzen.

## Interpretatie

- Oranje bij rotonde/ovonde/aansluiting is meestal een aandachtspunt, geen
  automatische fout.
- Rood bij einde of begin van een wegas is belangrijker, omdat dit projectgrenzen
  of trajectafbakening kan beïnvloeden.
- Rotondes zijn risicolocaties, maar niet per definitie foutlocaties.

## Technisch

Nieuwe modulefunctie:
- `build_nwb_wegas_deviation_zones`

Bestaande v0.33.3-detailkaart blijft aanwezig.

## Nog niet gedaan

- Nog geen automatische onderhoudsprojectgrenzen.
- Nog geen automatische beheerknippen.
- Nog geen mutaties in iASSET.
- Nog geen koppeling met projectnamen of vijfmeter-knipsystematiek.

## Volgende logische stap

Test dezelfde lijn met N398. Daarna kunnen we een eerste ontwerp maken voor:

- projectgrenzen vergelijken met NWB/iASSET-wegas;
- projectnaam-begin/eind vergelijken met zones;
- waarschuwingen zoals `projectgrens ligt in afwijkingszone`.
