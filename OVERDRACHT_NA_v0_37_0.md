# Overdracht na v0.37.0

## Status

Project Adviseur 2.0 is sinds v0.36.8 een werkbare basisversie. v0.37.0 start
de volgende fase: inhoudelijke kalibratie van de onderhoudscomplexlogica, met
N354 als praktijkcase.

## Belangrijkste besluit

We gebruiken oude handmatige N-wegentabbladen niet als waarheid. Ze kunnen
context bieden, maar de tool moet werken op actuele iASSET-data, kaartbeeld,
broninformatie en generieke beheerlogica.

## Wat v0.37.0 toevoegt

De app bevat een nieuwe downloadknop:

`🧭 Download kalibratierapport Project Adviseur`

Deze maakt:

`Projectadvies_Kalibratie_<weg>.xlsx`

met tabbladen:

- Samenvatting
- Kandidaat_samenvoegen
- Korte_reguliere_voorstellen
- Dubbele_projectnamen
- Objectfamilie_mismatch
- Knipreden_analyse
- Hectometerinterval_context

## Interpretatie

Dit rapport is geen extra losse debug, maar een automatische kalibratielaag. Het
moet voorkomen dat de gebruiker handmatig door tientallen voorstellen moet
zoeken om te begrijpen waarom N354 versnipperd raakt.

## Volgende stap

Gebruik de N354-uitvoer van v0.37.0 om te bepalen welke eerste inhoudelijke
wijziging v0.37.1 moet krijgen. Waarschijnlijke kandidaten:

1. korte technische eilandjes beter herkennen en als kandidaat-samenvoeging
   behandelen;
2. objectfamilie strenger bewaken;
3. dubbele projectnamen samenvoegen of expliciet als parallel/deelcontext
   presenteren.

Nog niet direct alle drie tegelijk aanpassen. Kies één regelwijziging en test
daarna opnieuw op N398 en N354.
