# Overdracht na v0.34.3

## Status

v0.34.3 is een verfijning van de v0.34-projectgrensdiagnose. De iASSET-wegas blijft leidend; NWB-hectopunten blijven ijkpunten.
De tool wijzigt niets in iASSET.

## Wat is nieuw

De diagnose houdt nu rekening met het feit dat objectgeometrie vrijwel nooit exact op een hectometerpunt begint of eindigt.
Daarom is een snap-tolerantie toegevoegd met standaardwaarde 2,5 meter.

Werkvolgorde voor fysieke grensinterpretatie:

1. Projecteer de fysieke grens op de geijkte iASSET-as.
2. Bepaal de dichtstbijzijnde hectometer.
3. Als de grens binnen 2,5 m van dat hectometerpunt ligt: gebruik dat hectometerpunt.
4. Anders: pas de bestaande naar-boven-naamregel toe.

## Waarom dit belangrijk is

Zonder snap-tolerantie zou 12.301 altijd 12.4 worden. In de praktijk kan dat een geometrische afwijking van 1 meter zijn bij een grens die bedoeld is als 12.3.
Met v0.34.3 wordt dit als 12.3 geïnterpreteerd, zolang de afstand binnen de ingestelde tolerantie valt.

## Testadvies

Draai opnieuw N398 en N354 en upload deze exports:

- Projectgrenzen_Referentieas_N398.csv
- Projectdekking_Referentieas_N398.csv
- Projectobjecten_Referentieas_N398.csv
- Projectgrenzen_Referentieas_N354.csv
- Projectdekking_Referentieas_N354.csv
- Projectobjecten_Referentieas_N354.csv

Voor N398 verwacht: rustig beeld zonder gaten/overlap.
Voor N354 verwacht: de echte overlapkandidaten blijven zichtbaar; gatmeldingen worden minder gevoelig voor afrondmarges bij projectgrenzen.

## Nog open

- Praktijkvalidatie op de echte N354/N398 exports.
- Eventueel snap-tolerantie finetunen na meerdere wegen.
- Daarna pas nadenken over projectnaamvoorstellen.
