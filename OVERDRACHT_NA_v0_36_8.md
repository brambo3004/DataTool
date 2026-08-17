# Overdracht na v0.36.8

## Status

v0.36.8 sluit de v0.36-reeks af als werkbare basisversie van Project Adviseur 2.0.

## Wat Project Adviseur nu kan

- Projectvoorstellen maken op basis van actuele iASSET-/paspoortdata.
- Voorstellen vergelijken met bestaande iASSET-projectnamen als controlecontext.
- Een werklijst maken met concrete databeheeractiepunten.
- Een runrapport maken met automatisch werkprocesadvies.
- Een concept-N-wegendocument exporteren.
- Volledige objecttoewijzing apart bewaren in `Objecttoewijzing_data`.
- De zichtbare kolom `objecten` beperken tot bijzondere objecten, wanneer die betrouwbaar herkenbaar zijn.

## Belangrijke ontwerpkeuze

Het concept-N-wegendocument is een werkblad, geen waarheid. Oude handmatige
N-wegentabbladen worden niet automatisch als validatiebron of kalibratiebron gebruikt.

## Volgende fase

De volgende grote stap is v0.37.0:

```text
N354 inhoudelijke kalibratie onderhoudscomplexen
```

Doel van die fase is onderzoeken waarom N354 nog veel meer voorstellen en werklijstregels
oplevert dan je als databeheerder waarschijnlijk als onderhoudscomplexen zou willen gebruiken.

Daarbij mag het handmatige N-wegendocument wel vergelijkingsmateriaal zijn om verschillen
te begrijpen, maar niet de bron van waarheid.
