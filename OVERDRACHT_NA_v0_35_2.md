# Overdracht na v0.35.2

## Stand van zaken

v0.35.2 bouwt voort op v0.35.1. De greenfield-projectvoorstellen en knipzwaarte blijven inhoudelijk gelijk; de nieuwe stap is visuele inspectie.

## Functionele toevoeging

In `🧪 NWB referentieproef` kan de gebruiker een projectvoorstel selecteren. De hoofdkaart toont:

- paars: objecten die volgens de tool bij het geselecteerde groenveldvoorstel horen;
- blauw: objecten uit bestaande iASSET-onderhoudsprojecten die volgens de vergelijking raken aan het voorstel, maar niet in het voorstel zitten.

Daarnaast toont de UI details over:

- voorgestelde onderhoudsprojectnaam;
- fysieke begin/eindpositie;
- status;
- knipreden begin/eind;
- harde knipsignalen;
- zachte signalen;
- bestaande iASSET-projecten in de vergelijking;
- objecttoewijzing.

## Waarom deze stap

CSV's zijn geschikt voor overdracht, maar het beoordelen van onderhoudsprojectgrenzen is ruimtelijk werk. Deze kaartinspectie maakt het mogelijk om N398 en N354 sneller naast de werkelijkheid in iASSET te leggen.

## Verwachte test

Gebruik dezelfde workflow als v0.35.1:

1. kies N398 of N354;
2. open `🧪 NWB referentieproef`;
3. upload `wegassen_paspoort.geojson`;
4. haal NWB-wegvakken en hectopunten op;
5. open het blok `v0.35.2 Projectvoorstel op kaart inspecteren`;
6. filter/selecteer een voorstel;
7. controleer de highlight op de hoofdkaart.

## Let op

De kaartinspectie is nog een ontwikkel-/diagnoselaag. De definitieve functionele plek wordt later waarschijnlijk `Project Adviseur`, zodat de app geen wildgroei aan losse functionaliteiten krijgt.

## Mogelijke vervolgstap v0.35.3

Met de kaartinspectie kunnen we gericht beoordelen welke resterende knippen te technisch zijn. Mogelijke verbeteringen:

- besteknummer alleen hard maken in combinatie met andere kenmerken;
- minimumlengte per projecttype verfijnen;
- korte afwijkende objecten als context binnen groter voorstel houden;
- rotondes/ovondes/kruispunten als aparte uitzonderingscategorie behandelen;
- onderscheid maken tussen HRB-, PW-, FP- en BB-knipregels.
