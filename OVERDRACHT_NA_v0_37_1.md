# Overdracht na v0.37.1

## Status

Project Adviseur 2.0 heeft nu een zichtbare onderhoudscomplexlaag.

Deze laag zit tussen:
- de ruwe projectvoorstellen uit de projectas-engine;
- het concept-N-wegendocument dat de databeheerder als werkblad gebruikt.

## Waarom toegevoegd?

N354 liet zien dat de ruwe projectvoorstellen technisch bruikbaar zijn, maar te veel versnippering bevatten om direct als zichtbaar onderhoudscomplexwerkblad te gebruiken.

v0.37.1 lost dat nog niet definitief op in de motor, maar zet wel een belangrijke werklaag neer:
- dubbele projectnamen worden geclusterd;
- micro-/eindzonegevallen worden niet als normale onderhoudscomplexen gepresenteerd;
- zeer korte reguliere technische segmenten worden controlepunt;
- objectfamilie-mismatches worden als controle benoemd.

## Nieuwe uitvoer

- `Projectadvies_Zichtbare_Onderhoudscomplexen_<weg>.csv`
- In `Projectadvies_Nwegendocument_<weg>.xlsx`:
  - HRB/PW/FP-tabbladen gebruiken zichtbare onderhoudscomplexen;
  - `Controlepunten_data` bevat regels die niet automatisch in het conceptwerkblad thuishoren;
  - `Objecttoewijzing_data` blijft de objectcontext bewaren;
  - `Conceptregels_data` blijft de zichtbare N-wegendocumentvertaling tonen.

## Belangrijk

De projectas-engine is nog niet inhoudelijk gewijzigd. Dit is bewust.

v0.37.1 maakt zichtbaar welke laag de databeheerder als werkblad gebruikt, zonder de ruwe technische onderbouwing weg te gooien.

## Volgende stap

v0.37.2 of v0.38.x kan inhoudelijker worden:
- beoordelen of korte technische segmenten automatisch bij aangrenzende onderhoudscomplexen moeten worden gevoegd;
- onderzoeken hoe kruispunten/rotondes/bruggen beter als onderhoudscomplexgrens of eigen complex worden herkend;
- objectfamilie-toewijzing strenger maken zonder kruispuntcontext te verliezen.
