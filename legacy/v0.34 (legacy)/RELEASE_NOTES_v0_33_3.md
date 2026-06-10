# Release notes v0.33.3

## NWB referentieproef

Deze patch voegt een kaartweergave toe aan de gedetailleerde iASSET-wegas versus NWB-vergelijking.

- Detailpunten langs de iASSET-wegas worden op kaart getoond.
- NWB-wegvakken worden als blauwe referentielijn getoond.
- iASSET-wegassen worden als paarse lijn getoond.
- Detailpunten krijgen een eenvoudige diagnosekleur:
  - groen: maximaal 10 meter van NWB;
  - oranje: tussen 10 meter en de ingestelde maximale afstand;
  - rood: groter dan de ingestelde maximale afstand.
- De kaart is uitsluitend bedoeld voor diagnose en overleg. Er worden geen iASSET-mutaties uitgevoerd of voorgesteld.

## Validatie

`163 passed`
