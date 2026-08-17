# Release notes v0.36.7

v0.36.7 is een kleine afrondende release voor Project Adviseur 2.0.

De versie corrigeert de formulering in het automatische runrapport. In v0.36.6
kon de vervolgstap bij `Verschillen met iASSET` nog de indruk wekken dat het
oude handmatige N-wegendocument als beoordelingsbron gebruikt moest worden.
Dat is niet de bedoeling.

## Wijzigingen

- De vervolgstap bij `Verschillen met iASSET` is aangepast naar:
  - beoordelen met kaartbeeld;
  - actuele iASSET-data;
  - beschikbare broninformatie.
- Het concept-N-wegendocument wordt expliciet gepresenteerd als werkblad, niet
  als waarheid.
- Oude handmatige N-wegentabbladen worden niet gebruikt als automatische norm.
- Geen wijzigingen aan projectlogica, werklijstselectie, hectometerijking of
  N-wegendocument-exportstructuur.

## Waarom

Het eindproduct moet actuele iASSET-data zo goed mogelijk vertalen naar
onderhoudscomplexvoorstellen en werkbestanden. Oude handmatige tabbladen kunnen
context geven, maar kunnen per weg verouderd of onvolledig zijn. Daarom mogen ze
niet als waarheid of kalibratiebron in het standaardproces terechtkomen.

## Validatie

- `pytest -q`: 201 passed
- Alle Python-bestanden zijn succesvol gecontroleerd met `py_compile`.

## Impact voor gebruikers

Na het draaien van Project Adviseur blijft het werkproces:

1. gebruik het runrapport als eerste oordeel;
2. gebruik de voorstellenlijst als basis;
3. werk de werklijst af;
4. gebruik het concept-N-wegendocument als werkblad;
5. beoordeel verschillen met kaartbeeld, actuele iASSET-data en broninformatie;
6. pas iASSET pas aan na menselijke beoordeling.
