# Release notes v0.34.4

## Doel

v0.34.4 is een gebruiksversie van **Projectgrenzen op referentieas**. De rekenlogica van v0.34.3 blijft bewust intact; deze versie maakt de resultaten beter bruikbaar voor databeheercontrole.

## Nieuw

- Nieuw samenvattingsblok in het NWB-scherm:
  - aantal projectgrenzen met `controleer`;
  - aantal projectgrenzen met `aandacht`;
  - aantal gaten;
  - aantal overlaps.
- Nieuwe compacte export:
  - `Projectcontrole_Referentieas_<weg>.csv`
- De compacte controlelijst bevat alleen regels die handmatige controle vragen:
  - projectnamen/projectgrenzen met `aandacht` of `controleer`;
  - gaten en overlaps uit de projectdekking.
- Objectligging blijft detailcontext en wordt niet als zelfstandige hoofdwaarschuwing in de compacte werklijst opgenomen.
- Als er geen gaten of overlaps zijn, toont het scherm nu expliciet een groene melding.

## Waarom

De test met N398 en N354 liet zien dat v0.34.3 inhoudelijk bruikbaar is:
- N398 blijft rustig;
- N354 toont gerichte controlepunten;
- valse gatmeldingen zijn verdwenen door snap- en naamzone-logica.

Voor databeheerders is de brede diagnose-export echter te technisch als eerste werklijst. v0.34.4 voegt daarom een compacte controlelijst toe, zonder de volledige exports te verwijderen.

## Niet gewijzigd

- Geen automatische mutaties in iASSET.
- Geen automatische beheerknippen.
- Geen automatische projectnaamvoorstellen.
- Geen wijziging in de snap-tolerantie van 2,5 meter.
- Geen wijziging in de hectometer-naar-boven-regel.

## Validatie

- Nieuwe tests toegevoegd voor:
  - compacte controlelijst;
  - robuuste samenvatting bij lege tabellen.
- Bestaande projectas-tests blijven behouden.
