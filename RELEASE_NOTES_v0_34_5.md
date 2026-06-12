# Release notes v0.34.5

## Doel

v0.34.5 is een kleine polish-versie van **Projectgrenzen op referentieas**. De rekenlogica van v0.34.3/v0.34.4 blijft bewust intact. Deze versie maakt vooral de compacte controle-export rustiger en beter leesbaar voor databeheercontrole.

## Nieuw

- `Projectcontrole_Referentieas_<weg>.csv` heeft nu aparte meldingskolommen:
  - `hoofdmelding`: de reden waarom de regel in de werklijst staat;
  - `contextmelding`: extra detailinformatie, zoals objectligging;
  - `melding`: blijft aanwezig voor herkenbaarheid, maar bevat nu dezelfde schone hoofdreden als `hoofdmelding`.
- Objectligging wordt niet meer vermengd met de hoofdreden van een projectgrensregel.
- De schermpreview van de compacte controlelijst toont nu ook `hoofdmelding` en `contextmelding`.

## Waarom

In v0.34.4 werkte de compacte controlelijst inhoudelijk goed, maar sommige regels lazen nog onrustig omdat objectligging-context in dezelfde tekstkolom stond als de hoofdreden. Voor een databeheerder moet de werklijst snel laten zien:

1. waarom moet ik deze regel controleren?
2. welke extra context is nuttig, maar niet de hoofdreden?

Daarom is de melding nu gesplitst.

## Niet gewijzigd

- Geen wijziging in projectgrensberekening.
- Geen wijziging in NWB-hectopuntprojectie.
- Geen wijziging in snap-tolerantie: standaard blijft 2,5 meter.
- Geen wijziging in de hectometer-naar-boven-regel.
- Geen automatische mutaties in iASSET.
- Geen automatische beheerknippen.
- Geen automatische projectnaamvoorstellen.

## Validatie

- Test toegevoegd voor de splitsing tussen `hoofdmelding` en `contextmelding`.
- Bestaande tests voor projectas, naamzones, gatcontrole, BBLR en compacte controlelijst blijven behouden.
