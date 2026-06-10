# Release notes v0.32.1

## Doel

v0.32.1 verstevigt de experimentele `🧪 Referentieas / PDOK-proef` op basis van de N354-testexports met 10m/25m/50m afstandsdrempels.

De wijziging blijft diagnostisch:
- geen automatische iASSET-mutaties;
- geen wijziging in Overzicht;
- geen wijziging in Onderhoudscontrole;
- geen wijziging in de v0.31-trajectlengtekeuze.

## Gewijzigd

### Extra kwaliteitsvlaggen op objectniveau

De objectdiagnose bevat nu extra kolommen:

- `binnen_afstandsdrempel`
- `buiten_projectrange`
- `projectiesprong`
- `bruikbaar_voor_projectsamenvatting`

Waarom?
De N354-export liet zien dat sommige objecten dicht op de referentieas liggen, maar toch naar een verkeerd of extreem deel van de opgebouwde as projecteren. Afstand tot de as alleen is dus onvoldoende als kwaliteitscontrole.

### Projectrange-bewaking

Een object krijgt de waarschuwing `referentie buiten projectrange` wanneer de experimentele referentieprojectie duidelijk buiten de range uit de onderhoudsprojectnaam valt.

De projectnaam blijft geen precieze waarheid, maar wordt wel gebruikt als vangrail tegen duidelijke projectiesprongen.

### Sprongdetectie

Een object krijgt de waarschuwing `onwaarschijnlijke referentiesprong` wanneer de referentielengte onlogisch groot is ten opzichte van de projectnaamlengte.

De gebruikte diagnosegrens is:

```text
referentie_lengte_m > max(500 m, project_lengte_m * 1.5)
```

### Robuustere projectsamenvatting

De projectsamenvatting gebruikt nu alleen objecten met:

```text
bruikbaar_voor_projectsamenvatting = True
```

Objecten met afstandsproblemen, projectrange-afwijkingen of projectiesprongen blijven zichtbaar in de objectdiagnose, maar trekken het begin/einde van een heel onderhoudsproject niet meer kapot.

### Extra kolommen in projectsamenvatting

Toegevoegd:

- `objecten_binnen_drempel`
- `objecten_buiten_projectrange`
- `objecten_met_projectiesprong`
- `objecten_bruikbaar_voor_projectsamenvatting`

## Tests

De testset is uitgebreid met een regressietest waarin één ontspoorde objectprojectie niet meer de projectsamenvatting mag domineren.

Validatie:

```text
151 passed
```

## Bekende beperking

De referentieas blijft experimenteel. Een object dat `bruikbaar_voor_projectsamenvatting = True` krijgt, is niet automatisch geschikt voor beheerknippen of projectnaamvoorstellen. Daarvoor is eerst visuele/domeincontrole nodig.
