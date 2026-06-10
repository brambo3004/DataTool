# Release notes v0.31.0

## Kern

v0.31 corrigeert de bronkeuze voor trajectlengtes in Overzicht.

De tool behandelt losse objectpaspoort-metrering niet langer automatisch als precieze voorkeursbron. Als er alleen grove puntmetrering beschikbaar is en de onderhoudsprojectnaam een bruikbare range bevat, gebruikt de tool de onderhoudsprojectnaam als administratieve voorkeurslengte.

## Toegevoegd

- bronkwaliteit voor trajectlengte;
- aparte kolommen voor voorkeurstrajectlengte, naamlengte en objectmetrering;
- waarschuwing wanneer grove objectmetrering afwijkt van de projectnaam;
- centrale vijfmeterregel voor toekomstige beheerknippunten;
- naamafronding naar tienden voor toekomstige projectnaamvoorstellen.

## Niet toegevoegd

- geen PDOK-koppeling;
- geen automatische naamvoorstellen;
- geen automatische iASSET-mutaties.

## Validatie

`pytest`: 146 passed.
