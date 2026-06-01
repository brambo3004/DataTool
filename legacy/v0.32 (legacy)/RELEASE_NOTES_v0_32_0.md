# Release notes v0.32.0

## Kern

v0.32 voegt een afgeschermde experimentele referentieas/PDOK-proef toe. De proef is bedoeld om voor N354 en N398 te onderzoeken of PDOK-hectometerpunten kunnen helpen bij preciezere begin- en eindmetrering.

Belangrijk: deze laag is diagnose. De bestaande v0.31-bronkeuze voor trajectlengtes blijft leidend in Overzicht en Onderhoudscontrole wordt niet geraakt. iASSET blijft de single source of truth.

## Toegevoegd

- Nieuw scherm `🧪 Referentieas / PDOK-proef`.
- Nieuwe module `iasset_tool/reference_axis.py`.
- Experimentele opbouw van een referentieas uit PDOK-hectometerpunten.
- Objectdiagnose voor rijstroken:
  - referentie-begin/eind in km;
  - vijfmeter-afleiding voor beheerknippen;
  - referentielengte;
  - afstand tot de as;
  - verschil met onderhoudsprojectnaam;
  - status en waarschuwing.
- Projectsamenvatting per onderhoudsproject.
- CSV-downloads voor projectsamenvatting en objectdiagnose.
- Tests voor de referentieasproef en de expliciete vijfmetergrenzen.

## Bewust niet gewijzigd

- Geen automatische mutaties in iASSET.
- Geen wijziging in `trajectory_quantity_for_group()`.
- Geen wijziging in Onderhoudscontrole.
- Geen wijziging in bestaande Overzicht-hoeveelheden.
- Geen automatische projectnaamvoorstellen.
- Parallelwegen, fietspaden, rotondes en kruispunten blijven buiten de eerste proef.

## Bronkwaliteit

Alle referentieaswaarden krijgen bronkwaliteit `experimenteel`.
