# Overdracht na v0.32 — referentieas/PDOK-proef

## Status na v0.32

v0.32 voegt een apart scherm `🧪 Referentieas / PDOK-proef` toe. De proef gebruikt de bestaande PDOK-hectometerfunctie om een experimentele referentieas op te bouwen en rijstrookobjecten hierop te projecteren.

De bestaande schermen blijven bewust gescheiden:

- Overzicht gebruikt nog steeds de v0.31-trajectlengtebronkeuze.
- Onderhoudscontrole is niet aangepast.
- Project Adviseur gebruikt de referentieas nog niet voor nieuwe namen of knippen.
- iASSET blijft de single source of truth.

## Nieuwe bestanden

- `iasset_tool/reference_axis.py`
- `tests/test_reference_axis.py`
- `RELEASE_NOTES_v0_32_0.md`

## Wat de proef doet

Voor de geselecteerde weg:

1. PDOK-hectometerpunten ophalen via de bestaande cache.
2. Hectometerpunten filteren op wegnummer als die kolom beschikbaar is.
3. Per hectometerwaarde één ankerpunt maken.
4. Een experimentele referentieas bouwen.
5. Alleen rijstroken projecteren op deze as.
6. Per object tonen:
   - referentie-begin/eind;
   - vijfmeter-afleiding;
   - referentielengte;
   - afstand tot as;
   - verschil met onderhoudsprojectnaam;
   - status en waarschuwing.
7. Per onderhoudsproject samenvatten en als CSV exporteren.

## Belangrijke beperkingen

- De referentieas is nog geen waarheid.
- PDOK-punten in een bbox kunnen punten van parallelle of kruisende wegen bevatten als filtering niet mogelijk is.
- De as is een experimentele lijn door hectometerpunten; bij gescheiden rijbanen, rotondes en kruispunten kan dit vertekenen.
- Alleen rijstroken zijn binnen scope van v0.32.
- De vijfmeterwaarden zijn alleen diagnosekolommen, nog geen beheerknippen.

## Voorstel v0.33

- Mogelijkheid toevoegen om een lokale referentieas/GeoJSON te uploaden naast PDOK.
- Betrouwbaarheidsscore toevoegen:
  - aantal ankerpunten;
  - monotoniciteit;
  - afstand tot as;
  - verschil tussen referentie en projectnaam;
  - dekking van begin/eind.
- Kaartoverlay uitbreiden met de opgebouwde referentieas zelf.
- Pas na voldoende betrouwbaarheid eventueel gebruiken als extra bron in Overzicht, maar nog steeds niet als automatische mutatiebron.
