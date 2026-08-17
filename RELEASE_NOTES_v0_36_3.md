# Release notes v0.36.3

## Doel

v0.36.3 maakt Project Adviseur bruikbaarder voor het dagelijkse databeheer door
een concept-export te maken in het format van het N-wegendocument.

De tool schrijft niet in het bestaande N-wegendocument en wijzigt niets in
iASSET. Het resultaat is een apart Excelbestand dat naast het handmatige
N-wegendocument gelegd kan worden.

## Nieuw

- Nieuwe downloadknop in Project Adviseur:
  `📘 Download concept N-wegendocument-tabblad`.
- Nieuwe export:
  `Projectadvies_Nwegendocument_<weg>.xlsx`.
- Nieuwe module:
  `iasset_tool/nwegendocument_export.py`.
- Concepttabbladen per wegdeeltype:
  - `<weg> (HRB)`
  - `<weg> (PW)`
  - `<weg> (FP)` indien aanwezig
- Ondersteunende tabbladen:
  - `Samenvatting`
  - `Niet automatisch gevuld`
  - `Runrapport`
  - `Conceptregels_data`

## Automatisch gevuld

Waar beschikbaar vult de export onder andere:

- onderhoudscomplex oud;
- onderhoudscomplex nieuw;
- knip begin/einde;
- verharding begin/einde;
- besteknummer;
- verhardingsoort;
- jaar aanleg;
- jaar deklaag;
- jaar conservering;
- jaar herstrating;
- bijzonderheden uit Project Adviseur.

## Bewust niet automatisch gevuld

Deze velden blijven conceptueel of leeg wanneer de brondata ze niet betrouwbaar
levert:

- locatie;
- documentatie/Dielplak-link;
- menselijke opmerkingen;
- definitieve keuze of een voorstel wordt overgenomen.

## Technisch

- Geen wijziging in de projectas-engine.
- Geen automatische iASSET-mutatie.
- Geen overschrijving van het bestaande N-wegendocument.
- Busbanen en landbouwpaden worden voor de concept-export bij het PW-tabblad
  geplaatst, conform het werkproces.

## Validatie

- `pytest -q`: 195 passed
- `python -m py_compile` op alle Python-bestanden
- Proefexport op basis van N354-exports gecontroleerd:
  - `Samenvatting`
  - `N354 (HRB)`
  - `N354 (PW)`
  - `N354 (FP)`
  - `Niet automatisch gevuld`
  - `Runrapport`
  - `Conceptregels_data`
