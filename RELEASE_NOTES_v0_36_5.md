# Release notes v0.36.5

## Doel

v0.36.5 maakt de concept-export in N-wegendocument-format beter vergelijkbaar
met het handmatige N-wegendocument.

v0.36.4 corrigeerde de hectometreringsschaal. Deze versie corrigeert de
zichtbare tabbladindeling.

## Gewijzigd

- `N354 (HRB)`/HRB-tabbladen gebruiken de statusblok-layout met oude complexen
  in kolom A en nieuwe complexen zichtbaar in kolom C.
- `N354 (PW)`/PW-tabbladen gebruiken de layout met:
  - kolom A: filterkolom;
  - kolom B: onderhoudscomplex oud;
  - kolom C: onderhoudscomplex nieuw.
- `N354 (FP)`/FP-tabbladen gebruiken de compacte fietspadlayout zonder
  knipkolommen.
- `Conceptregels_data` blijft het volledige technische brontabblad.

## Niet gewijzigd

- Geen nieuwe projectas-rekenlogica.
- Geen wijziging in werklijstselectie.
- Geen automatische mutatie in iASSET of in het bestaande N-wegendocument.

## Validatie

- `pytest -q`: 198 passed.
- `python -m py_compile` op alle Python-bestanden.
- Proefexport gecontroleerd op tabbladen, kolommen en hectometreringsmeters.
