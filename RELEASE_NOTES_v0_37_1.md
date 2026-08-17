# Release notes v0.37.1

## Doel

v0.37.1 voegt een zichtbare onderhoudscomplexlaag toe bovenop de ruwe Project Adviseur-voorstellen.

De ruwe projectvoorstellen blijven beschikbaar als technische onderbouwing. De nieuwe laag is de werklaag voor het concept-N-wegendocument en voor snelle databeheerbeoordeling.

## Nieuw

- Nieuwe module `iasset_tool/visible_complexes.py`.
- Nieuwe export:
  - `Projectadvies_Zichtbare_OnderhoudsComplexen_<weg>.csv`
- Streamlit toont een extra blok:
  - `Zichtbare onderhoudscomplexlaag`
- Het concept-N-wegendocument gebruikt de zichtbare laag als bron voor de HRB/PW/FP-tabbladen.
- Controlepunten die niet automatisch als normaal onderhoudscomplex thuishoren, worden apart bewaard in `Controlepunten_data`.

## Inhoudelijke regels

- Dubbele voorgestelde projectnamen worden als één zichtbaar cluster gepresenteerd.
- Micro-/eindzonevoorstellen blijven controlepunten.
- Reguliere voorstellen korter dan 100 meter blijven controlepunten en komen niet automatisch in de zichtbare HRB/PW/FP-tabbladen.
- Objectfamilie-mismatches worden benoemd als controlepunt, niet als zelfstandige projectbasis.
- Ruwe voorstellen, objecttoewijzing en kalibratierapport blijven beschikbaar.

## Niet veranderd

- De projectas-engine is niet inhoudelijk aangepast.
- Er is geen vergelijking met oude handmatige N-wegentabbladen toegevoegd.
- iASSET wordt niet automatisch gewijzigd.
- De ruwe Project Adviseur-voorstellen blijven downloadbaar.

## Validatie

- `pytest -q`
- `python -m py_compile` op alle Python-bestanden
- Nieuwe tests voor:
  - samenvoegen van dubbele projectnamen;
  - korte reguliere segmenten als controlepunt;
  - objectcontext bij zichtbare samengevoegde regels.
