# Release notes v0.34.3

## Doel

v0.34.3 verfijnt de projectgrensdiagnose met projectnaamzones en een instelbare snap-tolerantie.
De versie blijft nadrukkelijk diagnose/proef: iASSET wordt niet gewijzigd.

## Wijzigingen

- Standaard snap-tolerantie projectgrens naar hectometerpunt toegevoegd: **2,5 meter**.
- Fysieke objectgrenzen binnen deze tolerantie worden als het dichtstbijzijnde hectometerpunt behandeld.
- Buiten de tolerantie blijft de bestaande onderhoudsprojectnaamregel gelden: naar boven afronden op het volgende hectometerpunt.
- `Projectgrenzen_Referentieas_<weg>.csv` bevat extra kolommen voor:
  - dichtstbijzijnde hectometerpunt;
  - snap-afstand;
  - of de grens gesnapt is;
  - toegepast naamregelresultaat.
- Gatcontrole gebruikt nu projectnaamzones voordat primaire objecten in het gat worden beoordeeld.
- `Projectdekking_Referentieas_<weg>.csv` bevat extra kolommen voor:
  - hard gat na aftrek van naamzones;
  - marge aan linker- en rechterprojectgrens;
  - hard-gat-lengte.
- Streamlit-scherm uitgebreid met instelling:
  - `Snap-tolerantie projectgrens naar hectometerpunt (m)`.

## Belangrijke beheerregel

Voorbeeld bij snap-tolerantie 2,5 m:

- 12.302 ligt 2 m na 12.300 en wordt behandeld als 12.3.
- 12.303 ligt 3 m na 12.300 en wordt behandeld volgens de naar-boven-regel: 12.4.

## Validatie

- `177 passed`
- `python -m py_compile app.py iasset_tool/project_axis.py`
