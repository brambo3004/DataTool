# Release notes v0.37.2

## Doel

Hotfix voor een Streamlit-fout in v0.37.1.

## Opgelost

- Streamlit gaf `StreamlitDuplicateElementId` bij de downloadknop **Download zichtbare onderhoudscomplexlaag**.
- De oorzaak was dat dezelfde downloadknop op twee plekken in Project Adviseur zichtbaar kon zijn zonder expliciete `key`.
- Alle downloadknoppen binnen het Project Adviseur-hoofdblok hebben nu expliciete, unieke keys.

## Niet gewijzigd

- Geen nieuwe projectlogica.
- Geen wijziging in de zichtbare onderhoudscomplexlaag.
- Geen wijziging in het concept-N-wegendocument.
- Geen wijziging in de kalibratiediagnose.

## Test

- `pytest -q`
- `py_compile` op alle Python-bestanden.
