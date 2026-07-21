# Overdracht na v0.36.1

## Status

v0.36.1 is een gerichte kalibratie van Project Adviseur 2.0. De versie maakt
de werklijst bruikbaarder vanuit het perspectief van de databeheerder.

De centrale lijn blijft:

```text
Project Adviseur = voorstellen + kaartinspectie + werklijst.
iASSET blijft bron van waarheid.
De tool voert geen automatische mutaties uit.
```

## Wat is aangepast

### 1. Werklijst is nu een actielijst

In v0.36.0 kwamen bijna alle N398-voorstellen in de werklijst terecht. In
v0.36.1 wordt een voorstel alleen werklijstregel wanneer er een concrete reden is:

- verschil met bestaande iASSET-indeling;
- begin/eind of grens buiten betrouwbaar ijkbereik;
- eindgrens bij een afwijkend laatste hm-interval;
- micro-/eindzonevoorstel;
- expliciete handmatige beoordeling nodig.

Lokale datakwaliteit blijft zichtbaar, maar zet een voorstel niet automatisch in
de werklijst als het voorstel verder overeenkomt met iASSET.

### 2. Nieuwe functionele kolommen

De voorstellenlijst heeft extra presentatiewaarden:

| Kolom | Betekenis |
|---|---|
| `werkadvies` | Korte actie voor de databeheerder. |
| `werklijst_reden` | Waarom dit voorstel wel/niet in de werklijst staat. |
| `voorstelcategorie` | Regulier voorstel, micro-/eindzonevoorstel of buiten ijkbereik. |
| `iassetvergelijking` | Samenvatting van de vergelijking met bestaande iASSET-projecten. |
| `in_werklijst` | True/False voor de actielijst. |

### 3. N398-observatie na kalibratie

Voor N398 moet vooral zichtbaar worden:

- `N398-HRB-01.6-04.6` blijft één voorstel;
- lokale ontbrekende besteknummers blijven datakwaliteit, geen projectknip;
- iASSET-verschillen blijven actiepunt;
- micro-/nulmeterprojecten worden apart behandeld;
- de werklijst is korter en functioneler dan in v0.36.0.

## Bestanden

### Gewijzigd

```text
app.py
README.md
BACKLOG.md
iasset_tool/__init__.py
iasset_tool/config.py
iasset_tool/project_advisor_v2.py
tests/test_project_advisor_v2.py
```

### Nieuw

```text
RELEASE_NOTES_v0_36_1.md
OVERDRACHT_NA_v0_36_1.md
```

### Verwijderd

```text
geen
```

## Testplan

### Test 1 — basisroute N398

1. Start de app.
2. Controleer versie `v0.36.1`.
3. Kies `N398`.
4. Upload `wegassen_paspoort.geojson`.
5. Klik op `🚦 Maak projectadvies`.

Verwacht:

- Project Adviseur toont de nieuwe samenvatting.
- De voorstellenlijst bevat `werkadvies`, `voorstelcategorie`, `iassetvergelijking`
  en `werklijst_reden`.
- De werklijst bevat minder regels dan de voorstellenlijst.

### Test 2 — traject N398-HRB-01.6-04.6

Controleer dat dit voorstel:

- zichtbaar blijft in de voorstellenlijst;
- niet wordt opgeknipt door lokale ontbrekende besteknummers;
- niet alleen door lokale datakwaliteit in de werklijst komt.

### Test 3 — micro/eindzone

Controleer de regels rond:

```text
N398-HRB-04.8-04.8
N398-HRB-06.3-06.3
```

Verwacht:

- voorstelcategorie `micro-/eindzonevoorstel`;
- wel zichtbaar in de werklijst;
- niet presenteren als normaal onderhoudsproject.

## Volgende logische stap

Na v0.36.1 is het verstandig om N398 opnieuw te beoordelen in Streamlit. Als de
werklijst functioneel genoeg is, kan daarna N354 worden getest als complexe
praktijkcase.
