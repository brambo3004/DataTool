# Overdracht na v0.36.2

## Status

v0.36.2 rondt de eerste bruikbare Project Adviseur-stap voor N398 functioneel af.
De tool maakt nu niet alleen projectvoorstellen en een werklijst, maar geeft ook
automatisch aan hoe de volledige run moet worden gelezen.

De centrale lijn blijft:

```text
Project Adviseur = voorstellen + kaartinspectie + werklijst + runrapport.
iASSET blijft bron van waarheid.
De tool voert geen automatische mutaties uit.
```

## Wat is aangepast

### 1. Automatisch run-oordeel

Na het draaien van Project Adviseur verschijnt bovenaan een automatisch oordeel,
bijvoorbeeld:

```text
Bruikbaar als databeheeradvies met werklijst
```

Dat oordeel is bedoeld om de eerste databeheervraag te beantwoorden:

```text
Kan ik deze analyse gebruiken als werkbasis?
```

### 2. Runrapport

Er is een nieuw rapport toegevoegd:

```text
Projectadvies_Runrapport_<weg>.csv
```

Dit rapport voorkomt dat de gebruiker handmatig specifieke tabelregels moet
zoeken. Het vat de run samen in onderdelen, waarden, betekenis, vervolgstap en
urgentie.

### 3. Geen nieuwe engine-logica

De Project Adviseur gebruikt nog steeds de bestaande projectas-uitkomsten. De
wijziging zit in de presentatielaag en het werkproces.

## Praktische betekenis voor databeheer

Een databeheerder kan nu:

1. een weg selecteren;
2. iASSET-/wegasdata uploaden;
3. projectadvies draaien;
4. het runrapport bekijken/downloaden;
5. de werklijst afwerken;
6. daarna pas conclusies verwerken in iASSET, N-wegendocument of overleg.

De gebruiker hoeft niet meer handmatig te controleren of één specifiek voorstel
wel of niet in de werklijst staat om te weten of de run als geheel bruikbaar is.

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
RELEASE_NOTES_v0_36_2.md
OVERDRACHT_NA_v0_36_2.md
```

### Verwijderd

```text
geen
```

## Testplan vanaf nu

### Test 1 — N398 runrapport

1. Start de app.
2. Controleer versie `v0.36.2`.
3. Kies `N398`.
4. Upload `wegassen_paspoort.geojson`.
5. Klik op `🚦 Maak projectadvies`.
6. Download:
   - `Projectadvies_Runrapport_N398.csv`;
   - `Projectadvies_Voorstellen_N398.csv`;
   - `Projectadvies_Werklijst_N398.csv`.

Verwacht:

- het runrapport bevat een run-oordeel;
- het rapport benoemt de vervolgstap;
- de werklijst blijft de actielijst;
- de voorstellenlijst blijft de onderbouwing.

### Test 2 — geen handmatige regelcontrole als primaire test

Vanaf v0.36.2 is het niet meer de bedoeling dat de gebruiker specifieke regels
moet opzoeken om te bewijzen dat de release werkt. De primaire test is:

```text
Draai de analyse.
Download de drie Projectadvies-exports.
Beoordeel het runrapport en vergelijk de tellers met de werklijst.
```

## Volgende logische stap

Als N398 met v0.36.2 goed leest, is Project Adviseur voor deze gecontroleerde
case voldoende werkbaar om door te gaan naar:

```text
v0.36.3 / v0.37.0 — N354 praktijktoets
```

N354 moet aantonen of dezelfde werkwijze ook bruikbaar blijft bij een complexe,
verouderde en ruimtelijk lastige weg.
