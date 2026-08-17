# Overdracht na v0.36.7

## Status

v0.36.7 rondt de v0.36-fase van Project Adviseur 2.0 af. De module is nu een
werkbare eerste end-to-end versie voor:

- projectvoorstellen maken;
- voorstellen vergelijken met bestaande iASSET-indeling;
- werklijst maken;
- automatisch runrapport maken;
- concept-N-wegendocument exporteren;
- volledige objecttoewijzing apart bewaren.

## Belangrijkste correctie in v0.36.7

De runrapporttekst is aangescherpt. De tool verwijst bij `Verschillen met
iASSET` niet meer naar het N-wegendocument als beoordelingsbron.

De nieuwe lijn is:

```text
Beoordeel verschillen met kaartbeeld, actuele iASSET-data en beschikbare
broninformatie. Gebruik het concept-N-wegendocument als werkblad, niet als
waarheid.
```

## Beslissing die vastligt

Het bestaande handmatige N-wegendocument wordt gebruikt als:

- formatvoorbeeld;
- werkprocesvoorbeeld;
- exportstructuur.

Het wordt niet gebruikt als:

- waarheidsbron;
- kalibratiebron;
- automatische vergelijkingsnorm;
- reden om toolvoorstellen goed of fout te noemen.

## Status per testweg

### N398

N398 is werkbaar als gecontroleerde referentiecase. De Project Adviseur maakt
een bruikbaar runrapport, voorstellenlijst, werklijst en conceptwerkblad.

### N354

N354 draait technisch en levert alle werkbestanden op. De uitkomst is nog sterk
versnipperd en vraagt een volgende inhoudelijke kalibratiefase. Die fase moet
niet gaan over vergelijking met oude handmatige tabbladen als waarheid, maar over
de onderliggende onderhoudscomplexlogica:

- wanneer splitst de tool terecht;
- wanneer splitst de tool te veel;
- wanneer moeten korte trajecten worden samengevoegd;
- hoe worden kruispunten, rotondes, parallelwegen, busbanen en secundaire
  objecten meegenomen;
- hoe worden lokale datakwaliteitsafwijkingen onderscheiden van echte
  onderhoudsgrenzen.

## Logische volgende fase

```text
v0.37.0 — N354 inhoudelijke kalibratie onderhoudscomplexen
```

Doel:

- N354 gebruiken als moeilijke praktijkcase;
- verschillen inhoudelijk verklaren;
- samenvoeg- en kniplogica verbeteren waar nodig;
- voorkomen dat de tool wordt afgestemd op verouderd handmatig werk;
- principes verbeteren die ook voor andere wegen gelden.

## Testadvies na v0.36.7

Draai N354 en download:

```text
Projectadvies_Runrapport_N354.csv
Projectadvies_Voorstellen_N354.csv
Projectadvies_Werklijst_N354.csv
Projectadvies_Nwegendocument_N354.xlsx
```

Controleer automatisch of:

- de runrapporttekst geen oude waarheidsbron suggereert;
- de N-wegendocument-export een werkblad blijft;
- de objecttoewijzing apart beschikbaar blijft;
- de uitkomst gelijk blijft aan v0.36.6 behalve de tekstcorrectie.
