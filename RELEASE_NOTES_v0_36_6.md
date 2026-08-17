# Release notes v0.36.6

## Doel

v0.36.6 schoont de concept-export in N-wegendocument-format op. De wijziging is
gericht op de betekenis van de zichtbare werkbladkolommen, niet op nieuwe
projectas- of onderhoudscomplexlogica.

## Belangrijkste wijziging

De kolom `objecten` wordt niet meer gevuld met alle objecten binnen een
projectvoorstel. In het N-wegendocument is deze kolom bedoeld voor bijzondere
objecten binnen een onderhoudscomplex, zoals:

- rotonde;
- kruispunt;
- aansluiting;
- brug;
- tunnel;
- viaduct;
- aquaduct;
- spoorwegovergang.

Als zulke objecten niet betrouwbaar uit de paspoortvelden kunnen worden
herkend, blijft de cel leeg.

## Nieuwe ondersteunende data

Wanneer objecttoewijzing beschikbaar is, bevat de Excel-export nu ook:

```text
Objecttoewijzing_data
```

Dit tabblad bewaart de objectcontext die niet in de zichtbare kolom `objecten`
thuishoort. Geometrie-/WKT-kolommen worden bewust niet meegenomen, zodat de
Excel-export werkbaar blijft. Voor volledige technische GIS-diagnose blijven de
bestaande technische exports beschikbaar.

## Geen vergelijking met oude handmatige tabbladen

Het N-wegendocument wordt in deze versie gebruikt als format- en
werkprocesvoorbeeld. Oude handmatige tabbladen worden niet als waarheid gebruikt
voor automatische beoordeling.

## Gewijzigde bestanden

```text
README.md
BACKLOG.md
iasset_tool/__init__.py
iasset_tool/config.py
iasset_tool/nwegendocument_export.py
tests/test_project_advisor_v2.py
```

## Nieuw

```text
RELEASE_NOTES_v0_36_6.md
OVERDRACHT_NA_v0_36_6.md
```

## Verwijderd

```text
geen
```

## Validatie

```text
pytest -q
201 passed
```

Daarnaast is `python -m py_compile` succesvol uitgevoerd op alle Python-bestanden.

## Verwachte test

Draai N354 opnieuw in Project Adviseur en download:

```text
Projectadvies_Runrapport_N354.csv
Projectadvies_Voorstellen_N354.csv
Projectadvies_Werklijst_N354.csv
Projectadvies_Nwegendocument_N354.xlsx
```

De zichtbare N-wegendocument-tabbladen mogen geen volledige objectlijsten meer in
de kolom `objecten` tonen. Bijzondere objecten mogen daar wel staan. Volledige
objectcontext hoort in `Objecttoewijzing_data` of in de bestaande technische
objectexport.
