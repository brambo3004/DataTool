# Release notes v0.36.8

## Doel

v0.36.8 rondt de v0.36-fase af met één tekstcorrectie in de Excel-export
`Projectadvies_Nwegendocument_<weg>.xlsx`.

## Wijziging

In het tabblad `Samenvatting` is de waarschuwing aangepast.

Oude richting:

```text
Controleer altijd in kaart, iASSET en N-wegendocument voordat je gegevens overneemt.
```

Nieuwe richting:

```text
Controleer altijd met kaartbeeld, actuele iASSET-data en beschikbare broninformatie voordat je gegevens verwerkt.
Het concept-N-wegendocument is een werkblad, geen waarheid.
```

## Waarom

Het bestaande handmatige N-wegendocument is een werkdocument en kan per weg actueel,
verouderd of gedeeltelijk handmatig geïnterpreteerd zijn. De tool mag dit document
daarom niet als automatische waarheids- of beoordelingsbron presenteren.

## Geen wijzigingen aan

- projectlogica;
- werklijstselectie;
- runrapporttellingen;
- N-wegendocument-tabbladindeling;
- objecttoewijzing;
- rekenengine.

## Validatie

- `pytest -q`: 202 passed
- `python -m py_compile` op alle Python-bestanden
- aanvullende test voor de tekst in de Excel-samenvatting
