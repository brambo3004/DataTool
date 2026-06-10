# Overdracht na v0.32.1 — robuustere referentieasdiagnose

## Aanleiding

De N354-exportsets voor 10m, 25m en 50m lieten zien dat de afstandsdrempel technisch goed werkte, maar dat sommige objecten naar een verkeerd of extreem deel van de opgebouwde referentieas konden projecteren.

Belangrijke les:
afstand tot de as alleen is onvoldoende. Een object kan dicht bij de as liggen en toch een onwaarschijnlijke referentiesprong veroorzaken.

## Wat is aangepast

### Objectdiagnose

Nieuwe kolommen:

- `binnen_afstandsdrempel`
- `buiten_projectrange`
- `projectiesprong`
- `bruikbaar_voor_projectsamenvatting`

Nieuwe waarschuwingen:

- `referentie buiten projectrange`
- `onwaarschijnlijke referentiesprong`

### Projectsamenvatting

De projectbegin/eindwaarden worden niet meer bepaald op basis van alle objecten met projectie. Alleen objecten met `bruikbaar_voor_projectsamenvatting = True` tellen mee.

Hierdoor blijven foute objecten zichtbaar, maar domineren ze niet meer het projectresultaat.

Toegevoegde telkolommen:

- `objecten_binnen_drempel`
- `objecten_buiten_projectrange`
- `objecten_met_projectiesprong`
- `objecten_bruikbaar_voor_projectsamenvatting`

## Niet aangepast

- Overzicht
- Onderhoudscontrole
- Project Adviseur
- v0.31-trajectlengtekeuze
- iASSET-mutaties

## Testresultaat

```text
151 passed
```

## Advies voor vervolg

Draai dezelfde N354-test opnieuw met 10m/25m/50m. Let vooral op:

1. of de eerder ontspoorde projecten nu waarschuwingen krijgen;
2. of de projectsamenvatting niet meer extreem lange referentielengtes toont;
3. of de objectdiagnose de verdachte objecten nog steeds zichtbaar houdt.

Daarna dezelfde proef uitvoeren voor N398.
