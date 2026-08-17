# Release notes v0.37.0

## Doel

v0.37.0 start de inhoudelijke kalibratiefase voor N354. De versie past de
projectvormingsengine nog niet aan, maar voegt een automatisch kalibratierapport
toe waarmee duidelijk wordt waar de huidige onderhoudscomplexlogica waarschijnlijk
te fijn knipt.

## Nieuw

- Nieuwe Excel-export: `Projectadvies_Kalibratie_<weg>.xlsx`.
- Nieuw tabblad `Samenvatting` met kalibratie-oordeel en vervolgstappen.
- Nieuw tabblad `Kandidaat_samenvoegen` met geprioriteerde kalibratiepunten.
- Nieuw tabblad `Korte_reguliere_voorstellen`.
- Nieuw tabblad `Dubbele_projectnamen`.
- Nieuw tabblad `Objectfamilie_mismatch`.
- Nieuw tabblad `Knipreden_analyse`.
- Nieuw tabblad `Hectometerinterval_context`.

## Bewust niet gewijzigd

- Geen nieuwe samenvoegregel.
- Geen nieuwe knipregel.
- Geen automatische vergelijking met oude handmatige N-wegentabbladen.
- Geen wijziging in de bestaande Project Adviseur-voorstellenlijst, werklijst,
  runrapport of concept-N-wegendocument.

## Waarom

N354 levert veel projectvoorstellen op. Direct aan de engine draaien is riskant:
dan kunnen we echte onderhoudsverschillen wegpoetsen of N398 beschadigen. Het
kalibratierapport maakt eerst inzichtelijk welke soort problemen dominant zijn,
zodat v0.37.1 gericht een eerste inhoudelijke regelwijziging kan doen.

## Validatie

- `pytest -q`
- `python -m py_compile` op alle Python-bestanden
- proefrapport op basis van N354-exports
