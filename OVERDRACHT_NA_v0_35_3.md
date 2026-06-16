# Overdracht na v0.35.3

## Status

v0.35.3 is gebouwd als inhoudelijke verbetering van de greenfield-projectvoorstellen. De interactieve kaartinspectie uit v0.35.2 blijft aanwezig; de rekenwijziging zit in de projectvoorstel-engine.

## Beheerregel die nu in code zit

Lokale afwijking:

```text
maximaal 2 objecten
en korter dan 100 meter
en links/rechts hetzelfde technische profiel
```

Uitkomst:
- geen onderhoudsprojectknip;
- ontbrekende waarden worden datakwaliteitsmelding;
- echte technische afwijkingen worden `controleer` binnen hetzelfde voorstel.

## Belangrijke ontwerpkeuzes

- Bestaande onderhoudsprojecten zijn vergelijkingsmateriaal, geen basis voor voorstellen.
- `Besteknummer` is ondersteunend, niet zelfstandig leidend.
- `verhardingssoort` wordt niet gebruikt voor kniplogica.
- Structurele wijzigingen in technische paspoortvelden blijven knipkandidaat.

## Testadvies

Test eerst opnieuw N398:
- controleer of het traject met twee objecten zonder besteknummer niet meer als projectgrens wordt behandeld;
- bekijk in de kaartinspectie of `object_kniprol` en de signalen logisch zijn;
- controleer of echte verhardings-/jaarwissels nog wel als grens zichtbaar worden.

Daarna N354:
- let vooral op daling van onlogische korte voorstellen;
- noteer voorbeelden waar de reeksherkenning nog te veel samenvoegt of te veel splitst.

