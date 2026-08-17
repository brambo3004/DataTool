# Overdracht na v0.36.3

## Status

Project Adviseur heeft nu naast runrapport, voorstellenlijst en werklijst ook
een concept-export in het format van het N-wegendocument.

Dit is een belangrijke stap richting dagelijks gebruik: de output is niet alleen
een diagnose of actielijst, maar ook een werkblad dat de databeheerder naast het
bestaande handmatige N-wegendocument kan leggen.

## Belangrijkste toevoeging

Nieuwe export:

```text
Projectadvies_Nwegendocument_<weg>.xlsx
```

De export bevat concepttabbladen per wegdeeltype. Voor N354 worden bijvoorbeeld
tabbladen gemaakt zoals:

```text
N354 (HRB)
N354 (PW)
N354 (FP)
```

Het bestaande N-wegendocument wordt niet aangepast of overschreven.

## Wat dit oplost

Voor N354 bleek de werklijst groot. Dat betekent niet dat de tool faalt, maar
wel dat een losse actielijst onvoldoende aansluit op het echte werkproces. Het
N-wegendocument is de werklaag waarin onderhoudscomplexen worden gereconstrueerd.
v0.36.3 vertaalt de Project Adviseur-uitkomst daarom naar dat format.

## Wat automatisch gevuld wordt

- onderhoudscomplex oud;
- onderhoudscomplex nieuw;
- knip begin/einde;
- verharding begin/einde;
- besteknummer, indien uit objecttoewijzing beschikbaar;
- verhardingsoort;
- jaar aanleg/deklaag/conservering/herstrating;
- bijzonderheden vanuit werkadvies, iASSET-vergelijking, categorie en datakwaliteit.

## Wat bewust niet automatisch definitief wordt

- locatie;
- documentatie;
- herkenbare objectcontext zoals brug/kruising/rotonde;
- besluit om een voorstel over te nemen in iASSET;
- besluit om het handmatige N-wegendocument bij te werken.

## Testaanpak vanaf nu

De gebruiker hoeft niet in specifieke tabelregels te zoeken. Testen gebeurt via
exports:

1. Draai Project Adviseur voor de weg.
2. Download:
   - `Projectadvies_Runrapport_<weg>.csv`
   - `Projectadvies_Voorstellen_<weg>.csv`
   - `Projectadvies_Werklijst_<weg>.csv`
   - `Projectadvies_Nwegendocument_<weg>.xlsx`
3. Beoordeel automatisch of:
   - het runrapport bruikbaar is;
   - het concept-N-wegendocument de juiste tabbladen heeft;
   - HRB/PW/FP logisch zijn gevuld;
   - micro/eindzone en buiten-ijkbereik in bijzonderheden herkenbaar zijn.

## Nog niet af

- De export is een concept, geen definitieve wijziging.
- Locatie en documentatie worden nog niet automatisch betrouwbaar gevuld.
- Voor N354 moet de inhoudelijke bruikbaarheid van het concepttabblad nog worden beoordeeld.
- Een latere versie kan een netter `Projectadvies_<weg>.xlsx` maken met alle
  tabbladen in één rapport.
