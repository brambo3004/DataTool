# Overdracht na v0.36.6

## Status

v0.36.6 is een kleine maar inhoudelijk belangrijke correctie op de
N-wegendocument-export. De Project Adviseur-engine, werklijst en runrapport zijn
niet inhoudelijk gewijzigd.

## Wat is afgerond

De concept-export in N-wegendocument-format gebruikt het N-wegendocument nu als
werkvorm, niet als waarheidsbron.

Belangrijkste correcties:

- de zichtbare kolom `objecten` wordt niet meer gevuld met alle objecten;
- alleen bijzondere objecten worden daar voorzichtig geplaatst;
- als bijzondere objecten niet betrouwbaar herkenbaar zijn, blijft de cel leeg;
- volledige objectcontext wordt apart bewaard in `Objecttoewijzing_data`;
- WKT-/geometriekolommen worden uit dat ondersteunende tabblad geweerd.

## Waarom dit belangrijk is

De tool moet actuele iASSET-data gebruiken om onderhoudscomplexen zo veel
mogelijk automatisch te reconstrueren. Oude handmatige tabbladen kunnen nuttig
zijn als context, maar zijn niet automatisch de waarheid. Daarom mag de tool niet
gaan sturen op overeenkomst met oude handmatige lijsten.

## Huidige stand

- N398 is werkbaar als gecontroleerde referentiecase.
- N354 draait technisch en levert runrapport, voorstellenlijst, werklijst en
  concept-N-wegendocument op.
- De concept-export is nu schoner als werkexport.
- De inhoudelijke onderhoudscomplexlogica voor N354 is waarschijnlijk de volgende
  grotere stap.

## Nog niet opgelost

N354 blijft sterk versnipperd:

- veel voorstellen;
- veel werklijstregels;
- veel verschillen met bestaande iASSET-indeling;
- veel micro-/eindzonevoorstellen.

Dat moet niet opgelost worden door oude handmatige tabbladen als waarheid te
nemen, maar door inhoudelijk te onderzoeken waarom de tool splitst en welke
samenvoeg- of beheerregels ontbreken.

## Advies voor volgende fase

Na controle van v0.36.6 is de logische vervolgfase:

```text
v0.37.0 — N354 inhoudelijke kalibratie onderhoudscomplexen
```

Mogelijke onderzoeksvragen:

- splitst de tool te veel door korte technische profielwisselingen?
- moeten kruispunten/rotondes als eigen onderhoudscomplex of als onderdeel van
  aangrenzende complexen worden behandeld?
- hoe moeten korte micro-/eindzones worden samengevoegd of apart gemarkeerd?
- wanneer hoort secundaire verharding bij HRB, PW of FP?
- welke regels uit het handmatige werkproces moeten explicieter in de engine
  worden vastgelegd?

## Testadvies

Draai N354 met v0.36.6 en upload:

```text
Projectadvies_Runrapport_N354.csv
Projectadvies_Voorstellen_N354.csv
Projectadvies_Werklijst_N354.csv
Projectadvies_Nwegendocument_N354.xlsx
```

Controleer primair automatisch:

- tabbladstructuur aanwezig;
- objecten-kolom bevat geen generieke objectlijsten;
- `Objecttoewijzing_data` is aanwezig wanneer objecttoewijzing beschikbaar is;
- runrapport en werklijst blijven gelijk aan v0.36.5, behalve app-versie.
