# Overdracht na v0.31 — voorbereiding v0.32 en verder

## Status na v0.31

v0.31 corrigeert de bronkeuze voor trajectlengtes in Overzicht. De tool behandelt losse objectpaspoort-metrering niet langer automatisch als beste bron, omdat deze in de praktijk vaak grof is afgerond. De tool maakt nu expliciet onderscheid tussen:

- voorkeurstrajectlengte;
- administratieve trajectlengte uit onderhoudsprojectnaam;
- objectmetrering uit paspoortdata;
- objectlengte als technische optelsom van rijstrookobjecten;
- oppervlakte.

De vijfmeterregel voor beheerknippunten is centraal vastgelegd in `iasset_tool/trajectory.py`, maar wordt nog niet gebruikt om automatisch nieuwe onderhoudsprojectnamen te bepalen.

## Belangrijkste ontwerpbesluit

De tool blijft adviserend:

- signaleren;
- uitleggen;
- visualiseren;
- voorstellen;
- exporteren.

De tool voert geen automatische mutaties uit in iASSET.

## Waarom v0.31 nodig was

In v0.30 werd `Metrering` uit de paspoortexport nog te vaak als voorkeursbron gezien. Testen op N354 en N398 lieten zien dat deze kolom niet altijd precies genoeg is voor trajectlengte. Daarom geldt vanaf v0.31:

1. Expliciete begin/eindmetrering is sterk.
2. Een range in een metreringveld is bruikbaar als beheer-range.
3. Een onderhoudsprojectnaam is een administratieve bron.
4. Losse objectpaspoort-metrering is grof en alleen fallback.
5. Objectlengte blijft apart zichtbaar.

## Voorstel v0.32

v0.32 zou moeten gaan over een experimentele referentieas/PDOK-proef.

Doel:

- nog niet productielogica;
- wel onderzoeken of een referentieas nauwkeuriger trajectmetrering kan leveren;
- eerst testen op N354 en N398;
- resultaat naast bestaande bronnen tonen, niet automatisch als waarheid gebruiken.

Mogelijke aanpak:

1. Mogelijkheid toevoegen om een referentielaag te uploaden of lokaal mee te leveren.
2. Referentielaag bevat bij voorkeur wegas of hectometerpunten.
3. Objectbegin/eindpunten projecteren op de referentieas.
4. As-metrering berekenen.
5. Vergelijken met:
   - onderhoudsprojectnaam;
   - objectpaspoort-metrering;
   - objectlengte;
   - handmatige controle.
6. Bronkwaliteit tonen als `experimenteel`.

## Vragen voor v0.32

Deze knopen moeten nog worden doorgehakt:

1. Gebruiken we een lokaal bestand, een handmatig aangeleverde laag of live PDOK?
2. Beginnen we met hectometerpunten, een wegas, of beide?
3. Welke wegen zijn de eerste testcasussen? Voorstel: N354 en N398.
4. Welke objectcategorieën doen eerst mee? Voorstel: alleen HRB/rijstrook.
5. Wat is de maximale afstand waarbinnen een objectpunt op de as geprojecteerd mag worden?
6. Hoe gaan we om met parallelwegen, fietspaden, rotondes en kruispunten? Voorstel: nog buiten scope van de eerste proef.

## Voorstel v0.33

Als v0.32 betrouwbare resultaten geeft:

- referentieas-meting een betrouwbaarheidsscore geven;
- alleen bij voldoende betrouwbaarheid als voorkeursbron tonen;
- anders als diagnose naast de bestaande bronnen laten staan.

## Voorstel v0.34/v0.35

Daarna pas Project Adviseur uitbreiden:

- begin/eind van nieuwe onderhoudscomplexen bepalen;
- vijfmeterregel gebruiken voor beheerknippunten;
- projectnaam afronden op tienden naar boven;
- conceptnaam tonen, bijvoorbeeld `N354-HRB-04.2-11.8`;
- nooit automatisch iASSET aanpassen.
