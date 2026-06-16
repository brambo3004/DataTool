# Overdracht na v0.35.4

## Stand van zaken

v0.35.4 is een diagnose-release. De greenfield-kniplogica van v0.35.3
(reeksherkenning, lokale afwijkingen, besteknummer ondersteunend) is niet
inhoudelijk gewijzigd.

## Toegevoegd

De tool legt nu per projectvoorstelgrens vast in welk geijkt hectometerinterval
de grens valt. Daarbij worden de fysieke intervallengte en de administratief
verwachte lengte vergeleken.

Dit verklaart casussen zoals het einde van de N398:

- hm 6.2-6.3 is fysiek korter dan 100 m;
- een grens circa 53 m na hm 6.2 kan daardoor geijkt worden als circa 6.273;
- de export toont dit nu expliciet in `eind_grensdiagnose`.

## Belangrijke interpretatie

Een afwijkend hectometerinterval is niet automatisch een fout. Het is een
uitlegsignaal voor databeheercontrole. De tool blijft diagnose/voorstel doen en
muteert iASSET niet.

## Volgende aandachtspunten

1. Test N398 opnieuw en controleer vooral:
   - traject rond hm 1.1;
   - eindgebied rond hm 6.2-6.3;
   - voorstellen met `grensdiagnose`.
2. Analyseer of het 1.1-conflict door vlakobjectprojectie komt.
3. Overweeg in een latere versie kaartweergave van hectometerpunten bij het
   geselecteerde voorstel.
4. Blijf Project Adviseur en referentieas-tabblad functioneel consolideren om
   wildgroei te voorkomen.
