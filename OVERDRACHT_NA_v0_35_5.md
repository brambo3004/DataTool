# Overdracht na v0.35.5

## Stand van zaken

v0.35.5 sluit de huidige referentieas-/greenfield-diagnosefase grotendeels af.
De tool heeft nu:

- iASSET-wegas als leidende referentieas;
- NWB/PDOK-hectopunten als ijking;
- snap-tolerantie van 2,5 m;
- hectometer-naar-boven-regel voor onderhoudsprojectnamen;
- greenfield-projectvoorstellen op basis van primaire objecten;
- reeksherkenning en lokale afwijkingen;
- interactieve kaartinspectie voor voorstellen;
- grensdiagnose en hectometerintervalcontrole.

## Belangrijke toevoeging

De tool maakt nu standaard een intervaldiagnose van opeenvolgende hectometerpunten:

`Hectometerintervallen_Referentieas_<weg>.csv`

Deze tabel verklaart of een interval fysiek ongeveer de verwachte lengte heeft.
Afwijkingen worden gekoppeld aan projectvoorstelgrenzen.

## Beheerinterpretatie

- Afwijkend interval = betrouwbaarheidssignaal, geen automatische fout.
- Normaal interval maar grens ver van hm-punt = waarschijnlijk objectgrens- of
  projectievraagstuk.
- Einde referentieas = controleer; niet stilzwijgend als normaal kort project
  behandelen.

## Volgende fase

Na acceptatie van v0.35.5 is de logische volgende stap een nieuw gesprek / nieuwe
ontwikkelfase:

`v0.36.0 — Project Adviseur 2.0`

Doel v0.36:
- niet meer verder stapelen in het technische referentieas-tabblad;
- de v0.35-engine functioneel presenteren in Project Adviseur;
- compacte werklijst, kaartinspectie en projectvoorstellen samenbrengen;
- debugdiagnose beperken of verbergen waar mogelijk.

## Testadvies

Test v0.35.5 eerst op N398:

- `Projectvoorstellen_Referentieas_N398.csv`
- `Projectvoorstel_Objecten_N398.csv`
- `Projectvoorstel_Vergelijking_iASSET_N398.csv`
- `NWB_Hectopunten_op_iASSET_Wegas_N398.csv`
- `Hectometerintervallen_Referentieas_N398.csv`

Controleer vooral:
- hm 1.1: normaal interval maar objectgrens/projectieconflict;
- hm 6.2-6.3: afwijkend interval dat de waarde rond 6.273 verklaart.
