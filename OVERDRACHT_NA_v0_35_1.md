# Overdracht na v0.35.1

## Samenvatting

v0.35.1 verfijnt de groenveld-projectvoorstellen met knipzwaarte. De app probeert nog steeds onderhoudsprojecten vanaf nul te reconstrueren uit primaire objecten op de geijkte iASSET-referentieas, maar knipt minder snel op kleine detailverschillen.

## Waarom deze stap

De eerste v0.35.0-test liet zien dat de richting klopt, maar ook dat de tool te veel projectvoorstellen maakte:

- N398 werd als gecontroleerde referentiecase te veel gesplitst.
- N354 leverde veel korte segmenten op, waarvan een deel beter als detailcontext kan worden behandeld.

Daarom gebruikt v0.35.1 een voorzichtiger knipprofiel.

## Nieuwe logica

Een kenmerkwijziging wordt beoordeeld als:

- harde knip: voldoende sterk en over voldoende stabiele lengte;
- zacht signaal: zichtbaar in context, maar geen nieuw voorstel.

Voorbeelden van zachte signalen:

- alleen jaar aanleg wijzigt;
- lokaal kort verschil in deklaag;
- klein detailobject dat niet zelfstandig genoeg is voor een nieuw onderhoudsproject.

## Te testen

Gebruik opnieuw N398 en N354.

Upload na testen bij voorkeur:

- `Projectvoorstellen_Referentieas_N398.csv`
- `Projectvoorstel_Objecten_N398.csv`
- `Projectvoorstel_Vergelijking_iASSET_N398.csv`
- `Projectvoorstellen_Referentieas_N354.csv`
- `Projectvoorstel_Objecten_N354.csv`
- `Projectvoorstel_Vergelijking_iASSET_N354.csv`

Let vooral op:

- of N398 minder over-splitst dan v0.35.0;
- of N354 duidelijk minder dan 168 voorstellen geeft;
- of korte segmenten vaker als `zachte_signalen` verschijnen;
- of echte gaten/projectiewaarschuwingen zichtbaar blijven.

## Volgende stap

Als v0.35.1 inhoudelijk rustiger is, wordt v0.35.2 de interactieve kaartinspectie voor projectvoorstellen:

- voorstel selecteren;
- objecten van dat voorstel uitlichten;
- begin/eindgrenzen tonen;
- vergelijking met bestaande iASSET-situatie visueel maken.

## Niet doen

Nog geen automatische iASSET-mutaties, geen beheerknippen en geen automatische objectherverdeling.
