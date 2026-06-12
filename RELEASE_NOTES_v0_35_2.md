# Release notes v0.35.2

## Thema

Interactieve kaartinspectie voor groenveld-projectvoorstellen.

## Wat is toegevoegd

- In `🧪 NWB referentieproef` is een inspectieblok toegevoegd voor projectvoorstellen.
- Een gebruiker kan filteren op projecttype en status.
- Een gebruiker kan één projectvoorstel selecteren.
- De hoofdkaart licht de objecten van het geselecteerde voorstel paars uit.
- Objecten uit bestaande iASSET-onderhoudsprojecten die volgens de vergelijking raken aan het voorstel, maar niet in het voorstel zitten, worden blauw getoond.
- Het inspectieblok toont de belangrijkste voorstelgegevens, knipredenen, harde/zachte signalen en objecttoewijzing.
- Er is een zoomknop toegevoegd om de hoofdkaart op het geselecteerde voorstel te zetten.

## Belangrijk

v0.35.2 verandert de projectvoorstellenlogica van v0.35.1 niet. Deze versie is bewust een visuele inspectielaag bovenop de bestaande diagnose.

De app wijzigt niets in iASSET, maakt geen beheerknippen en schrijft geen onderhoudsprojecten terug.

## Techniek

- `build_road_map()` accepteert optioneel object-id's voor projectvoorstel-highlight.
- Object-id's worden genormaliseerd, zodat `1`, `1.0` en `"1"` robuust vergeleken worden.
- De kaart krijgt alleen een legenda voor projectvoorstel-inspectie wanneer er een selectie actief is.
- Extra UI-helperfuncties in `app.py` halen object-id's, bestaande iASSET-contextprojecten en kaartbounds veilig op.

## Validatie

- `python -m py_compile app.py iasset_tool/map_view.py iasset_tool/project_axis.py`
- `184 passed`
