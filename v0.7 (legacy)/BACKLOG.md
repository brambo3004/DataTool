# Takenlijst iASSET Advisor

Deze lijst bewaart de functionele en technische verbeterpunten die we nog niet
volledig hebben uitgewerkt.

## Klaar in v0.7

1. **Overzicht-tabblad: doorlopende kleurenschaal**
   - De legenda gebruikt geen willekeurige/cyclische categoriekleuren meer.
   - Waarden worden in gesorteerde volgorde op één kleurenspectrum geplaatst.
   - Lage/vroege waarden zijn blauw; hoge/recente waarden zijn rood.
   - `Onbekend` blijft grijs, zodat ontbrekende data herkenbaar blijft.
   - Dezelfde kleurmapping wordt gebruikt in de kaart én in de HTML-export.

## Klaar in v0.6

1. **Overzicht-tabblad: alle wegen**
   - Overzicht kan nu schakelen tussen `Geselecteerde weg` en `Alle wegen`.
   - Bij `Alle wegen` worden alle rijstrookobjecten uit de ingeladen dataset getoond.
   - Data Kwaliteit en Project Adviseur blijven per geselecteerde weg werken.

2. **Overzicht-tabblad: HTML-export**
   - De actuele Overzicht-instelling kan als interactieve HTML-kaart worden gedownload.
   - De export bevat kaart, legenda, tooltip/popup en de gekozen kleurvisualisatie.
   - Eerste exportvorm is HTML, omdat Folium/Leaflet dit natively ondersteunt.
   - PDF-export blijft op de lijst als aparte keuze: printkaart of screenshot van actuele browserweergave.

3. **Consolewaarschuwing datumparser opgelost**
   - Compacte iASSET-tijdstempels zoals `20260512095736` worden expliciet geparsed.
   - Daardoor wordt de pandas-waarschuwing over `dayfirst=True` voorkomen.

## Klaar in v0.5

1. **Overzicht-tabblad toegevoegd**
   - Nieuw tabblad naast `Data Kwaliteit` en `Project Adviseur`.
   - Alleen-lezen: geen mutaties en geen projectnaam-invoer.
   - Filtert op `subthema == rijstrook`.
   - Rechts in de werklijst staat `Visualiseer op`.
   - De kaart toont linksonder een legenda.
   - Ondersteunde attributen:
     - `Jaar aanleg`
     - `Jaar deklaag`
     - `Jaar herstrating`
     - `Jaar conservering`
     - `Besteknummer`
     - `Onderhoudsproject`
     - `Wegvaknum`
     - `Soort verharding_N`
     - `Soort deklaag specifiek`
   - `Soort verharding_N` valt terug op `verhardingssoort` als die kolom in de iASSET-export wordt gebruikt.
   - Numerieke legenda's worden oplopend gesorteerd; tekstuele legenda's alfabetisch.
   - Lege waarden worden als `Onbekend` getoond.

2. **Project Adviseur: eerste fietspadclassificatie**
   - Parallelfietspaden blijven een eigen onderhoudsprojectvoorstel.
   - Haakse/kruisende fietspaden worden als secundair object aan de hoofdgroep gekoppeld.
   - Rotonde-/kruispuntcontext wordt voorzichtig herkend via markers en geometrie.
   - Bij twijfel blijft het fietspad als eigen controlevoorstel zichtbaar.
   - Nog lokaal valideren op echte N398/N359-data.

## Eerstvolgende punten

1. **v0.7 lokaal valideren**
   - Controleer of Overzicht alleen rijstroken toont.
   - Controleer `Geselecteerde weg` versus `Alle wegen`.
   - Controleer `Jaar deklaag`, `Jaar aanleg`, `Besteknummer`, `Onderhoudsproject`.
   - Controleer of de legenda nu geleidelijk van blauw naar rood loopt.
   - Controleer of `Soort verharding_N` werkt bij exports met `verhardingssoort`.
   - Controleer popup/tooltip op echte data.
   - Download een HTML-export en open die buiten Streamlit.
   - Controleer prestaties bij alle wegen en bij langere wegen zoals N359.

2. **Data-inleeslaag uitbreiden**
   - CSV én Excel ondersteunen.
   - Herkenning van kolomnamen centraliseren.
   - Inleesrapport verbeteren.

3. **Autosave robuuster maken**
   - Opslag per weg/project/sessie.
   - Geen globale `autosave_log.csv` meer voor alles.

4. **Kaartlegenda en inspectie verbeteren**
   - Legenda voor statuskleuren in Data Kwaliteit en Project Adviseur.
   - Debug-informatie voor groepstoewijzing leesbaarder maken.

5. **Fietspadclassificatie verder valideren**
   - Controleer parallelfietspaden.
   - Controleer fietspaden bij rotondes.
   - Controleer fietspaden die haaks de hoofdrijbaan kruisen.
   - Controleer fietspaden bij complexe kruispunten en meerdere rijbanen.
   - Noteer false positives en false negatives, zodat we drempelwaarden kunnen aanscherpen.

## Later

- Naamvoorstellen voor onderhoudscomplexen automatisch genereren.
- PDOK-hectometerpunten niet alleen visualiseren, maar eventueel gebruiken als referentie.
- Wegas/hectometrering gebruiken voor robuustere sortering.
- Fietspadclassificatie verder verbeteren met echte wegas/hectometrering of betrouwbare wegvak-koppeling.
- Meergebruikersopslag onderzoeken: SQLite of PostgreSQL/PostGIS.

- PDF-export van Overzicht onderzoeken: statische printkaart of screenshot van actuele browserweergave.
