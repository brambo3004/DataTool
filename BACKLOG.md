# Takenlijst iASSET Advisor

Deze lijst bewaart de functionele en technische verbeterpunten die we nog niet
volledig hebben uitgewerkt.

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

1. **v0.5 lokaal valideren**
   - Controleer of Overzicht alleen rijstroken toont.
   - Controleer `Jaar deklaag`, `Jaar aanleg`, `Besteknummer`, `Onderhoudsproject`.
   - Controleer of `Soort verharding_N` werkt bij exports met `verhardingssoort`.
   - Controleer popup/tooltip op echte data.
   - Controleer prestaties bij langere wegen zoals N359.

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
