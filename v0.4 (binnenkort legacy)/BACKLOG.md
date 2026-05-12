# Takenlijst iASSET Advisor

Deze lijst bewaart de functionele en technische verbeterpunten die we nog niet
volledig hebben uitgewerkt.

## Klaar in v0.4

1. **Project Adviseur: eerste fietspadclassificatie**
   - Parallelfietspaden blijven een eigen onderhoudsprojectvoorstel.
   - Haakse/kruisende fietspaden worden als secundair object aan de hoofdgroep gekoppeld.
   - Rotonde-/kruispuntcontext wordt voorzichtig herkend via markers en geometrie.
   - Bij twijfel blijft het fietspad als eigen controlevoorstel zichtbaar.
   - Nog lokaal valideren op echte N398/N359-data.

## Eerstvolgende punten

1. **v0.4 valideren op echte N398/N359-data**
   - Controleer parallelfietspaden.
   - Controleer fietspaden bij rotondes.
   - Controleer fietspaden die haaks de hoofdrijbaan kruisen.
   - Controleer fietspaden bij complexe kruispunten en meerdere rijbanen.
   - Noteer false positives en false negatives, zodat we drempelwaarden kunnen aanscherpen.

2. **Overzicht-tabblad toevoegen**
   - Nieuw tabblad naast `Data Kwaliteit` en `Project Adviseur`.
   - Doel: alleen visualiseren, niet muteren.
   - Filter: alleen `subthema == rijstrook`.
   - Rechtsboven een keuzeveld `Visualiseer op`.
   - Linksonder een legenda.
   - Kaart toont rijstroken met kleuren per gekozen attribuut.
   - Voorlopige attributen:
     - `Jaar aanleg`
     - `Jaar deklaag`
     - `Jaar herstrating`
     - `Jaar conservering`
     - `Besteknummer`
     - `Onderhoudsproject`
     - `Wegvaknum`
     - `Soort verharding_N`
     - `Soort deklaag specifiek`
   - Voor numerieke waarden sorteren we de legenda oplopend.
   - Voor tekstwaarden sorteren we alfabetisch.
   - Popup/tooltip toont minimaal:
     - objectnaam of nummer
     - jaar aanleg
     - jaar deklaag
     - verhardingssoort
     - soort deklaag specifiek
     - besteknummer
     - onderhoudsproject
     - wegvaknummer

3. **Data-inleeslaag uitbreiden**
   - CSV én Excel ondersteunen.
   - Herkenning van kolomnamen centraliseren.
   - Inleesrapport verbeteren.

4. **Autosave robuuster maken**
   - Opslag per weg/project/sessie.
   - Geen globale `autosave_log.csv` meer voor alles.

5. **Kaartlegenda en inspectie verbeteren**
   - Legenda voor statuskleuren in Data Kwaliteit en Project Adviseur.
   - Debug-informatie voor groepstoewijzing leesbaarder maken.

## Later

- Naamvoorstellen voor onderhoudscomplexen automatisch genereren.
- PDOK-hectometerpunten niet alleen visualiseren, maar eventueel gebruiken als referentie.
- Wegas/hectometrering gebruiken voor robuustere sortering.
- Fietspadclassificatie verder verbeteren met echte wegas/hectometrering of betrouwbare wegvak-koppeling.
- Meergebruikersopslag onderzoeken: SQLite of PostgreSQL/PostGIS.
