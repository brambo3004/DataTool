# Takenlijst iASSET Advisor

Deze lijst bewaart de functionele en technische verbeterpunten die we nog niet
volledig hebben uitgewerkt.

## Eerstvolgende punten

1. **Project Adviseur: controle op secundaire toewijzing in echte N398/N359-data**
   - Met name situaties waarin bermen, goten, inritten of vluchtheuvels grenzen aan meerdere primaire objecten.
   - Controleren met netwerklaag aan.

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
- Meergebruikersopslag onderzoeken: SQLite of PostgreSQL/PostGIS.
