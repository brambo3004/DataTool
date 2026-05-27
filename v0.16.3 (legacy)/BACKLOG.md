# Takenlijst iASSET Advisor

## Klaar in v0.16.3

1. **Fase-4-actielijst geschikt gemaakt als afhandelwerkvoorraad**
   - Nieuwe kolom `praktische_categorie` toegevoegd aan de actielijst.
   - Nieuwe opvolgkolommen toegevoegd: `beoordeling_databeheerder`, `afhandelstatus`, `actiehouder` en `opmerking_afhandeling`.
   - Nieuwe actieregels krijgen standaard `afhandelstatus = nieuw`.
   - De extra kolommen maken het mogelijk om controlepunten direct vanuit de export inhoudelijk te beoordelen en op te volgen.
   - Testdekking toegevoegd voor praktische categorieën en afhandelkolommen.

## Klaar in v0.16.2

1. **Fase-4-actielijst toegevoegd**
   - Nieuwe export: `Fase4_Actielijst_[weg].csv`.
   - Technische Fase-4-statussen worden vertaald naar controlewerk voor de databeheerder.
   - De actielijst bevat per controlepunt: categorie, betrokken objecten, uitleg, mogelijke oorzaak en voorgestelde actie.
   - De bestaande controle-, samenvattings- en objectverschillenexports blijven beschikbaar.
   - Testdekking toegevoegd voor ontbrekende projecten, verdachte objectnummers en ongeldige metrering in de actielijst.

## Klaar in v0.16.1

1. **Fase-4-controle verdiept naar objectinhoudcontrole**
   - Projecten worden niet meer alleen op projectnaam gecontroleerd, maar ook op de objectsets per onderhoudsproject.
   - Nieuwe objectverschillenexport: `Fase4_Objectverschillen_[weg].csv`.
   - Statussen aangescherpt naar onder andere `OK_VOLLEDIG`, `OBJECTVERSCHIL`, `OBJECT_WEGNUMMER_VERDACHT` en `HM_BEREIK_VERDACHT`.
   - Objecten die alleen in de paspoortexport of alleen in de onderhoudsexport staan worden expliciet gemeld.
   - Objectnummers in de onderhoudsexport die bij een ander wegnummer lijken te horen worden als waarschuwing gemeld.
   - Ongeldige meteringen worden genegeerd in `hm_min`/`hm_max` en apart gerapporteerd.
   - Testdekking toegevoegd voor objectverschillen, verkeerd wegnummer in onderhoudsobjecten en ongeldige metrering.

## Klaar in v0.16.0

1. **Eerste Fase-4-controle paspoortexport versus onderhoudsexport**
   - Nieuw UI-onderdeel **Fase 4 Controle** toegevoegd.
   - Onderhoudsexports worden apart ingelezen, zonder geometrieverplichting.
   - Paspoortobjecten worden per onderhoudsproject samengevat.
   - Onderhoudsregels worden per projectnaam samengevat.
   - De vergelijking markeert projecten als `OK`, `ONTBREEKT_IN_ONDERHOUD` of `GEEN_PASPOORTOBJECTEN`.
   - Controletabellen zijn als CSV te downloaden voor overleg vóór verwerking in iASSET/maatregeltoets.
   - N398-smoketest: 11 projecten gecontroleerd, 11 OK, 0 waarschuwingen.

## Klaar in v0.15.1

1. **Robuustere Project Adviseur-sleutel bij primaire route-outliers**
   - Compacte hm-groepen met een extreem vroeg `route_start_m` gebruiken in overlapclusters `route_mid_m` als gecorrigeerde sorteersleutel.
   - De diagnose toont `advisor_sort_raw_m`, `advisor_sort_m` en `advisor_sort_correctie`, zodat de correctie controleerbaar blijft.
   - `GRP_RIJBAAN_40`-achtige situaties blijven een route-outlier-waarschuwing houden, maar trekken de projectvolgorde niet meer kilometers terug.
   - Primaire objecten met een extreem grote `object_route_span_m` krijgen nu een objectwaarschuwing.
   - Testdekking toegevoegd voor de N354-achtige route-startuitschieter en voor object-route-outliers.

## Klaar in v0.15.0

1. **Projectvolgorde baseren op primaire ruggengraat**
   - Project Adviseur gebruikt een expliciete `advisor_sort_m`-sleutel.
   - `advisor_sort_basis` maakt zichtbaar of de volgorde op primaire routepositie of fallback is gebaseerd.
   - Secundaire objecten blijven gekoppelde inhoud van de groep, maar bepalen de projectvolgorde niet.
   - Sorteerdiagnose toont `advisor_sort_terugval_vorige` naast de bestaande route-terugvalvelden.
   - Testdekking toegevoegd voor primaire-ruggengraatvolgorde met een secundaire route-uitschieter.

## Klaar in v0.14.2

1. **Diagnosefix routewaarden en primaire ruggengraat**
   - `route_mid_terugval_vorige` en `route_sort_terugval_vorige` zijn gescheiden.
   - De oude kolom `route_terugval_vorige` blijft bestaan, maar volgt nu de feitelijke sorteersleutel `route_sort_m`.
   - Groepsdiagnose toont `primary_route_*` en `all_route_*` naast elkaar.
   - `route_basis` maakt zichtbaar of de sortering op primaire objecten of op alle objecten terugvalt.
   - `route_outlier_warning` markeert compacte hm-groepen met grote routeverschillen, zoals een verdacht start-/mid-/eindverschil.
   - Diagnoseberichten zijn versie-onafhankelijk gemaakt.
   - Testdekking uitgebreid voor route-mid versus route-sort, outliers, primaire/all-routeverschillen en versie-onafhankelijke waarschuwingen.

## Gepland na v0.16.3

1. **Fase-4-controle verdiepen**
   - Naamvarianten slimmer herkennen, bijvoorbeeld `01.2` versus `1.2`.
   - Controlepakket uitbreiden met Excel/HTML-export voor overleg.
   - Later pas: gecontroleerde mutatievoorstellen richting iASSET.

## Klaar in v0.13.1

1. **Uitlegbaarheidsfix route-as-sortering**
   - Groepsdiagnose toont nu expliciet `route_sort_m`, `route_sort_bron`, `route_sort_verklaarbaar`, `fallback_sort_m` en `hm_route_conflict`.
   - De algemene `tie_breaker` is niet langer de enige aanwijzing voor de sortering; routewaarde en fallbackwaarde staan apart in de diagnose.
   - Overlapclusters waarbij meerdere groepen dezelfde lokale routepositie krijgen, tonen nu `stabiele_fallback`.
   - Eindpunt- en rotondeachtige situaties worden daardoor niet meer onterecht gepresenteerd alsof de lokale route-as de onderlinge volgorde volledig verklaart.

## Klaar in v0.13

1. **Overlapclusters sorteren op lokale route-as**
   - Bij overlappende hm-bereiken bepaalt de lokale routepositie de volgorde binnen het cluster.
   - Metrering blijft leidend buiten overlapclusters.
   - De globale richtingentabel blijft fallback.

## Klaar in v0.11

1. **Sorteerdiagnose als voorbereiding op betere projectvolgorde**
   - De Project Adviseur-sortering is nog niet inhoudelijk aangepast.
   - Onder de gewone kaart kan een sorteerdiagnose op verzoek worden berekend.
   - De diagnose bouwt een lokale route-as uit primaire objecten per wegvak/metrering-bucket.
   - Groepsdiagnose toont `sort_quality`, hectometerbereik, lokale routepositie en waarschuwingen.
   - Objectdiagnose toont onder andere dubbele objecten binnen hetzelfde wegvak/metrering/situering.
   - Groeps- en objectdiagnose zijn als CSV te downloaden.
   - De diagnose is bedoeld om de latere route-as-sortering veilig te ontwerpen en te testen.

## Klaar in v0.10.1

1. **Kaartcrash door niet-JSON-veilige paspoortwaarden opgelost**
   - De gewone kaartlaag zet tooltip-/kaartattributen nu om naar JSON-veilige waarden voordat Folium de GeoJSON-laag bouwt.
   - Pandas-waarden zoals `Timestamp`, `NaT`, `NA` en numpy/pandas-getaltypen laten de app niet meer crashen.
   - De originele iASSET-data wordt hierbij niet aangepast; alleen de lichte kaartkopie wordt opgeschoond.
   - Dit raakt de tabbladen Data Kwaliteit, Project Adviseur en Objectinspecteur. Het tabblad Overzicht gebruikte al een eigen kaartopbouw.

## Klaar in v0.10

1. **Objectinspecteur en individuele paspoortmutaties**
   - Nieuw tabblad Objectinspecteur toegevoegd.
   - Objecten kunnen worden gezocht op herkenbare paspoort- en liggingvelden.
   - Eén object kan worden geselecteerd, gemarkeerd en op de kaart ingezoomd.
   - De paspoortkern wordt als inspectietabel getoond.
   - Bewerkvelden worden bepaald op basis van het gekozen iASSET-exportprofiel.
   - Geometrie, bron-id's en afgeleide adviesvelden blijven niet-bewerkbaar.
   - Ontbrekende profielkolommen worden expliciet gemeld.
   - Wijzigingen worden celgewijs gelogd en blijven exporteerbaar via de bestaande exportprofielen.
   - Afgeleide kolommen voor subthema/rang/metrering worden bijgewerkt na mutatie.

## Klaar in v0.9.1

1. **Performance-hotfix naar aanleiding van lokale screenshots**
   - De performance-tabel toont nu de timing van de huidige Streamlit-run in plaats van opgetelde timings uit eerdere interacties.
   - De Project Adviseur gebruikt een multi-source BFS voor secundaire objecttoewijzing.
   - Fietspadclassificatie gebruikt een context met primaire objectindex, marker-cache en oriëntatie-cache.
   - Data Kwaliteit gebruikt snellere lookup-tabellen in de topologische controles.
   - Excel-import leest bij meerdere tabbladen eerst previews en daarna alleen het winnende tabblad volledig.

## Klaar in v0.9

1. **Performance-baseline en caching**
   - De zijbalk toont gemeten duur van zware stappen zoals data laden, netwerkopbouw, regelcheck, projectadvies, PDOK en kaartopbouw.
   - Data Kwaliteit gebruikt één gecachete regelcheck per weg/datasetrevisie.
   - PDOK-hectometerpunten worden standaard niet opgehaald en zijn sessie-gecached wanneer ze wel worden aangezet.
   - De gewone kaart stuurt minder attribuutkolommen naar Folium, zodat de GeoJSON-laag lichter is.

2. **Data Kwaliteit per issuecategorie**
   - Meldingen hebben nu `severity`, `category`, `rule_code` en `issue_key`.
   - De UI kan filteren op onder andere Onderhoudsprojectplicht, Onterecht onderhoudsproject, Topologie en Projectconsistentie.
   - Genegeerde meldingen worden voortaan per issue-key bijgehouden in plaats van alleen per object-id.

3. **Exportprofielen voor iASSET-import**
   - De export ondersteunt profielen voor Onderhoudsprojecten, Paspoortdata basis, Liggingdata en Volledige gecontroleerde mutatieset.
   - De UI toont hoeveel waarden worden meegeschreven en hoeveel daarvan niet daadwerkelijk gewijzigd zijn.
   - De app waarschuwt als gewijzigde velden niet in het gekozen exportprofiel zitten.


Deze lijst bewaart de functionele en technische verbeterpunten die we nog niet
volledig hebben uitgewerkt.

## Klaar in importfix na v0.8

1. **Import/upload robuuster**
   - CSV-import probeert meerdere encodings en scheidingstekens.
   - Kolomkopvarianten zoals `GPS Coördinaten`, `WEG NUMMER` en `Onderhoud project` worden herkend.
   - Excelkopregels mogen op een latere rij staan.
   - Lege thematabbladen winnen niet meer van een gevuld tabblad met dezelfde herkenningsscore.
   - `rds coordinaten` wordt gebruikt als geometrie-fallback als GPS-WKT ontbreekt of corrupt is.
   - Corrupte geometrieën worden gelogd en kunnen vanuit Datastatus als CSV worden gedownload.
   - Herladen van de actieve dataset wisselt niet meer ongemerkt naar een nieuw gekozen upload.


## Klaar in v0.8

1. **Databron uploaden**
   - iASSET-export kan direct in de app worden geüpload.
   - Upload ondersteunt CSV en Excel.
   - Meerdere bestanden kunnen tegelijk worden geüpload en samengevoegd.
   - Vaste bestanden naast `app.py` blijven beschikbaar als fallback.
   - Autosave is databron-afhankelijk gemaakt om oude wijzigingslogs niet op een andere export toe te passen.

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

1. **v0.8 lokaal valideren**
   - Start zonder upload en controleer of de vaste CSV's nog werken.
   - Upload één actuele CSV-export en controleer of de app herlaadt.
   - Upload twee deelbestanden en controleer of ze worden samengevoegd.
   - Upload een Excelbestand en controleer of het juiste tabblad wordt gebruikt.
   - Controleer of oude autosave-wijzigingen niet op een andere export worden toegepast.
   - Controleer Overzicht, Data Kwaliteit en Project Adviseur op de geüploade dataset.

2. **Kolomherkenning verder centraliseren**
   - Aliassen voor belangrijke iASSET-kolommen uitbreiden.
   - Inleesrapport verbeteren met aantallen per bronbestand en tabblad.
   - Duidelijker melden welke kolommen ontbreken.

3. **Autosave verder robuuster maken**
   - Opslag per gebruiker/weg/project/sessie.
   - Keuze toevoegen om oude autosave bewust te herstellen of te verwijderen.

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
