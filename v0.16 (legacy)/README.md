# iASSET Advisor - refactor v0.16.0

Deze versie splitst de bestaande Streamlit proof-of-concept op in een onderhoudbare projectstructuur.

## Doel van deze refactor

Eerst structuur, daarna bugs en nieuwe functionaliteit.

De huidige functionaliteit blijft zoveel mogelijk gelijk:
- iASSET CSV- en Excelbestanden inlezen via vaste bestandenmap óf upload;
- WKT-geometrie omzetten naar GeoDataFrame;
- wegnummer selecteren;
- ruimtelijk netwerk bouwen;
- datakwaliteitsmeldingen tonen;
- projectadviesgroepen maken;
- kaart tonen;
- wijzigingen loggen;
- exporteren naar CSV/Excel.

## Nieuwe structuur

```text
.
├── app.py
├── requirements.txt
├── iasset_tool/
│   ├── config.py
│   ├── domain.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── geometry.py
│   ├── rules.py
│   ├── fietspad.py
│   ├── advisor.py
│   ├── pdok.py
│   ├── performance.py
│   ├── map_view.py
│   ├── overview_map.py
│   ├── object_editor.py
│   ├── sorting_diagnostics.py
│   ├── maintenance_control.py
│   ├── changes.py
│   └── state.py
├── tests/
│   ├── test_advisor.py
│   ├── test_changes.py
│   ├── test_data_loader.py
│   ├── test_domain.py
│   ├── test_fietspad.py
│   ├── test_overview_map.py
│   ├── test_object_editor.py
│   ├── test_rules.py
│   ├── test_sorting_diagnostics.py
│   └── test_utils.py
└── legacy/
    └── app_v_11_5_2026.py
```

## Installatie

De app kan op twee manieren brondata lezen.

### Optie A: actuele export uploaden in de app

Start de app en gebruik in de zijbalk het onderdeel **Databron**. Upload daar één gecombineerd iASSET-exportbestand, of meerdere deelbestanden.

Ondersteund:

```text
.csv
.xlsx
.xls
.xlsm
```

### Optie B: vaste bronbestanden naast `app.py`

Laat de upload leeg als je de vaste bestandenmap wilt gebruiken. Dan verwacht de app voorlopig nog:

```text
N-allemaal-niet-rijstrook.csv
N-allemaal-alleen-rijstrook.csv
```

Installeer requirements:

```bash
pip install -r requirements.txt
```

Start de app:

```bash
streamlit run app.py
```


## Tests draaien

Optioneel voor ontwikkelaars:

```bash
pip install -r requirements-dev.txt
pytest
```





## Nieuw in v0.16.0

Deze versie voegt de eerste **Fase 4-controle** toe: paspoortexport en onderhoudsexport worden naast elkaar gelegd voordat er iets in iASSET of het maatregeltoetsdocument wordt verwerkt.

- Nieuw onderdeel **Fase 4 Controle** in de werklijst.
- Aparte upload voor onderhoudsexports, omdat deze bestanden meestal geen geometrie bevatten en dus niet door de gewone paspoort/geometrielader horen te gaan.
- De controle groepeert paspoortobjecten per `Onderhoudsproject` en onderhoudsregels per projectnaam.
- De vergelijking markeert:
  - `OK`: project komt in beide exports voor;
  - `ONTBREEKT_IN_ONDERHOUD`: project staat bij objecten in de paspoortexport, maar ontbreekt in de onderhoudsexport;
  - `GEEN_PASPOORTOBJECTEN`: project staat in de onderhoudsexport, maar er zijn geen paspoortobjecten met die projectnaam.
- Downloadbare controletabellen:
  - volledige Fase-4-vergelijking;
  - samenvatting paspoortprojecten;
  - samenvatting onderhoudsexport.
- De N398-voorbeeldset is als smoke test gebruikt: 11 projecten, 11x `OK`, 0 waarschuwingen.
- De sorteerbasis uit v0.15.1 blijft ongewijzigd; deze versie voert geen automatische mutaties uit.



## Nieuw in v0.15.1

Deze patch maakt de v0.15-sortering robuuster voor compacte groepen met een verdachte route-startuitschieter, zoals de N354-casus rond `GRP_RIJBAAN_40`.

- Compacte hm-groepen waarvan `route_start_m` kilometers afwijkt van `route_mid_m` gebruiken in overlapclusters nu de mediane routepositie als veilige Project Adviseur-sleutel.
- De ruwe en gecorrigeerde sorteersleutel blijven controleerbaar via `advisor_sort_raw_m`, `advisor_sort_m` en `advisor_sort_correctie`.
- Primaire objecten met een extreem grote route-span krijgen nu ook een duidelijke objectwaarschuwing, zodat ze in de waarschuwingen-/aandachtspuntenexport zichtbaar zijn.
- De bestaande v0.15-velden `advisor_sort_basis`, `advisor_sort_fallback_m` en `advisor_sort_terugval_vorige` blijven bestaan.
- De sorteerdiagnose gebruikt schema-versie `sortdiag-v0.15.1`, zodat oude Streamlit-session-state niet stil wordt hergebruikt.


## Nieuw in v0.14.2

Deze patch verbetert de sorteerdiagnose zonder de Project Adviseur-sortering inhoudelijk om te gooien.

- `route_terugval_vorige` volgt nu de feitelijke `route_sort_m`-sleutel.
- Nieuwe kolommen `route_mid_terugval_vorige` en `route_sort_terugval_vorige` maken onderscheid tussen een middelpunt-terugval en een echte sorteerterugval.
- De groepsdiagnose toont nu primaire-routevelden naast alle-object-routevelden, zodat secundaire geometrische uitschieters zichtbaar worden.
- Compacte groepen met grote routeverschillen krijgen een expliciete `route_outlier_warning`.
- Waarschuwingsteksten in de diagnose zijn versie-onafhankelijk gemaakt.
- De sorteerdiagnose gebruikt schema-versie `sortdiag-v0.14.2`, zodat oude Streamlit-session-state niet stil wordt hergebruikt.



## Nieuw in v0.11

Deze versie voegt een eerste **Sorteerdiagnose** toe zonder de bestaande Project Adviseur-sortering al inhoudelijk te vervangen.

- Onder de gewone kaart staat een expander **Diagnose huidige projectvolgorde**.
- De diagnose wordt alleen op verzoek berekend, zodat de app niet onnodig trager wordt.
- De groepsdiagnose toont per adviesgroep onder andere `sort_mode`, `sort_quality`, hectometerbereik, routepositie langs een lokaal afgeleide route-as en waarschuwingen.
- De objectdiagnose toont dubbele objecten binnen hetzelfde wegvak/metrering/situering en berekende posities langs de lokale route-as.
- De lokale route-as wordt afgeleid uit primaire objecten per wegvak/metrering-bucket. Dit is bewust diagnose-informatie; de definitieve sortering wordt nog niet aangepast.
- Diagnose-tabellen kunnen als CSV worden gedownload voor overleg en controle.


## Nieuw in v0.10

Deze versie voegt de eerste objectinspecteur en individuele paspoortmutaties toe.

- Nieuw tabblad **Objectinspecteur** naast Data Kwaliteit, Project Adviseur en Overzicht.
- Objecten zoeken op onder andere objectnummer, naam, onderhoudsproject, besteknummer, metrering en liggingvelden.
- Eén object selecteren en op de kaart markeren/naar toe zoomen.
- Huidige paspoortkern tonen als compacte inspectietabel.
- Individuele velden aanpassen via een gekozen iASSET-mutatie-/exportprofiel.
- Geometrie en bron-id's blijven bewust niet bewerkbaar; de app blijft een mutatievoorbereidingstool.
- Ontbrekende profielkolommen worden gemeld, zodat duidelijk is wat niet kan worden geëxporteerd.
- Wijzigingen worden celgewijs gelogd, maar de uiteindelijke export blijft profielgestuurd met één vaste kolommenset voor alle objecten.
- Afgeleide velden zoals `subthema_clean`, `Rank` en `hm_sort` worden bijgewerkt wanneer relevante paspoortvelden worden aangepast.
- Tests toegevoegd voor de objectinspecteur, zoekfunctie, profielvelden en objectpreview.


## Nieuw in v0.9.1

Deze hotfix is gemaakt naar aanleiding van lokale performance-metingen op een grotere wegselectie.

- De performance-zijbalk meet nu alleen de huidige Streamlit-run, zodat oude wachttijden niet blijven meetellen in het totaal.
- De Project Adviseur wijst secundaire objecten toe met één multi-source BFS in plaats van een losse netwerkzoektocht per secundair object.
- Fietspadclassificatie gebruikt herbruikbare caches en een ruimtelijke index voor primaire objecten, zodat niet elk fietspad alle primaire objecten hoeft door te meten.
- Data Kwaliteit gebruikt in de ruimtelijke controles vooraf gemaakte lookup-tabellen in plaats van duizenden losse `gdf.loc`-aanroepen.
- Excel-import scant eerst de kopregels van tabbladen en leest daarna alleen het gekozen tabblad volledig in.

## Nieuw in v0.9

Deze versie richt zich op werkbaarheid en veilige mutatievoorbereiding:

- Data Kwaliteit krijgt issuecategorieën, rule-codes en een categoriefilter.
- `check_rules()` wordt per weg/datasetrevisie gecachet, zodat de werklijst en kaartkleuring niet dezelfde zware regelcheck dubbel uitvoeren.
- PDOK-hectometerpunten zijn standaard uitgeschakeld en worden alleen opgehaald als de gebruiker de kaartlaag aanzet.
- PDOK-resultaten worden per weg/geometrische omvang in de sessie gecachet.
- De Folium-kaart stuurt minder attribuutkolommen naar de browser; zware WKT/meta-kolommen gaan niet meer standaard mee in de GeoJSON-laag.
- De zijbalk toont een eenvoudige performance-baseline met gemeten stappen.
- Export gebruikt iASSET-importprofielen, zodat duidelijk is welke kolommenset voor alle gewijzigde objecten wordt meegeschreven.
- De export toont aantallen voor gewijzigde objecten, gewijzigde cellen, meegeschreven waarden en ongewijzigde meegeschreven waarden.



## Importfix na v0.8

Deze patch houdt v0.8 als basis, maar maakt de upload/importlaag robuuster:

- CSV-import probeert meerdere encodings en scheidingstekens;
- CSV- en Excelwaarden worden als tekst gelezen, zodat ids en voorloopnullen niet onbedoeld worden gewijzigd;
- bekende kolomkopvarianten worden centraal vertaald naar de kolomnamen die de app verwacht;
- Excelimport kan een kopregel herkennen die niet op rij 1 staat;
- bij Excelwerkboeken met meerdere tabbladen wint bij gelijke herkenningsscore het tabblad met echte datarijen;
- geometrie ondersteunt nu ook `SRID=...;WKT`;
- als `gps coordinaten` leeg of corrupt is, kan `rds coordinaten` per rij als fallback worden gebruikt;
- ongeldige geometrieën worden gelogd in `invalid_geometry_rows` en stoppen de import niet;
- het Datastatus-scherm kan alle inleesfouten als CSV downloaden;
- de knop **Actieve dataset opnieuw laden** schakelt niet meer per ongeluk naar een net gekozen upload; wisselen blijft via **Gebruik deze databron** lopen.
- tests toegevoegd voor kolomaliassen, EWKT, RD-fallback, corrupte geometrie en Excelkopregels buiten rij 1.



## Wijzigingen in v0.8

Deze versie maakt de importlaag flexibeler:

- nieuw onderdeel **Databron** in de zijbalk;
- actuele iASSET-export kan direct in de app worden geüpload;
- upload ondersteunt CSV en Excel;
- meerdere bestanden kunnen tegelijk worden geüpload en samengevoegd;
- de vaste CSV-bestanden naast `app.py` blijven werken als fallback;
- bij Excelbestanden met meerdere tabbladen kiest de app het tabblad met de meeste iASSET-kolommen, met extra voorkeur voor `gps coordinaten`;
- autosave is nu databron-afhankelijk, zodat wijzigingen uit export A niet per ongeluk op export B worden toegepast;
- uploadsets krijgen een korte hash op basis van bestandsnaam, grootte en inhoud;
- tests toegevoegd voor CSV-upload, Excel-upload en dataset-hashing.

## Wijzigingen in v0.7

Deze versie bevat een kleine verbetering voor het Overzicht-tabblad:

- de kleuren in de legenda zijn geen willekeurige/cyclische categoriekleuren meer;
- alle echte waarden worden op een doorlopende kleurenschaal geplaatst;
- lage of vroege waarden beginnen blauw;
- middelste waarden lopen via turquoise/geel;
- hoge of recente waarden eindigen rood;
- `Onbekend` blijft bewust grijs en telt niet mee in de schaal;
- dezelfde kleurmapping wordt gebruikt voor legenda, kaartobjecten en HTML-export.

## Wijzigingen in v0.2

Deze versie bevat de eerste inhoudelijke bugfix na de structurele refactor:

- uitzonderingenlijst voor onderhoudsprojectplicht gelijkgezet met het werkproces Grijs;
- centrale module `iasset_tool/domain.py` toegevoegd voor domeinpredicaten;
- `NaN`, `None`, lege tekst en de tekst `"nan"` worden gelijk behandeld als lege onderhoudsprojectwaarde;
- objecten met een uitgezonderd subthema krijgen geen melding meer "Mist verplicht onderhoudsproject";
- uitgezonderde objecten die tóch een onderhoudsproject hebben, krijgen nu een waarschuwing;
- objecten met marker `Oorspronkelijke BGT-data` worden uitgezonderd, ook als die marker niet in `subthema` staat;
- Project Adviseur gebruikt dezelfde uitzonderingslogica als Data Kwaliteit;
- tests toegevoegd voor `domain.py`, `rules.py` en `advisor.py`.

Let op: `geleideconstructie` stond in de oude app als uitzondering, maar staat niet in het werkprocesdocument. Daarom is die in v0.2 niet opgenomen als uitzondering. Voeg deze alleen opnieuw toe aan `SUBTHEMA_EXCEPTIONS` als dat inhoudelijk wordt bevestigd.


## Wijzigingen in v0.3

Deze versie bevat de tweede inhoudelijke verbetering na de structurele refactor:

- Project Adviseur bouwt nu eerst alle primaire ruggengraatgroepen op;
- secundaire objecten worden pas daarna toegewezen;
- directe koppeling aan meerdere primaire objecttypen volgt de hiërarchie: rijstrook > parallelweg/landbouwpad/busbaan > fietspad;
- indirecte secundaire ketens worden verdeeld op basis van kortste graafafstand naar een primaire ruggengraat;
- bij gelijke afstand wint de hiërarchie;
- bij volledig gelijke kandidaten gebruikt de app een stabiele ruimtelijke fallback;
- groepen bevatten nu ook `primary_ids` en `secondary_ids`, zodat we later beter kunnen debuggen waarom een object in een adviesgroep zit;
- extra tests toegevoegd voor secundaire toewijzing.

Belangrijk uitgangspunt in v0.3:
de hiërarchie geldt hard bij gelijke topologische afstand. Voor indirecte ketens gaat kortste afstand vóór rang, zodat een ver weg gelegen rijstrook niet automatisch een object wegtrekt bij een direct aangrenzend fietspad.


## Wijzigingen in v0.4

Deze versie bevat de eerste uitwerking van de fietspadregel in de Project Adviseur:

- nieuwe module `iasset_tool/fietspad.py`;
- fietspaden worden geclassificeerd als:
  - `parallel_own_project`: parallelfietspad, blijft eigen onderhoudsprojectvoorstel;
  - `attached_to_main_project`: haaks/kruisend of rotonde-/kruispuntgebonden, wordt als secundair object gekoppeld aan hoofdrijbaan/parallelweg;
  - `unknown_keep_own`: onvoldoende zeker, blijft voorlopig eigen voorstelgroep met controlewaarschuwing;
- de classificatie gebruikt lokale geometrie in RD-meters, niet de globale noord-zuid/oost-westrichting van de N-weg;
- voor langgerekte vlakken gebruikt de app de hoofdrichting van de georiënteerde bounding box;
- voor lijnen gebruikt de app de richting en rechtheid van de lijn;
- rotonde-/kruispuntcontext wordt herkend via tekstmarkers en nabijgelegen objecten;
- Project Adviseur behandelt alleen duidelijke parallelfietspaden als fietspad-ruggengraat;
- duidelijke kruisende/rotondegebonden fietspaden worden toegevoegd aan de hoofdgroep en komen terug in `attached_fietspad_ids`;
- bij twijfel automatiseert de app niet door, maar markeert de groep als `review_needed`;
- de Streamlit-werklijst toont nu extra toelichting bij fietspadlogica;
- tests toegevoegd voor parallelle fietspaden, haakse fietspaden en advisor-koppeling.

Belangrijk uitgangspunt in v0.4:
liever een twijfelachtig fietspad als eigen controlevoorstel tonen dan een echt parallelfietspad per ongeluk aan de hoofdrijbaan koppelen.


## Wijzigingen in v0.5

Deze versie voegt het nieuwe tabblad **Overzicht** toe:

- derde modus naast `Data Kwaliteit` en `Project Adviseur`;
- nieuwe module `iasset_tool/overview_map.py`;
- Overzicht is alleen-lezen en voert geen mutaties uit;
- de kaart toont alleen objecten met `subthema == rijstrook`;
- de gebruiker kiest rechts in de werklijst het veld `Visualiseer op`;
- de Folium-kaart toont linksonder een legenda;
- ondersteunde attributen:
  - `Jaar aanleg`
  - `Jaar deklaag`
  - `Jaar herstrating`
  - `Jaar conservering`
  - `Besteknummer`
  - `Onderhoudsproject`
  - `Wegvaknum`
  - `Soort verharding_N`
  - `Soort deklaag specifiek`
- `Soort verharding_N` heeft een alias naar `verhardingssoort`, zodat de bestaande iASSET-export bruikbaar blijft;
- numerieke waarden, zoals jaren, worden oplopend gesorteerd in de legenda;
- lege waarden worden als `Onbekend` getoond;
- popup en tooltip tonen de belangrijkste paspoortvelden;
- tests toegevoegd voor attribuutaliases, legendasortering en rijstrookfiltering.

Belangrijk uitgangspunt in v0.5:
dit tabblad is bedoeld voor snelle visuele inspectie van bestaande rijstrookdata. Het is bewust geen mutatiescherm.


## Wijzigingen in v0.6

Deze versie werkt het tabblad **Overzicht** verder uit:

- Overzicht kan nu kiezen tussen:
  - `Geselecteerde weg`;
  - `Alle wegen`;
- bij `Alle wegen` worden alle rijstrookobjecten uit de ingeladen dataset gevisualiseerd;
- de gewone sidebar-keuze `Kies Wegnummer` blijft bestaan voor `Data Kwaliteit` en `Project Adviseur`;
- de actuele Overzicht-kaart kan als interactieve HTML-kaart worden geëxporteerd;
- de HTML-export gebruikt hetzelfde bereik en hetzelfde visualisatieattribuut als het Overzicht-tabblad;
- de export bevat de Folium/Leaflet-kaart, kleurlaag, legenda en popups;
- `parse_date_info()` herkent compacte iASSET-tijdstempels zoals `20260512095736` nu expliciet, waardoor de pandas-waarschuwing over `dayfirst=True` verdwijnt;
- tests toegevoegd voor compacte datum/tijdstempels, bestandsnaamopschoning, alle-wegen-overzicht en HTML-export.

Belangrijk uitgangspunt in v0.6:
HTML is de eerste exportvorm, omdat Folium zelf HTML/JavaScript genereert. PDF-export blijft een later verbeterpunt; daarvoor moeten we bepalen of we een statische printkaart of een screenshot van de actuele browserweergave willen.
