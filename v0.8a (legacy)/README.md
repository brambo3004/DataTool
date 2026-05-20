# iASSET Advisor - refactor v0.8 + importfix

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
│   ├── map_view.py
│   ├── overview_map.py
│   ├── changes.py
│   └── state.py
├── tests/
│   ├── test_utils.py
│   ├── test_data_loader.py
│   ├── test_domain.py
│   ├── test_rules.py
│   ├── test_advisor.py
│   ├── test_fietspad.py
│   └── test_overview_map.py
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
