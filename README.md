\
# iASSET Advisor - refactor v0.5

Deze versie splitst de bestaande Streamlit proof-of-concept op in een onderhoudbare projectstructuur.

## Doel van deze refactor

Eerst structuur, daarna bugs en nieuwe functionaliteit.

De huidige functionaliteit blijft zoveel mogelijk gelijk:
- iASSET CSV's inlezen;
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

Plaats de twee iASSET-exportbestanden naast `app.py`:

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
