# iASSET Advisor - refactor v0.34.1

Deze versie splitst de bestaande Streamlit proof-of-concept op in een onderhoudbare projectstructuur.


## Nieuw in v0.34.1

v0.34.1 is een stabilisatieversie van **Projectgrenzen op referentieas** binnen `🧪 NWB referentieproef`.

Kern:
- projectdekking, gaten en overlap worden nu per projecttype gecontroleerd, bijvoorbeeld HRB met HRB en PWR met PWR;
- HRB, parallelwegen en fietspaden worden niet meer als onderlinge overlap gezien;
- onderhoudsprojectnamen krijgen een aparte validatie op naamvorm, wegnummer, projecttype, situering, voorloopnul en begin/einde;
- de beheerregel voor projectnaam-hectometrering blijft expliciet vastgelegd: 12.300 -> 12.3, maar 12.301 -> 12.4;
- fysieke objectligging blijft zichtbaar, maar bepaalt niet meer automatisch de hoofdstatus van de projectgrens;
- exports bevatten aparte kolommen voor `status_projectnaam`, `status_projectgrens`, `objectligging_status` en eindstatus `status`;
- er worden nog steeds geen beheerknippen of mutaties in iASSET gemaakt.

Nieuwe of aangepaste exports:
- `Projectgrenzen_Referentieas_<weg>.csv`
- `Projectdekking_Referentieas_<weg>.csv`
- `NWB_Hectopunten_op_iASSET_Wegas_<weg>.csv`
- `Projectobjecten_Referentieas_<weg>.csv`

De functionaliteit blijft bewust proef/diagnose. iASSET blijft de bron van waarheid; de NWB-ijking is alleen een hulpmiddel voor databeheercontrole.


## Nieuw in v0.34.0

v0.34.0 voegde de diagnose **Projectgrenzen op referentieas** toe binnen het scherm `🧪 NWB referentieproef`.

Kern:
- de iASSET-wegas uit `wegassen_paspoort.geojson` blijft de leidende as;
- NWB-hectopunten worden op die iASSET-wegas geprojecteerd en gebruikt als ijkpunten;
- onderhoudsprojectnamen zoals `N354-HRB-11.5-12.8` worden op de geijkte as geplaatst;
- begin- en eindgrenzen worden gewaarschuwd wanneer ze in of vlak bij oranje/rode NWB-afwijkingszones liggen;
- projectdekking, gaten, overlap en lengteverschillen worden als diagnose geëxporteerd;
- fysieke objectligging wordt indicatief vergeleken met de projectnaamrange;
- er worden geen beheerknippen of mutaties in iASSET gemaakt.

Nieuwe module:
- `iasset_tool/project_axis.py`

## Nieuw in v0.33.0

Deze versie voegt een afgeschermde experimentele **NWB referentieproef** toe.

- Nieuw scherm `🧪 NWB referentieproef` naast de bestaande modi.
- Nieuwe module `iasset_tool/nwb.py` voor bronverkenning via de officiële NWB OGC API Features.
- De proef haalt NWB-wegvakken en NWB-hectopunten op binnen de bbox van de geselecteerde iASSET-weg.
- Wegvakken worden gefilterd op wegnummer/routenummer; hectopunten worden gekoppeld via `wvk_id`.
- Optioneel kan een iASSET-wegassen-GeoJSON worden geüpload om de interne Dielplak-wegas ruimtelijk met NWB-wegvakken te vergelijken.
- Output: NWB-bronsamenvatting, wegvakken-attributen, hectopunten-attributen en optionele wegasvergelijking als CSV.
- De bestaande v0.32.1 **Referentieas / PDOK-proef** blijft beschikbaar, maar wordt niet verder als eindrichting gebruikt.
- De proef is bewust alleen-lezen: iASSET blijft de single source of truth.

## Nieuw in v0.32.0

Deze versie voegt een afgeschermde experimentele **Referentieas / PDOK-proef** toe.

- Nieuw scherm `🧪 Referentieas / PDOK-proef` naast de bestaande modi.
- Nieuwe module `iasset_tool/reference_axis.py` voor diagnostiek op basis van PDOK-hectometerpunten.
- De proef is bewust alleen-lezen: iASSET blijft de single source of truth.
- De bestaande Onderhoudscontrole, Overzicht en v0.31-trajectlengtekeuze blijven ongemoeid.
- Scope v0.32: eerst rijstroken/HRB, bedoeld voor N354 en N398.
- Output: projectsamenvatting en objectdiagnose met ruwe referentiemetrering, vijfmeter-afleiding, afstand tot as en bronkwaliteit `experimenteel`.
- Robuustheid: ontbrekende PDOK-data, lege geometrieën en schema-afwijkingen leveren waarschuwingen op in plaats van crashes.


## Nieuw in v0.30.0

Deze versie maakt de metrering- en trajectlogica explicieter en herbruikbaar.

- Nieuwe centrale module `iasset_tool/trajectory.py`.
- Overzicht maakt nu onderscheid tussen:
  - **trajectlengte precies/voorkeur**: bij voorkeur uit objectmetrering;
  - **trajectlengte naam**: administratieve lengte uit de onderhoudsprojectnaam;
  - **objectlengte**: technische optelsom van rijstrookobjecten;
  - **oppervlakte**: m² uit polygongeometrie of betrouwbare bronkolom.
- Als objectmetrering en onderhoudsprojectnaam duidelijk verschillen, toont de hoeveelheidstabel een waarschuwing.
- De logica voorkomt nog steeds dat losse wegdelen met hetzelfde legenda-item automatisch als één lang traject worden opgeteld.
- Dezelfde metreringlogica is voorbereid voor latere naamvoorstellen in Project Adviseur.
- De veiligheidsregel blijft ongewijzigd: de tool controleert, visualiseert en exporteert, maar voert geen mutaties door in iASSET.

## Nieuw in v0.29.0

Het tabblad **Overzicht** maakt nu expliciet onderscheid tussen:
- **trajectlengte**: lengte van het wegdeel langs de metrering of onderhoudsprojectnaam;
- **oppervlakte**: m² uit polygongeometrie of betrouwbare bronkolom;
- **objectlengte**: technische optelsom van rijstrookobjecten.

Objectlengte blijft beschikbaar, maar trajectlengte en oppervlakte staan prominenter in legenda, tabel en HTML-export.

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
├── config/
│   └── onderhoudsregels.json
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
│   ├── reference_axis.py
│   ├── performance.py
│   ├── map_view.py
│   ├── overview_map.py
│   ├── object_editor.py
│   ├── sorting_diagnostics.py
│   ├── maintenance_control.py
│   ├── maintenance_map.py
│   ├── changes.py
│   └── state.py
└── tests/
    ├── test_advisor.py
    ├── test_changes.py
    ├── test_data_loader.py
    ├── test_domain.py
    ├── test_fietspad.py
    ├── test_overview_map.py
    ├── test_object_editor.py
    ├── test_reference_axis.py
    ├── test_rules.py
    ├── test_sorting_diagnostics.py
    ├── test_maintenance_map.py
    └── test_utils.py
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

## Nieuw in v0.28.0

Deze versie breidt het tabblad **Overzicht** uit met hoeveelheden per legenda-item.

- De legenda toont nu per waarde:
  - aantal rijstrookobjecten;
  - kilometers;
  - vierkante meters wanneer een betrouwbare oppervlaktebron beschikbaar is.
- De app toont onder de Overzichtkaart een tabel `Hoeveelheden per legenda-item`.
- Deze tabel kan als CSV worden gedownload.
- De HTML-export van de Overzichtkaart bevat nu ook totaalhoeveelheden en de gebruikte bronnen.
- Lengtes worden veilig bepaald:
  - eerst uit een expliciete lengtekolom als die aanwezig is;
  - anders uit lijngeometrie;
  - bij polygonen alleen als controle-indicatie uit oppervlakte / administratieve breedte.
- Oppervlakte wordt bepaald uit polygongeometrie of, als terugval, uit een beschikbare oppervlaktekolom.
- De veiligheidsregel blijft ongewijzigd: Overzicht is alleen-lezen en voert geen mutaties door in iASSET.

## Nieuw in v0.27.0

Deze versie maakt de **Onderhoudscontrole** meer geschikt als controledossier.

- De app heeft nu een controleprofiel:
  - Volledige controle;
  - Snelle controle;
  - Alleen datakwaliteit;
  - Werkvoorraadcontrole.
- Het profiel verandert niets automatisch in iASSET. Het bepaalt vooral hoe de resultaten worden gepresenteerd en waar de gebruiker begint.
- De Onderhoudscontrole toont nu een duidelijk eindoordeel met aanbevolen eerste stap.
- Het Excel-controlepakket bevat een nieuw tabblad `Voorblad`.
- Het voorblad bevat:
  - gekozen controleprofiel;
  - controlebereik;
  - eindoordeel;
  - aanbevolen eerste stap;
  - datakwaliteitscijfers;
  - open controlepunten;
  - prioriteiten;
  - mogelijke oude/nieuwe projectnaam-hints;
  - veiligheidsmelding dat niets automatisch wordt doorgevoerd.
- Het voorblad toont daarnaast de belangrijkste projectgroepen, zodat het controlepakket direct bruikbaar is in overleg of archivering.
- De veiligheidsregel blijft ongewijzigd: de tool signaleert, ordent en exporteert, maar voert geen mutaties door in iASSET.

## Nieuw in v0.26.0

Deze versie verplaatst de belangrijkste **beheerregels** naar één configuratiebestand:

```text
config/onderhoudsregels.json
```

Daarin staan onder meer:
- primaire ruggengraat-subthema's;
- subthema's die geen onderhoudsproject moeten krijgen;
- herkenningsmarkers zoals oorspronkelijke BGT-data;
- categoriefamilies zoals HRB/HRBR/HRBL, PW/PWR/PWL en FP/FPR/FPL;
- knipvelden voor de Project Adviseur;
- exportprofielen;
- wegrichtingen voor sortering.

De bestaande Python-constanten blijven beschikbaar voor de code, maar worden nu uit dit configuratiebestand gevuld. Als het bestand ontbreekt of stuk is, valt de tool veilig terug op ingebouwde standaardregels en toont de app een waarschuwing in de zijbalk onder **Beheerregels**.

Belangrijk: deze configuratie verandert alleen controle-, sorteer-, uitleg- en exportlogica. De tool voert nog steeds geen automatische mutaties door in iASSET.

## Nieuw in v0.25.0

Deze versie verdiept de **kaartcontrole** binnen de Onderhoudscontrole.

- De kaart van een controlepunt maakt primaire ruggengraatobjecten duidelijker zichtbaar dan secundaire objecten.
- Uitgezonderde objecten krijgen een aparte gestippelde stijl.
- De kaart toont een legenda voor verschiltypen en objectlagen.
- Pop-ups tonen nu meer controlecontext, zoals oude projectnaam, mogelijke vervangende projectnaam, prioriteit, duiding en voortgang.
- De app toont samenvattende kaarttellingen: hoeveel objecten primair, secundair of uitgezonderd zijn.
- De kaart kan als HTML-controlebeeld worden gedownload.
- De veiligheidsregel blijft ongewijzigd: de kaart is alleen controlehulp. De tool voert niets automatisch door in iASSET.

## Nieuw in v0.24.0

Deze versie verdiept de **voortgangsvergelijking** tussen opeenvolgende Onderhoudscontroles.

- Naast losse tellingen voor `nieuw`, `bestaand` en `opgelost_of_niet_meer_gevonden` maakt de tool nu een voortgangsrapport per weg en per onderhoudsproject.
- Het rapport laat zien welke projectgroepen:
  - nieuw zijn in deze controle;
  - blijven terugkomen;
  - blijven terugkomen met hoge prioriteit;
  - deels nieuw en deels bestaand zijn;
  - mogelijk opgelost zijn of buiten de nieuwe exportselectie vallen.
- De app toont extra voortgangsmetrics zodra een vorige actielijst is ingelezen.
- Er is een download beschikbaar voor `Onderhoudscontrole_Voortgangsrapport.csv`.
- Het Excel-controlepakket bevat een extra tabblad `Voortgangsrapport`.
- De veiligheidsregel blijft ongewijzigd: voortgangsconclusies zijn alleen controle-informatie. De tool voert niets automatisch door in iASSET.

## Nieuw in v0.23.0

Deze versie voegt een **Datakwaliteitsrapport** toe aan de Onderhoudscontrole.

- Voor de inhoudelijke vergelijking controleert de tool of de gebruikte paspoort- en onderhoudsexports voldoende betrouwbaar zijn.
- De voorcontrole signaleert onder meer ontbrekende verplichte kolommen, lege of dubbele objectnummers, ontbrekende projectnamen, afwijkende projectnaam-syntax, ongeldige metrering en ontbrekende geometrie.
- De app toont dashboardcijfers voor blokkerende datakwaliteitsproblemen, waarschuwingen en aandachtspunten.
- Het Excel-controlepakket bevat een extra tabblad `Datakwaliteit`.
- De controle blijft veilig: het rapport signaleert alleen en past niets aan in iASSET.

## Nieuw in v0.22.0

Deze versie maakt de **Onderhoudscontrole** rustiger en beter afhandelbaar.

- De werkvoorraad krijgt per controlepunt een prioriteit:
  - `prioriteit`;
  - `prioriteit_score`;
  - `prioriteit_uitleg`;
  - `project_samenvatting`.
- De prioriteit wordt uitlegbaar opgebouwd uit ernst, technische status, aantal betrokken objecten, primaire paspoortobjecten en eventuele projectmatch-hints.
- Er is een nieuwe samenvatting per onderhoudsproject:
  - aantal controlepunten;
  - open controlepunten;
  - waarschuwingen/aandachtspunten;
  - aantal primaire en secundaire paspoortobjecten;
  - mogelijke vervangende projectnaam;
  - korte conclusie;
  - aanbevolen volgende stap.
- De Streamlit-app toont deze projectsamenvatting boven de detailwerkvoorraad, zodat de databeheerder eerst op projectniveau kan prioriteren.
- Het Excel-controlepakket bevat een extra tabblad `Projectsamenvatting` en extra samenvattingscijfers per prioriteit.
- De veiligheidsregel blijft ongewijzigd: prioriteit en projectconclusie zijn alleen controlehulp. De tool voert niets automatisch door in iASSET.

## Nieuw in v0.21.0

Deze versie verbetert de herkenning van oude versus nieuwe onderhoudsprojectnamen in de **Onderhoudscontrole**.

- Wanneer een project in de paspoortexport staat maar in de onderhoudsexport ontbreekt, zoekt de tool nu explicieter naar een mogelijke vervangende projectnaam.
- De matchhint gebruikt meerdere veilige signalen:
  - hetzelfde wegnummer;
  - dezelfde projectcategorie of categoriefamilie, zoals HRB/HRBR/HRBL, PW/PWR/PWL en FP/FPR/FPL;
  - overlappende of nabije hectometrering uit projectnaam en paspoortobjecten;
  - objectnummers die in de onderhoudsexport onder een andere projectnaam terugkomen;
  - beperkte tekstuele overeenkomst tussen projectnamen.
- De werkvoorraad krijgt extra kolommen:
  - `mogelijke_vervangende_projectnaam`;
  - `vervangende_projectnaam_score`;
  - `vervangende_projectnaam_uitleg`;
  - `vervangende_projectnaam_criteria`.
- De mutatievoorstellen bevatten dezelfde mogelijke vervangende projectnaam en matchuitleg.
- De veiligheidsregel blijft ongewijzigd: dit zijn alleen hints en conceptvoorstellen. De tool hernoemt niets, koppelt niets automatisch en voert geen mutaties door in iASSET.


## Nieuw in v0.20.0

Deze versie voegt detail- en kaartweergave toe aan de **Onderhoudscontrole**.

- Bij een geselecteerd controlepunt kan de gebruiker nu een objectdetailtabel openen.
- De detailtabel toont per betrokken object:
  - of het object in de paspoortexport staat;
  - of het object in de onderhoudsexport staat;
  - verschiltype;
  - subthema, metrering, situering en onderhoudsproject waar beschikbaar;
  - of er paspoortgeometrie beschikbaar is voor kaartweergave.
- De app toont de betrokken paspoortobjecten op een kaart.
- Objecten die alleen in de onderhoudsexport staan blijven zichtbaar in de detailtabel, maar kunnen zonder paspoortgeometrie niet op kaart worden getekend.
- Per controlepunt kan de objectdetailtabel als CSV worden gedownload.
- De veiligheidsregel blijft ongewijzigd: de tool visualiseert en ondersteunt, maar voert nooit automatisch wijzigingen door.


## Nieuw in v0.19.2

Deze versie voegt voortgangsbewaking toe aan de Onderhoudscontrole.

- Wanneer een eerdere actielijst wordt meegegeven, markeert de tool controlepunten als:
  - `nieuw_controlepunt`;
  - `bestaand_controlepunt`;
  - `opgelost_of_niet_meer_gevonden`.
- De werkvoorraad krijgt nieuwe kolommen `voortgang_status` en `voortgang_uitleg`.
- Het dashboard toont hoeveel controlepunten nieuw, bestaand of opgelost/niet meer gevonden zijn.
- Er is een aparte download voor controlepunten die in de vorige actielijst stonden, maar nu niet meer terugkomen: `Onderhoudscontrole_Opgelost*.csv`.
- Het Excel-controlepakket krijgt een extra tabblad `Opgelost`.
- De tool voert nooit automatisch wijzigingen door; een verdwenen controlepunt moet altijd nog inhoudelijk worden beoordeeld.

## Nieuw in v0.19.1

Deze versie voegt een netwerkbreed **Onderhoudscontrole-controlepakket in Excel** toe.

- Nieuwe downloadknop: `Onderhoudscontrole_Controlepakket*.xlsx`.
- Het Excelbestand bundelt de belangrijkste controle-output in één deelbaar bestand.
- Tabbladen:
  - `Samenvatting`;
  - `Werkvoorraad`;
  - `Mutatievoorstellen`;
  - `Resultaten`;
  - `Objectverschillen`;
  - `Paspoortprojecten`;
  - `Onderhoudsexport`.
- De samenvatting bevat kerncijfers en verdelingen per weg, duidingsgroep, afhandelstatus en actiehouder.
- Bijgewerkte opvolgvelden uit de app worden meegenomen in het Excelpakket.
- De tool voert nooit automatisch wijzigingen door; het Excelpakket is alleen bedoeld voor controle, overleg en archivering.

## Nieuw in v0.19.0

Deze versie maakt van de voormalige Fase-4-weergave een begrijpelijke **Onderhoudscontrole** die ook netwerkbreed kan draaien.

- Het tabblad heet nu **Onderhoudscontrole**.
- De gebruiker kiest in het tabblad het controlebereik:
  - `Geselecteerde weg`;
  - `Hele dataset / wegennet`.
- Bij netwerkbrede controle gebruikt de tool de volledige actieve paspoortexport en de geüploade onderhoudsexport.
- De resultaten krijgen een kolom `wegnummer`, zodat de werkvoorraad per weg kan worden gefilterd.
- Het dashboard toont hoeveel wegen in de controle zitten.
- Bij netwerkbrede controle toont de werkvoorraad een compacte verdeling van controlepunten per weg.
- Nieuwe exportnamen gebruiken `Onderhoudscontrole_*`, bijvoorbeeld:
  - `Onderhoudscontrole_Actielijst.csv`;
  - `Onderhoudscontrole_Resultaten.csv`;
  - `Onderhoudscontrole_Mutatievoorstellen.csv`.
- Bij losse wegcontrole wordt het wegnummer nog aan de bestandsnaam toegevoegd, bijvoorbeeld `Onderhoudscontrole_Actielijst_N354.csv`.
- Oude `Fase4_Actielijst`-bestanden kunnen nog steeds worden ingelezen om eerdere beoordelingen opnieuw mee te nemen.
- Bij netwerkbrede controle vergelijkt de tool verdachte objectnummers met het wegnummer uit de onderhoudsprojectnaam.
- De tool voert nooit automatisch wijzigingen door.

## Nieuw in v0.18.2

Deze versie maakt de Fase-4-output beter te beoordelen zonder iets automatisch te wijzigen.

- De map `legacy/` wordt niet meer opgenomen in release-zips.
- De Fase-4-actielijst krijgt een extra kolom `duiding`.
- `duiding` helpt meldingen sneller te ordenen, bijvoorbeeld:
  - `waarschijnlijke_fout_in_paspoort`;
  - `fout_of_grensgeval_controleren`;
  - `mogelijke_oude_projectnaam`;
  - `ontbrekend_project_of_exportfilter_controleren`;
  - `koppeling_of_exportmoment_controleren`.
- Mutatievoorstellen krijgen dezelfde duiding naast de bestaande veiligheidskolommen.
- De app toont samenvattende aantallen per duiding en kan de werkvoorraad erop filteren.
- De tool voert nog steeds nooit automatische wijzigingen door.

## Nieuw in v0.18.1

Deze versie scherpt de veiligheidslaag rond **mutatievoorstellen** aan.

- De tool voert nog steeds nooit automatische wijzigingen door.
- Mutatievoorstellen zijn expliciet gemarkeerd als `concept_voorstel`.
- Nieuwe veiligheidskolommen in `Fase4_Mutatievoorstellen_[weg].csv`:
  - `voorstelstatus`;
  - `menselijke_controle_verplicht`;
  - `automatisch_doorvoeren`;
  - `veiligheidsmelding`.
- `automatisch_doorvoeren` staat altijd op `nee`.
- `menselijke_controle_verplicht` staat altijd op `True`.
- De Streamlit-app toont bij mutatievoorstellen een duidelijke veiligheidsmelding.
- Dit bevestigt het ontwerpprincipe: de tool mag signaleren, uitleggen, voorstellen en exporteren, maar de gebruiker houdt altijd de regie.

## Nieuw in v0.18.0

Deze versie voegt **veilige mutatievoorstellen** toe aan de Fase-4-controle.

- Nieuwe export: `Fase4_Mutatievoorstellen_[weg].csv`.
- De tool vertaalt controlepunten naar voorstelregels voor correctie of controle in iASSET.
- Voorstellen bevatten onder andere:
  - `voorsteltype`;
  - `bron_export`;
  - `objectnummer`;
  - `veld`;
  - `huidige_waarde`;
  - `voorgestelde_waarde`;
  - `zekerheid`;
  - `voorgestelde_controle`;
  - `alleen_na_controle`.
- De app voert nog steeds géén mutaties uit. Elk voorstel blijft een controle-/werkregel voor de databeheerder.
- De bestaande Fase-4-actielijst, filters, eerdere beoordelingen en mogelijke onderhoudsmatches blijven behouden.

## Nieuw in v0.17.1

Deze versie helpt bij ontbrekende onderhoudsprojecten door mogelijke bestaande onderhoudsprojecten als **hint** te tonen.

- Bij `ONTBREEKT_IN_ONDERHOUD` zoekt de tool voorzichtig naar een mogelijke match in de onderhoudsexport.
- De match kijkt naar hetzelfde wegnummer, vergelijkbare projectcategorie en overlappend of nabij hm-bereik.
- Nieuwe kolommen in de Fase-4-controle en actielijst:
  - `mogelijke_onderhoudsmatch`;
  - `onderhoudsmatch_type`;
  - `onderhoudsmatch_score`;
  - `onderhoudsmatch_uitleg`.
- De hint wordt ook in de detailweergave van de werkvoorraad getoond.
- De app corrigeert niets automatisch; de databeheerder beoordeelt of de match echt klopt.


## Nieuw in v0.17.0

Deze versie maakt de **Fase-4-actielijst** in de app zelf bruikbaar als filterbare en bewerkbare werkvoorraad.

- Nieuw werkvoorraadoverzicht met aantallen voor controlepunten, open, nieuw, afgehandeld, waarschuwingen en aandachtspunten.
- Filters toegevoegd voor ernst, technische status, praktische categorie, afhandelstatus, actiehouder en vrije zoektekst.
- De opvolgvelden kunnen direct in de Streamlit-tabel worden bewerkt:
  - `beoordeling_databeheerder`;
  - `afhandelstatus`;
  - `actiehouder`;
  - `opmerking_afhandeling`.
- De download `Fase4_Actielijst_[weg].csv` bevat de bijgewerkte opvolgvelden.
- Detailweergave toegevoegd met uitleg, mogelijke oorzaak en voorgestelde actie per controlepunt.
- De controle voert nog steeds géén mutaties uit in iASSET of het maatregeltoetsdocument.


## Nieuw in v0.16.4

Deze versie maakt de **Fase-4-actielijst herbruikbaar** bij herhaalde controles.

- In de zijbalk kan optioneel een eerder ingevulde `Fase4_Actielijst` worden geüpload.
- Als dezelfde controlepunten opnieuw voorkomen, neemt de app deze opvolgvelden opnieuw mee:
  - `beoordeling_databeheerder`;
  - `afhandelstatus`;
  - `actiehouder`;
  - `opmerking_afhandeling`.
- Nieuwe of gewijzigde controlepunten blijven op `afhandelstatus = nieuw` staan.
- De app meldt hoeveel eerdere beoordelingen opnieuw zijn overgenomen.
- De technische controle blijft leidend: alleen de beoordeling/afhandeling wordt meegenomen, niet de oude status zelf.


## Nieuw in v0.16.3

Deze versie maakt de **Fase-4-actielijst** geschikt als werkvoorraad voor afhandeling door databeheer.

- Nieuwe kolom `praktische_categorie`, zodat technische meldingen praktischer gegroepeerd kunnen worden.
- Nieuwe standaardkolommen voor opvolging:
  - `beoordeling_databeheerder`;
  - `afhandelstatus`;
  - `actiehouder`;
  - `opmerking_afhandeling`.
- Nieuwe actieregels starten met `afhandelstatus = nieuw`.
- De actielijst blijft een veilige controle-export: de app voert géén mutaties uit in iASSET of het maatregeltoetsdocument.

## Nieuw in v0.16.2

Deze versie maakt de **Fase 4-controle** praktischer voor dagelijks gebruik door een actielijst toe te voegen.

- Nieuwe downloadbare tabel:
  - `Fase4_Actielijst_[weg].csv`.
- De actielijst vertaalt technische statussen naar controlewerk:
  - wat is er aan de hand;
  - welke objecten zijn betrokken;
  - wat is een mogelijke oorzaak;
  - welke controleactie ligt voor de hand.
- De technische exports blijven bestaan:
  - `Fase4_Controle_[weg].csv`;
  - `Fase4_Paspoortprojecten_[weg].csv`;
  - `Fase4_Onderhoudsexport_[weg].csv`;
  - `Fase4_Objectverschillen_[weg].csv`.
- De controle voert nog steeds géén mutaties uit. Hij is bedoeld als overleg- en acceptatiecontrole vóór verwerking in iASSET/maatregeltoets.

## Nieuw in v0.16.1

Deze versie verdiept de **Fase 4-controle** van projectnaamcontrole naar objectinhoudcontrole.

- Projectnamen worden nog steeds vergeleken tussen paspoortexport en onderhoudsexport.
- Daarnaast vergelijkt de tool nu per onderhoudsproject de objectsets:
  - object staat wel in paspoort, niet in onderhoud;
  - object staat wel in onderhoud, niet in paspoort;
  - objectnummer in onderhoud lijkt bij een ander wegnummer te horen.
- Ongeldige meteringen, zoals `4,,9`, worden niet meer meegenomen in `hm_min`/`hm_max`.
- Zulke meteringen worden apart gerapporteerd via `paspoort_ongeldige_metrering_aantal` en de objectverschillenexport.
- Nieuwe/strengere statussen:
  - `OK_VOLLEDIG`: projectnaam én objectset kloppen;
  - `OBJECTVERSCHIL`: projectnaam klopt, maar objectsets verschillen;
  - `OBJECT_WEGNUMMER_VERDACHT`: onderhoudsexport bevat een objectnummer dat bij een ander wegnummer lijkt te horen;
  - `HM_BEREIK_VERDACHT`: hm-bereik is berekend met genegeerde ongeldige metrering;
  - `ONTBREEKT_IN_ONDERHOUD`;
  - `GEEN_PASPOORTOBJECTEN`.
- Nieuwe downloadbare tabel:
  - `Fase4_Objectverschillen_[weg].csv`.
- De controle voert nog steeds géén mutaties uit. Hij is bedoeld als overleg- en acceptatiecontrole vóór verwerking in iASSET/maatregeltoets.

## Nieuw in v0.16.0

Deze versie voegt de eerste **Fase 4-controle** toe: paspoortexport en onderhoudsexport worden naast elkaar gelegd voordat er iets in iASSET of het maatregeltoetsdocument wordt verwerkt.

- Nieuw onderdeel **Fase 4 Controle** in de werklijst.
- Aparte upload voor onderhoudsexports, omdat deze bestanden meestal geen geometrie bevatten en dus niet door de gewone paspoort/geometrielader horen te gaan.
- De controle groepeert paspoortobjecten per `Onderhoudsproject` en onderhoudsregels per projectnaam.
- De vergelijking markeert projectnamen die ontbreken of verweesd zijn.
- Downloadbare controletabellen:
  - volledige Fase-4-vergelijking;
  - samenvatting paspoortprojecten;
  - samenvatting onderhoudsexport.
- De sorteerbasis uit v0.15.1 blijft ongewijzigd; deze versie voert geen automatische mutaties uit.


## v0.23.0

- Toegevoegd: datakwaliteitsrapport voor de Onderhoudscontrole.
- Controleert de gebruikte paspoort- en onderhoudsexports op ontbrekende kolommen, lege waarden, dubbele objectnummers, afwijkende projectnamen, ongeldige metrering en geometrieproblemen.
- Blijft alleen signalerend: de tool voert geen automatische wijzigingen door in iASSET.
