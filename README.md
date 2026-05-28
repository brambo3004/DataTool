# iASSET Advisor - refactor v0.21.0

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
