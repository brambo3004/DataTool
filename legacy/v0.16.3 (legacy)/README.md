# iASSET Advisor - refactor v0.16.3

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
