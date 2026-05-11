\
# iASSET Advisor - refactor v0.1

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
│   ├── utils.py
│   ├── data_loader.py
│   ├── geometry.py
│   ├── rules.py
│   ├── advisor.py
│   ├── pdok.py
│   ├── map_view.py
│   ├── changes.py
│   └── state.py
├── tests/
│   ├── test_utils.py
│   └── test_data_loader.py
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

## Belangrijk

De uitzonderingenlijst voor subthema's is bewust nog gelijk gehouden aan de bestaande app.  
In het werkprocesdocument staat een bredere lijst. Dat is een inhoudelijke bug/verbetering voor de volgende fase.
