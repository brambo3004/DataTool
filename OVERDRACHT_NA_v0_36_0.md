# Overdracht na v0.36.0

## Status

v0.36.0 is de eerste versie van **Project Adviseur 2.0**. De bestaande
v0.35.5-projectas/greenfield-engine is inhoudelijk behouden. De grote wijziging
zit in de Streamlit-presentatie: het hoofdscherm is nu gericht op dagelijks
databeheer in plaats van technische diagnose.

## Wat is gebouwd

### 1. Project Adviseur als hoofdmodus

`🏗️ Project Adviseur` staat nu als eerste modus. Dit scherm is bedoeld als
functionele ingang voor databeheerders.

Het scherm bevat:

1. upload van de iASSET-wegassen GeoJSON;
2. knop `🚦 Maak projectadvies`;
3. compacte samenvatting;
4. voorstellenlijst met gescheiden statussen;
5. kaartinspectie van één voorstel;
6. detailpaneel;
7. werklijst;
8. technische diagnose onder een expander.

### 2. Gescheiden statussen

De bestaande projectvoorstellen worden in de UI verrijkt met:

| Status | Betekenis |
|---|---|
| `adviesstatus` | Inhoudelijke status uit de bestaande engine, meestal gebaseerd op `status_voorstel`. |
| `datakwaliteitstatus` | Signalen zoals ontbrekend besteknummer of lokale datakwaliteit. |
| `grensstatus` | Signalen rond begin/eindgrens, hm-interval en ijking. |
| `referentieasstatus` | Signalen dat de gebruikte as/ijking niet betrouwbaar genoeg is. |
| `eindadvies` | Leesbare samenvatting voor de databeheerder. |

Belangrijk: datakwaliteit is nu losgetrokken van het inhoudelijke projectadvies.
Een voorstel kan dus inhoudelijk `ok` zijn terwijl de datakwaliteit nog aandacht
vraagt.

### 3. Nieuwe presentatiemodule

Nieuw bestand:

```text
iasset_tool/project_advisor_v2.py
```

Deze module bevat geen GIS-rekenlogica. Zij doet alleen presentatielogica:

- projectvoorstellen verrijken met statussen;
- samenvatting maken;
- werklijst maken.

Daarmee blijft `project_axis.py` de centrale rekenmotor.

### 4. Oude adviesgroepen behouden

De oude v0.35.5-adviesgroepenworkflow is niet verwijderd. Zij staat onder:

```text
Toon oude adviesgroepen uit v0.35.5
```

Dit voorkomt verlies van functionaliteit, maar maakt duidelijk dat dit niet meer
het hoofdpad is.

## Bestanden

### Gewijzigd

```text
app.py
README.md
iasset_tool/__init__.py
iasset_tool/config.py
```

### Nieuw

```text
iasset_tool/project_advisor_v2.py
tests/test_project_advisor_v2.py
RELEASE_NOTES_v0_36_0.md
OVERDRACHT_NA_v0_36_0.md
```

### Verwijderd

```text
geen
```

## Validatie

Uitgevoerd:

```text
pytest -q
```

Resultaat:

```text
189 passed
```

Daarnaast zijn alle Python-bestanden met `py_compile` gecontroleerd.

## Testplan voor Streamlit

### Test 1 — basisroute N398

1. Open de Streamlit-app.
2. Controleer dat `🏗️ Project Adviseur` de eerste modus is.
3. Kies `N398`.
4. Upload `wegassen_paspoort.geojson`.
5. Klik op `🚦 Maak projectadvies`.

Verwachte uitkomst:

- Project Adviseur toont een samenvatting.
- Projectvoorstellen worden getoond met gescheiden statussen.
- De technische diagnose staat niet centraal.

### Test 2 — kaartinspectie

1. Kies een projectvoorstel.
2. Klik op `🔎 Zoom hoofdkaart op geselecteerd voorstel`.

Verwachte uitkomst:

- Objecten in het voorstel worden links op de kaart uitgelicht.
- Bestaande iASSET-contextobjecten worden apart als context getoond indien
  beschikbaar.

### Test 3 — N398-rust

Controleer specifiek:

- het traject `N398-HRB-01.6-04.6` blijft als logisch voorstel zichtbaar;
- lokale ontbrekende besteknummers worden als datakwaliteit gepresenteerd, niet
  als zelfstandige harde projectknip;
- het eindgebied rond hm 6.3 wordt niet als normale nulmeterprojectlogica
  gepresenteerd, maar via grens-/ijkingsinformatie.

### Test 4 — technische diagnose

Open de expander `Technische diagnose en oude exports`.

Verwachte uitkomst:

- bestaande exports blijven beschikbaar;
- intervaldiagnose blijft beschikbaar;
- objecttoewijzing en vergelijking met iASSET blijven beschikbaar;
- deze tabellen staan niet meer bovenaan als hoofdworkflow.

### Test 5 — N354-praktijktoets

Herhaal de basisroute met `N354`.

Verwachte uitkomst:

- N354 mag meer aandachtspunten tonen dan N398;
- voorstellen blijven uitlegbaar;
- het scherm moet niet dichtslibben met technische detailtabellen.

## Belangrijke ontwerpkeuze

De berekening voor Project Adviseur gebruikt nog steeds dezelfde sessiestatus
als de NWB-referentieproef:

```text
st.session_state["nwb_reference_diagnostics"]
```

Dat is bewust gedaan om in v0.36.0 geen dubbele projectaslogica of parallelle
state-structuur te introduceren. De consequentie is dat Project Adviseur en NWB
debug hetzelfde resultaat kunnen tonen vanuit twee presentaties.

Voor een latere refactor kan dit netter worden gemaakt, bijvoorbeeld:

```text
st.session_state["project_advisor_result"]
```

Maar dat hoort bij een kleine Route B-refactor, niet bij deze eerste UI-stap.

## Bekende aandachtspunten voor v0.36.1

1. Project Adviseur gebruikt nog steeds een uploadknop voor de wegassen binnen
   het scherm. Later kan dit centraler in de zijbalk of in een databronblok.
2. De statusafleiding is een presentatielaag. Als uit N398/N354 blijkt dat een
   status structureel verkeerd wordt geïnterpreteerd, moet eerst worden bepaald
   of het presentatie of engine is.
3. De oude NWB-referentieproef bevat nog veel overlappende tabellen. Voor v0.36.0
   is dat bewust niet verwijderd.
4. Kaartinspectie gebruikt de bestaande hoofdkaart. Een later detailkaartje per
   voorstel kan nuttig zijn, maar is nu niet toegevoegd om de stap klein te
   houden.

## Advies volgende stap

Eerst v0.36.0 in Streamlit testen met N398.

Als N398 rustig en bruikbaar is:

```text
v0.36.1 — N398-kalibratie van statuspresentatie
```

Als N398 niet rustig is, onderscheid maken tussen:

- presentatieprobleem: Project Adviseur-statussen aanpassen;
- engineprobleem: pas dan gericht `project_axis.py` aanpassen.

Daarna pas N354 als praktijktoets.
