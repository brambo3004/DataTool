# Overdracht na v0.37.2

v0.37.2 is een hotfix op v0.37.1.

## Aanleiding

Bij het draaien van Project Adviseur gaf Streamlit onderaan het scherm een foutmelding:

```text
StreamlitDuplicateElementId
```

De fout ontstond doordat de downloadknop voor de zichtbare onderhoudscomplexlaag twee keer in dezelfde Streamlit-run kon voorkomen.

## Oplossing

De downloadknoppen in het Project Adviseur-hoofdblok hebben nu expliciete keys, waaronder:

```text
download_visible_complexes_inline_<weg>
download_visible_complexes_exports_<weg>
download_project_advisor_runreport_inline_<weg>
download_project_advisor_runreport_exports_<weg>
download_project_advisor_nwegendocument_<weg>
download_project_advisor_calibration_<weg>
```

## Belangrijk

Deze versie wijzigt alleen de Streamlit-presentatielaag. De rekenmotor, projectvoorstellen, zichtbare onderhoudscomplexlaag en exports blijven inhoudelijk gelijk aan v0.37.1.

## Volgende stap

Draai N354 opnieuw en download:

```text
Projectadvies_Zichtbare_Onderhoudscomplexen_N354.csv
Projectadvies_Nwegendocument_N354.xlsx
Projectadvies_Runrapport_N354.csv
Projectadvies_Voorstellen_N354.csv
```

Gebruik die bestanden om te beoordelen of de zichtbare onderhoudscomplexlaag inhoudelijk goed werkt.
