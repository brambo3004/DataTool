# Release notes v0.35.1

## Doel

v0.35.1 is een verfijning van de groenveld-projectvoorstellen uit v0.35.0.

De richting blijft hetzelfde: de tool bouwt onderhoudsprojectvoorstellen vanaf nul op uit primaire iASSET-objecten op de geijkte iASSET-referentieas. Bestaande iASSET-onderhoudsprojectnamen worden alleen achteraf gebruikt als vergelijking.

## Belangrijkste wijziging

v0.35.1 introduceert **knipzwaarte**.

In v0.35.0 werd vrijwel elk verschil in beheerkenmerken een nieuw projectvoorstel. Dat gaf vooral bij N354 te veel korte segmenten. In v0.35.1 wordt onderscheid gemaakt tussen:

- harde knippen: sterke signalen die een nieuw voorstel kunnen starten;
- zachte signalen: lokale of korte verschillen die binnen het voorstel zichtbaar blijven als context.

## Nieuwe kolommen

`Projectvoorstellen_Referentieas_<weg>.csv` bevat nu extra kolommen:

- `knipprofiel`
- `harde_knipsignalen`
- `zachte_signalen`

## Functioneel effect

- Minder korte detailvoorstellen.
- Meer robuuste trajecten.
- Kleine verschillen in bijvoorbeeld jaar aanleg of lokale deklaagverschillen worden niet automatisch een zelfstandig onderhoudsproject.
- De zachte verschillen blijven wel zichtbaar, zodat databeheer ze kan controleren.

## Niet gewijzigd

- Geen automatische mutaties in iASSET.
- Geen automatische beheerknippen.
- Geen automatische objectverplaatsingen.
- De projectgrenscontrole van v0.34.x blijft intact.
- De compacte controle-export van v0.34.5 blijft intact.

## Validatie

- `183 passed`
- `python -m py_compile app.py iasset_tool/project_axis.py`
