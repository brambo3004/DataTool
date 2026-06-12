# Overdracht na v0.35.0

## Stand van zaken

v0.35.0 bouwt voort op de stabiele referentieasdiagnose uit v0.34.5.

Nieuw is een eerste proeflaag voor **Onderhoudsprojectvoorstellen vanaf nul**. Deze laag reconstrueert projectsegmenten vanuit primaire objecten op de geijkte iASSET-referentieas, zonder de bestaande onderhoudsprojectnaam als uitgangspunt te nemen.

## Belangrijk uitgangspunt

iASSET blijft de bron van waarheid.

De tool:

- wijzigt niets in iASSET;
- maakt geen beheerknippen;
- voert geen automatische mutaties uit;
- toont alleen diagnose, voorstellen en vergelijkingen.

## Wat de nieuwe v0.35-laag doet

Per geselecteerde weg:

1. primaire objecten projecteren op de geijkte iASSET-as;
2. objecten per spoor sorteren, bijvoorbeeld HRB/PWR/PWL/FPR/FPL/BBLR;
3. segmenten maken op basis van fysieke ligging en beheerkenmerken;
4. voorgestelde onderhoudsprojectnamen afleiden via de snap- en naamregel;
5. achteraf vergelijken met bestaande iASSET-onderhoudsprojecten.

## Nieuwe exports

```text
Projectvoorstellen_Referentieas_<weg>.csv
Projectvoorstel_Objecten_<weg>.csv
Projectvoorstel_Vergelijking_iASSET_<weg>.csv
```

De bestaande exports blijven bestaan:

```text
Projectgrenzen_Referentieas_<weg>.csv
Projectdekking_Referentieas_<weg>.csv
Projectobjecten_Referentieas_<weg>.csv
Projectcontrole_Referentieas_<weg>.csv
NWB_Hectopunten_op_iASSET_Wegas_<weg>.csv
```

## Testadvies

Gebruik opnieuw N398 en N354.

Voor N398 verwachten we:

- een rustige projectgrensdiagnose;
- weinig verrassende projectvoorstellen;
- eventuele verschillen vooral bij de bekende begin/eindzone.

Voor N354 verwachten we:

- voorstellen per projecttype/spoor;
- BBLR als geldige LR-situering;
- vergelijking waarin bestaande overlap/splitsing zichtbaar kan worden;
- geen automatische correcties.

## Aandacht voor vervolg

v0.35.0 is bewust een eerste proef. Mogelijke vervolgstappen:

- knipregels verfijnen met domeininput;
- rotondes/ovondes/kruispunten explicieter als eigen logische segmenten herkennen;
- objecten met ontbrekende beheerkenmerken apart markeren;
- voorstel-export beter geschikt maken als databeheerwerklijst;
- later eventueel richting v0.36: beheergrensadvies, nog steeds zonder automatische mutatie.
