# Overdracht na v0.36.4

## Status

Project Adviseur heeft nu een werkende concept-export in N-wegendocument-format
waarvan de knip- en verhardingswaarden vergelijkbaar zijn met het handmatige
N-wegendocument.

v0.36.3 bewees dat de export technisch gemaakt kon worden. v0.36.4 corrigeert de
belangrijkste inhoudelijke fout in die export: relatieve projectas-meters worden
niet meer als N-wegendocumentknip gebruikt.

## Belangrijkste correctie

De volgende kolommen worden nu gevuld vanuit de administratieve hectometrering:

```text
knip (begin)
knip (einde)
verharding (begin)
verharding (einde)
```

De bron is primair:

```text
fysiek_begin_km
fysiek_eind_km
```

De export zet deze om naar meterwaarden:

```text
25.800 km → 25800
26.300 km → 26300
```

Alleen wanneer deze km-waarden ontbreken, gebruikt de export de oude relatieve
meterwaarden als fallback. Dat houdt de export robuust voor oudere of onvolledige
tussenbestanden.

## Wat hiermee is opgelost

Het concepttabblad kan nu beter naast de bestaande handmatige N-wegentabbladen
worden gelegd. Voor N354 worden regels zoals `N354-HRB-25.8-26.3` niet meer met
relatieve waarden zoals `0–524` geëxporteerd, maar met N-wegendocumentwaarden
rond `25800–26300`.

## Wat nog niet af is

- Het bestand blijft een concept-export; het echte N-wegendocument wordt niet
  automatisch bijgewerkt.
- Locatie en documentatie blijven grotendeels handmatig.
- De inhoudelijke vergelijking tussen handmatig N354 en toolvoorstel moet nog
  beoordeeld worden.
- De werklijst voor N354 is nog groot; de concept-export is daarom belangrijker
  dan losse werklijstcontrole.
- Een latere versie kan alle Project Adviseur-output bundelen in één netter
  `Projectadvies_<weg>.xlsx`.

## Testaanpak vanaf nu

De gebruiker hoeft geen specifieke regels handmatig op te zoeken. Testen gebeurt
via exports:

1. Draai Project Adviseur voor N354.
2. Download:
   - `Projectadvies_Runrapport_N354.csv`
   - `Projectadvies_Voorstellen_N354.csv`
   - `Projectadvies_Werklijst_N354.csv`
   - `Projectadvies_Nwegendocument_N354.xlsx`
3. Upload deze bestanden.
4. Beoordeel automatisch:
   - of het runrapport bruikbaar is;
   - of het concept-N-wegendocument de juiste tabbladen heeft;
   - of knipwaarden in hectometreringsmeters staan;
   - of HRB/PW/FP logisch gevuld zijn;
   - of micro-/eindzone en buiten-ijkbereik herkenbaar blijven.

## Volgende logische stap

Na v0.36.4 is de technische correctie van de N-wegendocument-export afgerond.
De volgende stap is beoordelen of het concepttabblad voor N354 functioneel
bruikbaar is als vergelijkingslaag naast het handmatig gemaakte N-wegendocument.
