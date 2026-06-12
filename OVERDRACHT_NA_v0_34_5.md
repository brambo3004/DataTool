# Overdracht na v0.34.5

## Stand van zaken

v0.34.5 is de actuele versie. De tool bevat nu een bruikbare en rustiger controlelaag voor **Projectgrenzen op referentieas** binnen de NWB-referentieproef.

De rekenkundige kern is hetzelfde als v0.34.3/v0.34.4:
- iASSET-wegas blijft leidend;
- NWB-hectopunten ijken de iASSET-wegas naar hectometrering;
- projectgrenzen worden diagnostisch gecontroleerd;
- snap-tolerantie naar hectometerpunt staat standaard op 2,5 meter;
- de hectometer-naar-boven-regel wordt toegepast;
- projectdekking wordt per projecttype beoordeeld;
- valse gaten door projectnaamzones worden onderdrukt;
- `BBLR` wordt voorlopig geaccepteerd als gecombineerde `LR`-situering.

## Belangrijkste wijziging v0.34.5

De compacte controlelijst is opgeschoond.

Bestand:

```text
Projectcontrole_Referentieas_<weg>.csv
```

Nieuwe/gewijzigde meldingskolommen:

```text
hoofdmelding
contextmelding
melding
```

Gebruik:
- `hoofdmelding` als hoofdreden voor handmatige controle;
- `contextmelding` voor extra details, bijvoorbeeld objectligging;
- `melding` blijft beschikbaar voor herkenbaarheid en is gelijk aan de schone hoofdreden.

## Testverwachting

Voor N398:
- compacte controlelijst blijft klein;
- geen gaten;
- geen overlap;
- bekende aandacht/controleer-regels blijven zichtbaar.

Voor N354:
- compacte controlelijst bevat de 5 projectgrensregels en 6 overlapregels;
- geen gatregels;
- `BBLR` blijft `ok`;
- objectligging staat hooguit als context, niet als zelfstandige hoofdreden.

## Volgende logische stap

Als v0.34.5 goed test, is de projectgrensdiagnose stabiel genoeg om richting v0.35 te denken.

Voor v0.35 past een eerste proef met **projectnaamvoorstellen als diagnose**:
- nog steeds geen automatische iASSET-mutaties;
- nog geen beheerknippen;
- alleen voorstellen tonen op basis van fysieke objectligging, snap-tolerantie en hectometer-naar-boven-regel;
- met duidelijke waarschuwing als de grens in oranje/rode afwijkingszones of buiten ijkbereik ligt.
