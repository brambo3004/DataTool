# Release notes v0.34.1

## Type release

Stabilisatieversie van **Projectgrenzen op referentieas**.

Deze release voegt geen automatische mutaties of beheerknippen toe. De tool blijft alleen diagnose/proef. iASSET blijft de bron van waarheid.

## Belangrijkste wijzigingen

### 1. Projectdekking per projecttype

v0.34.0 controleerde dekking, gaten en overlap op één gezamenlijke referentieas. Daardoor kon bijvoorbeeld `HRB` overlappen met `PWR` of `FPR`.

v0.34.1 controleert per spoor:

- `HRB` met `HRB`;
- `PWR` met `PWR`;
- `PWL` met `PWL`;
- `FPR` met `FPR`;
- `FPL` met `FPL`;
- enzovoort.

Hiermee verdwijnen de meeste valse overlapmeldingen bij parallelwegen en fietspaden.

### 2. Onderhoudsprojectnaam-validatie

De projectgrenstabel bevat nu aparte kolommen voor naamvalidatie:

- `naam_wegnummer`
- `project_type`
- `project_family`
- `situering`
- `naam_begin_label`
- `naam_eind_label`
- `naam_validatie_status`
- `naam_validatie_melding`
- `status_projectnaam`

De validatie controleert onder meer:

- patroon `Nxxx-type-begin-einde`;
- koppeltekens;
- exact één cijfer na de punt;
- voorloopnul bij metrering onder 10;
- begin < einde;
- projecttype/familie;
- L/R-situering bij parallelweg, fietspad, busbaan en landbouwpad.

### 3. Beheerregel voor projectnaam-hectometrering vastgelegd

De bestaande centrale functie `format_name_hm()` blijft leidend voor naamvorming:

- 12.300 -> 12.3
- 12.301 -> 12.4
- 12.350 -> 12.4
- 12.399 -> 12.4
- 12.400 -> 12.4

Deze regel wordt nu ook zichtbaar gebruikt in de projectgrensdiagnose via:

- `object_begin_naamregel`
- `object_eind_naamregel`

Deze kolommen zijn nog diagnostisch. De tool stelt nog geen nieuwe onderhoudsprojectnamen automatisch voor.

### 4. Objectligging losgekoppeld van hoofdstatus

Objectligging blijft zichtbaar, maar bepaalt niet meer automatisch de hoofdstatus `status`.

Nieuwe kolommen:

- `objectligging_status`
- `objectligging_melding`
- `status_projectgrens`
- `status_projectnaam`
- `status`

De eindstatus `status` wordt nu primair bepaald door:

- projectnaamvalidatie;
- projectgrensstatus;
- ijkbereik;
- oranje/rode afwijkingszones;
- lengteverschil naam versus geijkte as.

Objectligging blijft context, vooral belangrijk omdat fietspaden en parallelwegen fysiek logisch verder van de hoofdas kunnen liggen.

### 5. Rustigere exports

Aangepast:

- `Projectgrenzen_Referentieas_<weg>.csv`
- `Projectdekking_Referentieas_<weg>.csv`

`Projectdekking_Referentieas_<weg>.csv` bevat nu onder meer:

- `project_type`
- `project_family`
- `situering`
- `projectbereik_m`

Er worden geen begin/eind-gaten richting volledige ijkingsrange meer gemeld per parallelweg/fietspad, omdat dat meestal ruis is.

## Tests

De volledige testset is gedraaid:

```text
172 passed
```

Nieuw toegevoegd aan `tests/test_project_axis.py`:

- overlap tussen HRB en FPR wordt niet meer gemeld;
- overlap binnen HRB blijft zichtbaar;
- projectnaamvalidatie signaleert ontbrekende voorloopnul;
- objectligging overschreeuwt projectgrensstatus niet;
- naamregel rondt fysieke objectligging naar boven af op hectometerpunt.

## Bekende beperkingen

- De diagnose gebruikt nog steeds één iASSET-referentieas per asdeel; parallelwegen en fietspaden worden projectmatig gescheiden via projecttype, niet via eigen fysieke assen.
- De tool doet nog geen beheergrensvoorstellen.
- De tool stelt nog geen nieuwe onderhoudsprojectnamen voor.
- De tool wijzigt niets in iASSET.
