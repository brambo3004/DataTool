# Overdracht na v0.34.4

## Stand van zaken

v0.34.4 is de actuele versie. De app bevat nu een bruikbare controlelaag voor **Projectgrenzen op referentieas** binnen de NWB-referentieproef.

De belangrijkste functionele lijn is:

1. iASSET-wegas blijft leidend.
2. NWB-hectopunten ijken de iASSET-wegas naar hectometrering.
3. Onderhoudsprojectnamen worden diagnostisch op deze as gecontroleerd.
4. Oranje/rode NWB-afwijkingszones blijven aandachtspunten.
5. Dekking, gaten en overlap worden per projecttype beoordeeld.
6. v0.34.3 introduceerde snap-tolerantie van 2,5 meter en naamzone-logica.
7. v0.34.4 voegt een compacte controlelijst toe voor databeheer.

## Nieuwe export

Naast de bestaande exports is toegevoegd:

```text
Projectcontrole_Referentieas_<weg>.csv
```

Deze export is bedoeld als praktische werklijst. De volledige exports blijven beschikbaar voor detailcontrole:

```text
Projectgrenzen_Referentieas_<weg>.csv
Projectdekking_Referentieas_<weg>.csv
Projectobjecten_Referentieas_<weg>.csv
NWB_Hectopunten_op_iASSET_Wegas_<weg>.csv
```

## Testverwachting

Voor N398:
- geen gaten;
- geen overlap;
- meestal `ok`;
- bekende controle bij het einde kan blijven bestaan als eind buiten ijkbereik valt.

Voor N354:
- geen valse gatmeldingen uit v0.34.2;
- `BBLR` blijft toegestaan;
- bekende overlapkandidaten blijven zichtbaar;
- eindgebied blijft controlepunt wanneer het buiten ijkbereik valt.

## Belangrijke ontwerpkeuzes

- Objectligging blijft detaildiagnose en bepaalt niet zelfstandig de hoofdstatus.
- Compacte controlelijst bevat alleen actiegerichte regels.
- De tool wijzigt niets in iASSET.
- De tool geeft nog geen beheerknip- of projectnaamvoorstellen.

## Mogelijke volgende stap v0.34.5

Als v0.34.4 goed test, ligt de volgende stap niet in nieuwe rekenlogica maar in presentatie en acceptatie:
- kolomnamen in exports verder verduidelijken;
- optioneel een Excel-rapport met tabbladen maken;
- toelichtende legenda toevoegen voor `ok`, `aandacht`, `controleer`;
- handmatige acceptatiecheck met databeheer Grijs documenteren.
