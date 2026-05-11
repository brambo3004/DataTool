\
"""
Centrale configuratie voor de iASSET Advisor.

Alle spelregels die in meerdere modules nodig zijn, staan hier bij elkaar.
Daardoor hoeven we in de rest van de code geen losse lijsten of kolomnamen
te kopiëren.
"""

from pathlib import Path

# --- Bestanden -------------------------------------------------------------

DATA_DIR = Path(".")
FILE_NIET_RIJSTROOK = DATA_DIR / "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = DATA_DIR / "N-allemaal-alleen-rijstrook.csv"
AUTOSAVE_FILE = DATA_DIR / "autosave_log.csv"

INPUT_FILES = (FILE_NIET_RIJSTROOK, FILE_WEL_RIJSTROOK)

# --- Domeinregels ----------------------------------------------------------

# Rangorde bij het koppelen van secundaire objecten.
# Lager getal = belangrijker in de hiërarchie.
HIERARCHY_RANK = {
    "rijstrook": 1,
    "parallelweg": 2,
    "landbouwpad": 2,
    "busbaan": 2,
    "fietspad": 3,
}

# Let op: dit is bewust dezelfde lijst als in de bestaande app.
# In het werkprocesdocument staat een bredere uitzonderingenlijst.
# Die inhoudelijke correctie pakken we na de structurele refactor op.
SUBTHEMA_EXCEPTIONS = [
    "fietsstalling",
    "geleideconstructie",
    "parkeerplaats",
    "rotonderand",
    "verkeerseiland of middengeleider",
]

BACKBONE_TYPES = ["rijstrook", "parallelweg", "landbouwpad", "busbaan", "fietspad"]

HIERARCHY_CONFIG = [
    {"rank": 1, "types": ["rijstrook"], "prefix": "GRP_RIJBAAN"},
    {"rank": 2, "types": ["parallelweg", "landbouwpad", "busbaan"], "prefix": "GRP_PARALLEL"},
    {"rank": 3, "types": ["fietspad"], "prefix": "GRP_FIETSPAD"},
]

# Eigenschappen waarop de Project Adviseur de ruggengraat mag knippen.
SEGMENTATION_ATTRIBUTES = [
    "verhardingssoort",
    "Soort deklaag specifiek",
    "Jaar aanleg",
    "Jaar deklaag",
    "Besteknummer",
]

MUTATION_REQUIRED_COLS = [
    "subthema",
    "naam",
    "Gebruikersfunctie",
    "Type onderdeel",
    "verhardingssoort",
    "Onderhoudsproject",
]

FRIENDLY_LABELS = {
    "verhardingssoort": "Verhardingssoort",
    "Soort deklaag specifiek": "Deklaag",
    "Jaar aanleg": "Aanleg",
    "Jaar deklaag": "Deklaagjaar",
    "Besteknummer": "Bestek",
    "Onderhoudsproject": "Huidig Project",
}

# Kolommen die de app verwacht voor kaart, popup, analyse en export.
ALL_META_COLS = [
    "subthema",
    "Situering",
    "verhardingssoort",
    "Soort deklaag specifiek",
    "Jaar aanleg",
    "Jaar deklaag",
    "Onderhoudsproject",
    "Advies_Onderhoudsproject",
    "validation_error",
    "Spoor_ID",
    "Is_Project_Grens",
    "Advies_Bron",
    "Wegnummer",
    "Besteknummer",
    "tijdstipRegistratie",
    "nummer",
    "gps coordinaten",
    "rds coordinaten",
    "Metrering",
]

EXPORT_COLUMNS = [
    "bron_id",
    "nummer",
    "Wegnummer",
    "subthema",
    "Onderhoudsproject",
    "verhardingssoort",
    "Soort deklaag specifiek",
    "Jaar aanleg",
    "Jaar deklaag",
    "Besteknummer",
    "gps coordinaten",
    "rds coordinaten",
]

# --- Wegsortering ----------------------------------------------------------

# NTS = Noord -> Zuid, STN = Zuid -> Noord,
# WTE = West -> Oost, ETW = Oost -> West.
ROAD_DIRECTIONS = {
    "N351": "ETW",
    "N353": "STN",
    "N354": "ETW",
    "N355": "WTE",
    "N356": "NTS",
    "N357": "STN",
    "N358": "WTE",
    "N359": "ETW",
    "N361": "ETW",
    "N369": "STN",
    "N380": "WTE",
    "N381": "NTS",
    "N383": "STN",
    "N384": "STN",
    "N390": "ETW",
    "N392": "WTE",
    "N393": "ETW",
    "N398": "ETW",
    "N910": "WTE",
    "N913": "ETW",
    "N917": "WTE",
    "N918": "STN",
    "N919": "ETW",
    "N924": "ETW",
    "N927": "ETW",
    "N928": "NTS",
}
