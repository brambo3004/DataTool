"""
Centrale configuratie voor de iASSET Advisor.

Alle spelregels die in meerdere modules nodig zijn, staan hier bij elkaar.
Daardoor hoeven we in de rest van de code geen losse lijsten of kolomnamen
te kopiëren.
"""

from pathlib import Path

# --- Bestanden -------------------------------------------------------------

# Versie die in de Streamlit-interface zichtbaar wordt getoond.
APP_VERSION = "v0.14.2"

# Versie van het sorteerdiagnose-frame in session_state.
# Waarom apart?
# Streamlit kan session_state bewaren terwijl de code al is bijgewerkt.
# Zonder deze sleutel kan de app na een deploy nog een oude diagnose-tabel
# exporteren, terwijl het versienummer al nieuw toont.
SORT_DIAGNOSTICS_SCHEMA_VERSION = "sortdiag-v0.14.2"

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

# Objecten met deze subthema's krijgen volgens het werkproces géén
# onderhoudsproject. Let op: "geleideconstructie" stond in de oude app, maar
# staat niet in het werkproces als uitzondering. Voeg die alleen terug toe als
# dat inhoudelijk alsnog wordt bevestigd.
SUBTHEMA_EXCEPTIONS = [
    "carpoolplaats",
    "fietsstalling",
    "parkeerplaats",
    "perron",
    "picknickplaats",
    "rotonderand",
    "wegas koppeling dielplak",
    "opstelplaats",
    "verkeerseiland of middengeleider",
    "verharding derden",
]

# Sommige uitzonderingen zijn in iASSET niet altijd zuiver als subthema vastgelegd,
# maar als tekstwaarde in een ander paspoortveld. Deze herkenningswoorden worden
# daarom rijbreed gezocht. We houden de lijst bewust specifiek om vals-positieve
# matches te voorkomen.
MAINTENANCE_PROJECT_EXEMPTION_MARKERS = [
    "oorspronkelijke bgt-data",
    "oorspronkelijke bgt data",
    "oorspronkelijke bgt",
]

BACKBONE_TYPES = ["rijstrook", "parallelweg", "landbouwpad", "busbaan", "fietspad"]

HIERARCHY_CONFIG = [
    {"rank": 1, "types": ["rijstrook"], "prefix": "GRP_RIJBAAN"},
    {"rank": 2, "types": ["parallelweg", "landbouwpad", "busbaan"], "prefix": "GRP_PARALLEL"},
    {"rank": 3, "types": ["fietspad"], "prefix": "GRP_FIETSPAD"},
]

# Standaard issuecategorieën voor het tabblad Data Kwaliteit.
# De volgorde is de volgorde waarin de gebruiker ze in de filter ziet.
ISSUE_CATEGORIES = [
    "Onderhoudsprojectplicht",
    "Onterecht onderhoudsproject",
    "Ontbrekende paspoortdata",
    "Liggingdata",
    "Topologie",
    "Projectconsistentie",
    "Import / geometrie",
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
    "Jaar herstrating",
    "Jaar conservering",
    "Wegvaknum",
    "Soort verharding_N",
    "naam",
]


# --- Overzicht-tabblad -----------------------------------------------------

# Labels zoals de gebruiker ze in de app ziet, met daarachter mogelijke
# kolomnamen uit verschillende iASSET-/voorbeeldexports.
OVERVIEW_ATTRIBUTE_ALIASES = {
    "Jaar aanleg": ["Jaar aanleg"],
    "Jaar deklaag": ["Jaar deklaag"],
    "Jaar herstrating": ["Jaar herstrating"],
    "Jaar conservering": ["Jaar conservering"],
    "Besteknummer": ["Besteknummer"],
    "Onderhoudsproject": ["Onderhoudsproject"],
    "Wegvaknum": ["Wegvaknum", "Wegvaknum V", "Wegvaknum G"],
    "Soort verharding_N": ["Soort verharding_N", "verhardingssoort"],
    "Soort deklaag specifiek": ["Soort deklaag specifiek"],
}

# Popupvelden voor de Overzichtkaart. Ook hier gebruiken we aliases, omdat
# bronexports niet altijd exact dezelfde kolomkoppen hebben.
OVERVIEW_POPUP_COLUMNS = {
    "Jaar aanleg": ["Jaar aanleg"],
    "Jaar deklaag": ["Jaar deklaag"],
    "Soort verharding_N": ["Soort verharding_N", "verhardingssoort"],
    "Soort deklaag specifiek": ["Soort deklaag specifiek"],
    "Besteknummer": ["Besteknummer"],
    "Onderhoudsproject": ["Onderhoudsproject"],
    "Wegvaknum": ["Wegvaknum", "Wegvaknum V", "Wegvaknum G"],
}

# Exportprofielen sluiten aan op de manier waarop iASSET imports verwerkt:
# per importbestand wordt dezelfde kolommenset voor alle objecten meegeschreven.
DEFAULT_EXPORT_PROFILE = "Onderhoudsprojecten"

EXPORT_PROFILES = {
    "Onderhoudsprojecten": [
        "bron_id",
        "nummer",
        "Onderhoudsproject",
    ],
    "Paspoortdata basis": [
        "bron_id",
        "nummer",
        "subthema",
        "naam",
        "Gebruikersfunctie",
        "Type onderdeel",
        "verhardingssoort",
        "Soort verharding_N",
        "Soort deklaag specifiek",
        "Soort conservering",
        "Jaar aanleg",
        "Jaar deklaag",
        "Jaar conservering",
        "Jaar herstrating",
        "Besteknummer",
        "Onderhoudsproject",
    ],
    "Liggingdata": [
        "bron_id",
        "nummer",
        "Wegnummer",
        "Wegvak",
        "Wegvak V",
        "Wegvak G",
        "Wegvaknum",
        "Wegvaknum V",
        "Wegvaknum G",
        "Metrering",
        "Metrering V",
        "Metrering G",
        "Situering",
        "Situering V",
        "Bebouwde_kom",
        "gps coordinaten",
        "rds coordinaten",
    ],
    "Volledige gecontroleerde mutatieset": [
        "bron_id",
        "nummer",
        "Wegnummer",
        "Wegvak",
        "Wegvak V",
        "Wegvak G",
        "Wegvaknum",
        "Wegvaknum V",
        "Wegvaknum G",
        "Metrering",
        "Metrering V",
        "Metrering G",
        "Situering",
        "Situering V",
        "Bebouwde_kom",
        "subthema",
        "naam",
        "Gebruikersfunctie",
        "Type onderdeel",
        "verhardingssoort",
        "Soort verharding_N",
        "Soort deklaag specifiek",
        "Soort conservering",
        "Jaar aanleg",
        "Jaar deklaag",
        "Jaar conservering",
        "Jaar herstrating",
        "Besteknummer",
        "Onderhoudsproject",
        "gps coordinaten",
        "rds coordinaten",
    ],
}

# Achterwaartse compatibiliteit voor bestaande tests/functies die nog geen
# profielnaam meegeven.
EXPORT_COLUMNS = EXPORT_PROFILES[DEFAULT_EXPORT_PROFILE]

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
