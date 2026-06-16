"""
Centrale configuratie voor de iASSET Advisor.

Vanaf v0.26 worden de belangrijkste beheerregels geladen uit
``config/onderhoudsregels.json``. De Python-code behoudt dezelfde constanten
voor achterwaartse compatibiliteit, maar de inhoud komt uit één leesbaar
configuratiebestand.

Waarom zo?
- Databeheerregels veranderen soms sneller dan applicatiecode.
- Project Adviseur, Onderhoudscontrole en Datakwaliteit moeten dezelfde regels
  gebruiken.
- Een kapot of ontbrekend configuratiebestand mag de tool niet laten crashen:
  dan valt de app veilig terug op de ingebouwde standaardregels.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


# --- Bestanden -------------------------------------------------------------

# Versie die in de Streamlit-interface zichtbaar wordt getoond.
APP_VERSION = "v0.35.3"

# Versie van het sorteerdiagnose-frame in session_state.
# Waarom apart?
# Streamlit kan session_state bewaren terwijl de code al is bijgewerkt.
# Zonder deze sleutel kan de app na een deploy nog een oude diagnose-tabel
# exporteren, terwijl het versienummer al nieuw toont.
SORT_DIAGNOSTICS_SCHEMA_VERSION = "sortdiag-v0.21.0"

DATA_DIR = Path(".")
FILE_NIET_RIJSTROOK = DATA_DIR / "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = DATA_DIR / "N-allemaal-alleen-rijstrook.csv"
AUTOSAVE_FILE = DATA_DIR / "autosave_log.csv"

INPUT_FILES = (FILE_NIET_RIJSTROOK, FILE_WEL_RIJSTROOK)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"
MAINTENANCE_RULES_PATH = CONFIG_DIR / "onderhoudsregels.json"


DEFAULT_MAINTENANCE_RULES: dict[str, Any] = {
    "metadata": {
        "naam": "onderhoudsregels",
        "versie": "v0.26.0",
        "toelichting": (
            "Beheerregels voor de iASSET-tool. De tool gebruikt deze regels "
            "alleen voor controle, sortering, uitleg en exportvoorstellen; nooit "
            "voor automatische mutaties in iASSET."
        ),
    },
    "domeinregels": {
        # Rangorde bij het koppelen van secundaire objecten.
        # Lager getal = belangrijker in de hiërarchie.
        "hierarchy_rank": {
            "rijstrook": 1,
            "parallelweg": 2,
            "landbouwpad": 2,
            "busbaan": 2,
            "fietspad": 3,
        },
        # Objecten met deze subthema's krijgen volgens het werkproces géén
        # onderhoudsproject. Let op: "geleideconstructie" stond in de oude app,
        # maar staat niet in het werkproces als uitzondering.
        "subthema_exceptions": [
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
        ],
        # Sommige uitzonderingen zijn in iASSET niet altijd zuiver als subthema
        # vastgelegd, maar als tekstwaarde in een ander paspoortveld.
        "maintenance_project_exemption_markers": [
            "oorspronkelijke bgt-data",
            "oorspronkelijke bgt data",
            "oorspronkelijke bgt",
        ],
        "backbone_types": ["rijstrook", "parallelweg", "landbouwpad", "busbaan", "fietspad"],
        "hierarchy_config": [
            {"rank": 1, "types": ["rijstrook"], "prefix": "GRP_RIJBAAN"},
            {"rank": 2, "types": ["parallelweg", "landbouwpad", "busbaan"], "prefix": "GRP_PARALLEL"},
            {"rank": 3, "types": ["fietspad"], "prefix": "GRP_FIETSPAD"},
        ],
        "project_category_families": {
            "HRB": ["HRB", "HRBR", "HRBL"],
            "PW": ["PW", "PWR", "PWL"],
            "FP": ["FP", "FPR", "FPL"],
            "LBP": ["LBP", "LBPR", "LBPL"],
            "BB": ["BB", "BBR", "BBL"],
        },
    },
    "datakwaliteit": {
        "issue_categories": [
            "Onderhoudsprojectplicht",
            "Onterecht onderhoudsproject",
            "Ontbrekende paspoortdata",
            "Liggingdata",
            "Topologie",
            "Projectconsistentie",
            "Import / geometrie",
        ],
        "mutation_required_cols": [
            "subthema",
            "naam",
            "Gebruikersfunctie",
            "Type onderdeel",
            "verhardingssoort",
            "Onderhoudsproject",
        ],
    },
    "project_adviseur": {
        # Eigenschappen waarop de Project Adviseur de ruggengraat mag knippen.
        "segmentation_attributes": [
            "verhardingssoort",
            "Soort deklaag specifiek",
            "Jaar aanleg",
            "Jaar deklaag",
            "Besteknummer",
        ],
        "friendly_labels": {
            "verhardingssoort": "Verhardingssoort",
            "Soort deklaag specifiek": "Deklaag",
            "Jaar aanleg": "Aanleg",
            "Jaar deklaag": "Deklaagjaar",
            "Besteknummer": "Bestek",
            "Onderhoudsproject": "Huidig Project",
        },
    },
    "kolommen": {
        "all_meta_cols": [
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
        ],
    },
    "overzicht": {
        "overview_attribute_aliases": {
            "Jaar aanleg": ["Jaar aanleg"],
            "Jaar deklaag": ["Jaar deklaag"],
            "Jaar herstrating": ["Jaar herstrating"],
            "Jaar conservering": ["Jaar conservering"],
            "Besteknummer": ["Besteknummer"],
            "Onderhoudsproject": ["Onderhoudsproject"],
            "Wegvaknum": ["Wegvaknum", "Wegvaknum V", "Wegvaknum G"],
            "Soort verharding_N": ["Soort verharding_N", "verhardingssoort"],
            "Soort deklaag specifiek": ["Soort deklaag specifiek"],
        },
        "overview_popup_columns": {
            "Jaar aanleg": ["Jaar aanleg"],
            "Jaar deklaag": ["Jaar deklaag"],
            "Soort verharding_N": ["Soort verharding_N", "verhardingssoort"],
            "Soort deklaag specifiek": ["Soort deklaag specifiek"],
            "Besteknummer": ["Besteknummer"],
            "Onderhoudsproject": ["Onderhoudsproject"],
            "Wegvaknum": ["Wegvaknum", "Wegvaknum V", "Wegvaknum G"],
        },
    },
    "export": {
        "default_export_profile": "Onderhoudsprojecten",
        "export_profiles": {
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
        },
    },
    "wegsortering": {
        # NTS = Noord -> Zuid, STN = Zuid -> Noord,
        # WTE = West -> Oost, ETW = Oost -> West.
        "road_directions": {
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
        },
    },
}


def _as_list(value: Any) -> list[Any]:
    """Geef een veilige lijst terug; ongeldige waarden worden een lege lijst."""
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    """Geef een veilige dict terug; ongeldige waarden worden een lege dict."""
    return value if isinstance(value, dict) else {}


def _deep_merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Voeg een gebruikersconfiguratie veilig samen met de standaardregels.

    Alleen dictionaries worden recursief samengevoegd. Lijsten en losse waarden
    vervangen de standaardwaarde volledig. Dat maakt het gedrag voorspelbaar:
    als de databeheerder een uitzonderingenlijst beheert, is die lijst leidend.
    """
    merged = deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_maintenance_rules(path: str | Path | None = None) -> tuple[dict[str, Any], list[str], str]:
    """
    Laad de onderhoudsregels uit JSON met veilige fallback.

    Parameters
    ----------
    path:
        Optioneel pad naar een configuratiebestand. Zonder pad gebruikt de tool
        ``config/onderhoudsregels.json`` naast de applicatie.

    Returns
    -------
    tuple
        ``(regels, waarschuwingen, bron)``. De bron is ``configuratiebestand``
        of ``ingebouwde standaard``.
    """
    config_path = Path(path) if path is not None else MAINTENANCE_RULES_PATH
    warnings: list[str] = []

    if not config_path.exists():
        if path is not None:
            warnings.append(f"Beheerregelsbestand niet gevonden: {config_path}. Ingebouwde standaard gebruikt.")
        return deepcopy(DEFAULT_MAINTENANCE_RULES), warnings, "ingebouwde standaard"

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception as exc:  # pragma: no cover - exacte JSON-fout verschilt per Python-versie
        warnings.append(
            f"Beheerregelsbestand kon niet worden gelezen ({type(exc).__name__}: {exc}). "
            "Ingebouwde standaard gebruikt."
        )
        return deepcopy(DEFAULT_MAINTENANCE_RULES), warnings, "ingebouwde standaard"

    if not isinstance(loaded, dict):
        warnings.append("Beheerregelsbestand bevat geen JSON-object. Ingebouwde standaard gebruikt.")
        return deepcopy(DEFAULT_MAINTENANCE_RULES), warnings, "ingebouwde standaard"

    return _deep_merge(DEFAULT_MAINTENANCE_RULES, loaded), warnings, "configuratiebestand"


MAINTENANCE_RULES, MAINTENANCE_RULES_WARNINGS, MAINTENANCE_RULES_SOURCE = load_maintenance_rules()


def maintenance_rules_summary() -> dict[str, Any]:
    """
    Geef een compacte, UI-vriendelijke samenvatting van de geladen beheerregels.
    """
    domain = _as_dict(MAINTENANCE_RULES.get("domeinregels"))
    advisor = _as_dict(MAINTENANCE_RULES.get("project_adviseur"))
    export = _as_dict(MAINTENANCE_RULES.get("export"))
    roads = _as_dict(_as_dict(MAINTENANCE_RULES.get("wegsortering")).get("road_directions"))

    return {
        "bron": MAINTENANCE_RULES_SOURCE,
        "pad": str(MAINTENANCE_RULES_PATH),
        "waarschuwingen": list(MAINTENANCE_RULES_WARNINGS),
        "primaire_subthemas": len(_as_list(domain.get("backbone_types"))),
        "uitzonderingssubthemas": len(_as_list(domain.get("subthema_exceptions"))),
        "uitzonderingsmarkers": len(_as_list(domain.get("maintenance_project_exemption_markers"))),
        "categoriefamilies": len(_as_dict(domain.get("project_category_families"))),
        "knipvelden_project_adviseur": len(_as_list(advisor.get("segmentation_attributes"))),
        "exportprofielen": len(_as_dict(export.get("export_profiles"))),
        "wegrichtingen": len(roads),
    }


# --- Domeinregels ----------------------------------------------------------

_DOMAIN_RULES = _as_dict(MAINTENANCE_RULES.get("domeinregels"))

HIERARCHY_RANK = _as_dict(_DOMAIN_RULES.get("hierarchy_rank"))
SUBTHEMA_EXCEPTIONS = _as_list(_DOMAIN_RULES.get("subthema_exceptions"))
MAINTENANCE_PROJECT_EXEMPTION_MARKERS = _as_list(_DOMAIN_RULES.get("maintenance_project_exemption_markers"))
BACKBONE_TYPES = _as_list(_DOMAIN_RULES.get("backbone_types"))
HIERARCHY_CONFIG = _as_list(_DOMAIN_RULES.get("hierarchy_config"))
PROJECT_CATEGORY_FAMILIES = _as_dict(_DOMAIN_RULES.get("project_category_families"))


# --- Datakwaliteit ---------------------------------------------------------

_DATA_QUALITY_RULES = _as_dict(MAINTENANCE_RULES.get("datakwaliteit"))

ISSUE_CATEGORIES = _as_list(_DATA_QUALITY_RULES.get("issue_categories"))
MUTATION_REQUIRED_COLS = _as_list(_DATA_QUALITY_RULES.get("mutation_required_cols"))


# --- Project Adviseur ------------------------------------------------------

_ADVISOR_RULES = _as_dict(MAINTENANCE_RULES.get("project_adviseur"))

SEGMENTATION_ATTRIBUTES = _as_list(_ADVISOR_RULES.get("segmentation_attributes"))
FRIENDLY_LABELS = _as_dict(_ADVISOR_RULES.get("friendly_labels"))


# --- Kolommen --------------------------------------------------------------

_COLUMN_RULES = _as_dict(MAINTENANCE_RULES.get("kolommen"))

ALL_META_COLS = _as_list(_COLUMN_RULES.get("all_meta_cols"))


# --- Overzicht-tabblad -----------------------------------------------------

_OVERVIEW_RULES = _as_dict(MAINTENANCE_RULES.get("overzicht"))

OVERVIEW_ATTRIBUTE_ALIASES = _as_dict(_OVERVIEW_RULES.get("overview_attribute_aliases"))
OVERVIEW_POPUP_COLUMNS = _as_dict(_OVERVIEW_RULES.get("overview_popup_columns"))


# --- Export ---------------------------------------------------------------

_EXPORT_RULES = _as_dict(MAINTENANCE_RULES.get("export"))

DEFAULT_EXPORT_PROFILE = str(_EXPORT_RULES.get("default_export_profile") or "Onderhoudsprojecten")
EXPORT_PROFILES = _as_dict(_EXPORT_RULES.get("export_profiles"))

# Achterwaartse compatibiliteit voor bestaande tests/functies die nog geen
# profielnaam meegeven.
EXPORT_COLUMNS = EXPORT_PROFILES.get(DEFAULT_EXPORT_PROFILE, [])


# --- Wegsortering ----------------------------------------------------------

_SORTING_RULES = _as_dict(MAINTENANCE_RULES.get("wegsortering"))

ROAD_DIRECTIONS = _as_dict(_SORTING_RULES.get("road_directions"))
