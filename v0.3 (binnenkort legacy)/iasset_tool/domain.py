"""
Domeinpredicaten voor onderhoudsprojectlogica.

Deze module bevat kleine, herbruikbare vragen zoals:
- heeft dit object een onderhoudsproject nodig?
- is de onderhoudsprojectwaarde leeg?
- waarom is een object uitgezonderd?

Waarom apart?
Zowel de datakwaliteitsregels als de Project Adviseur moeten dezelfde
uitzonderingen gebruiken. Als die logica in twee bestanden staat, ontstaan er
snel verschillen tussen de werklijst en de adviesgroepen.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .config import MAINTENANCE_PROJECT_EXEMPTION_MARKERS, SUBTHEMA_EXCEPTIONS
from .utils import clean_display_value, normalize_text


_SKIP_MARKER_COLUMNS = {
    "geometry",
    "gps coordinaten",
    "rds coordinaten",
}


def normalize_domain_text(value: Any) -> str:
    """
    Normaliseer domeintekst voor robuuste vergelijkingen.

    We trekken hoofdletters, extra spaties en verschillende streepjes gelijk.
    Daardoor matchen waarden zoals "Oorspronkelijke BGT data" en
    "oorspronkelijke BGT-data" op dezelfde regel.
    """
    text = normalize_text(value)
    if not text:
        return ""

    # Verschillende streepjes uit Excel/Word gelijk trekken.
    text = text.replace("–", "-").replace("—", "-").replace("_", " ")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalized_subthema_exceptions() -> set[str]:
    """Geef de genormaliseerde subthema-uitzonderingen terug."""
    return {normalize_domain_text(value) for value in SUBTHEMA_EXCEPTIONS}


def _normalized_exemption_markers() -> set[str]:
    """Geef de genormaliseerde rijbrede herkenningswoorden terug."""
    return {normalize_domain_text(value) for value in MAINTENANCE_PROJECT_EXEMPTION_MARKERS}


def is_project_value_empty(value: Any) -> bool:
    """
    Controleer of Onderhoudsproject inhoudelijk leeg is.

    iASSET-export kan lege tekst, None, NaN, <NA> of de tekst "nan" bevatten.
    Voor de regels behandelen we die allemaal als leeg.
    """
    return clean_display_value(value) == ""


def subthema_is_maintenance_project_exempt(value: Any) -> bool:
    """
    Controleer of een subthema op de uitzonderingenlijst staat.
    """
    return normalize_domain_text(value) in _normalized_subthema_exceptions()


def find_exemption_marker(row: Mapping[str, Any] | Any) -> str:
    """
    Zoek rijbreed naar een specifieke uitzonderingsmarker.

    Dit is nodig voor waarden zoals "oorspronkelijke BGT-data". Die staan niet
    altijd als subthema in iASSET, maar kunnen in een ander paspoortveld staan.
    De zoekwoorden zijn bewust specifiek om vals-positieve matches te beperken.
    """
    if row is None or not hasattr(row, "items"):
        return ""

    markers = sorted(_normalized_exemption_markers(), key=len, reverse=True)
    if not markers:
        return ""

    for column, value in row.items():
        if normalize_domain_text(column) in _SKIP_MARKER_COLUMNS:
            continue

        text = normalize_domain_text(value)
        if not text:
            continue

        for marker in markers:
            if marker and marker in text:
                return marker

    return ""


def is_maintenance_project_exempt(row: Mapping[str, Any] | Any) -> bool:
    """
    Bepaal of een object géén onderhoudsproject nodig heeft.

    De volgorde is:
    1. exact subthema uit de werkproces-uitzonderingenlijst;
    2. specifieke marker, bijvoorbeeld "oorspronkelijke BGT-data", in een ander veld.
    """
    if row is None:
        return False

    # Eerst subthema_clean gebruiken als die inhoud heeft; anders het ruwe subthema.
    # We vermijden `a or b`, omdat pandas.NA geen veilige boolean-waarde heeft.
    subthema = ""
    if hasattr(row, "get"):
        subthema_clean = row.get("subthema_clean", "")
        subthema = subthema_clean if normalize_domain_text(subthema_clean) else row.get("subthema", "")

    if subthema_is_maintenance_project_exempt(subthema):
        return True

    return bool(find_exemption_marker(row))


def maintenance_project_exemption_reason(row: Mapping[str, Any] | Any) -> str:
    """
    Geef een korte reden waarom een object is uitgezonderd.

    Deze tekst is bedoeld voor meldingen in de werklijst en voor tests/debugging.
    """
    if row is None or not hasattr(row, "get"):
        return ""

    subthema_clean = row.get("subthema_clean", "")
    subthema = subthema_clean if normalize_domain_text(subthema_clean) else row.get("subthema", "")
    if subthema_is_maintenance_project_exempt(subthema):
        return f"Subthema '{clean_display_value(row.get('subthema', subthema))}' is uitgezonderd"

    marker = find_exemption_marker(row)
    if marker:
        return f"Uitzonderingsmarker '{marker}' gevonden"

    return ""
