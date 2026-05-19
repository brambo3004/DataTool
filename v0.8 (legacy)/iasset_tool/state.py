"""
Session-state helpers voor Streamlit.

Alle sleutels die de UI gebruikt staan hier centraal. Daardoor ontstaan minder
typo's en is duidelijk welke status tijdelijk in de browser/app-sessie leeft.
"""

from __future__ import annotations

from collections.abc import MutableMapping


def init_session_state(session_state: MutableMapping) -> None:
    """Initialiseer de tijdelijke UI-status."""
    defaults = {
        "processed_groups": set,
        "ignored_groups": set,
        "ignored_errors": set,
        "change_log": list,
        "selected_group_id": lambda: None,
        "selected_error_id": lambda: None,
        "zoom_bounds": lambda: None,
        "computed_groups": lambda: None,
    }

    for key, factory in defaults.items():
        if key not in session_state:
            session_state[key] = factory()


def reset_selection(session_state: MutableMapping) -> None:
    """Reset selectie en kaartzoom."""
    session_state["selected_error_id"] = None
    session_state["selected_group_id"] = None
    session_state["zoom_bounds"] = None
    if "folium_map" in session_state:
        session_state["folium_map"] = None


def reset_after_road_change(session_state: MutableMapping, selected_road: str) -> None:
    """Reset wegafhankelijke berekeningen als de gebruiker een andere weg kiest."""
    session_state["last_road"] = selected_road
    session_state["computed_groups"] = None
    session_state["zoom_bounds"] = None
    session_state["selected_error_id"] = None
    session_state["selected_group_id"] = None


def reset_after_data_source_change(session_state: MutableMapping) -> None:
    """
    Reset alle dataset-afhankelijke status.

    Dit gebruiken we wanneer de gebruiker een nieuwe iASSET-export uploadt.
    Bestaande selecties, adviesgroepen en logboekregels horen niet automatisch
    bij een andere bronexport.
    """
    keys_to_remove = [
        "data_complete",
        "invalid_geometry_rows",
        "load_warnings",
        "graph_current",
        "last_road",
        "computed_groups",
        "zoom_bounds",
        "selected_error_id",
        "selected_group_id",
        "processed_groups",
        "ignored_groups",
        "ignored_errors",
        "change_log",
        "folium_map",
    ]

    for key in keys_to_remove:
        session_state.pop(key, None)
