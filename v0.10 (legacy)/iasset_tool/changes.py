"""
Wijzigingslogboek, autosave en export.

Deze module bevat bestands- en mutatielogica. Streamlit bepaalt alleen wanneer
een knop wordt ingedrukt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd

from .config import DEFAULT_EXPORT_PROFILE, EXPORT_COLUMNS, EXPORT_PROFILES
from .utils import clean_display_value, safe_int


@dataclass(frozen=True)
class ExportSummary:
    """Samenvatting van wat een iASSET-importbestand gaat meeschrijven."""

    profile_name: str
    changed_object_count: int
    changed_cell_count: int
    changed_cells_in_profile_count: int
    written_value_count: int
    unchanged_written_value_count: int
    valid_export_columns: list[str]
    omitted_changed_fields: list[str]


def apply_change_to_data(gdf: gpd.GeoDataFrame, object_id: Any, field: str, new_value: Any) -> bool:
    """
    Pas één wijziging toe op de werkdata.

    Geeft True terug als de wijziging is toegepast, anders False.
    """
    coerced_id = safe_int(object_id)
    if coerced_id is None:
        return False

    if coerced_id not in gdf.index:
        return False

    if field not in gdf.columns:
        gdf[field] = ""

    gdf.at[coerced_id, field] = new_value
    return True


def add_log_entry(
    change_log: list[dict[str, Any]],
    object_id: Any,
    field: str,
    old_value: Any,
    new_value: Any,
    status: str = "Succes",
) -> dict[str, Any]:
    """
    Voeg een regel toe aan het wijzigingslogboek.
    """
    entry = {
        "Tijd": datetime.now().strftime("%H:%M:%S"),
        "ID": safe_int(object_id) if safe_int(object_id) is not None else object_id,
        "Veld": field,
        "Oud": str(clean_display_value(old_value)),
        "Nieuw": str(clean_display_value(new_value)),
        "Status": status,
    }
    change_log.append(entry)
    return entry


def save_autosave(change_log: list[dict[str, Any]], autosave_path: str | Path) -> None:
    """
    Schrijf het wijzigingslogboek naar schijf.

    Bij een leeg logboek wordt het oude autosave-bestand verwijderd.
    """
    path = Path(autosave_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if change_log:
        pd.DataFrame(change_log).to_csv(path, index=False, sep=";", encoding="utf-8-sig")
        return

    if path.exists():
        path.unlink()


def load_autosave(autosave_path: str | Path) -> list[dict[str, Any]]:
    """
    Lees een autosave-logboek in.

    Ongeldige of ontbrekende bestanden leveren een leeg logboek op.
    """
    path = Path(autosave_path)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return []

    if df.empty:
        return []

    records = df.fillna("").to_dict("records")

    for record in records:
        coerced_id = safe_int(record.get("ID"))
        if coerced_id is not None:
            record["ID"] = coerced_id

    return records


def collect_changed_ids(change_log: Iterable[dict[str, Any]]) -> set[int]:
    """Verzamel alle object-id's die in het logboek voorkomen."""
    ids: set[int] = set()

    for entry in change_log:
        coerced_id = safe_int(entry.get("ID"))
        if coerced_id is not None:
            ids.add(coerced_id)

    return ids


def available_export_profiles() -> list[str]:
    """Geef de beschikbare iASSET-exportprofielen in vaste volgorde terug."""
    return list(EXPORT_PROFILES.keys())


def get_export_profile_columns(profile_name: str | None = None) -> list[str]:
    """
    Geef de kolommen van een exportprofiel terug.

    Onbekende profielnamen vallen bewust terug op het standaardprofiel. Daardoor
    crasht een oude autosave of sessie niet als profielen later worden aangepast.
    """
    profile = profile_name or DEFAULT_EXPORT_PROFILE
    return list(EXPORT_PROFILES.get(profile, EXPORT_PROFILES[DEFAULT_EXPORT_PROFILE]))


def build_export_dataframe(
    gdf: gpd.GeoDataFrame,
    changed_ids: Iterable[int],
    export_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bouw de exporttabel met alleen gewijzigde objecten.

    Let op: de gekozen kolommenset wordt voor alle gewijzigde objecten
    meegeschreven. Dat past bij iASSET-importprofielen, maar betekent ook dat
    ongewijzigde waarden binnen het profiel in de export terechtkomen.
    """
    export_columns = export_columns or EXPORT_COLUMNS
    valid_ids = [object_id for object_id in changed_ids if object_id in gdf.index]
    valid_columns = [column for column in export_columns if column in gdf.columns]

    if not valid_ids:
        return pd.DataFrame(columns=valid_columns)

    df_export = gdf.loc[valid_ids, valid_columns].copy()

    for column in ["Jaar aanleg", "Jaar deklaag", "Jaar conservering", "Jaar herstrating"]:
        if column in df_export.columns:
            df_export[column] = df_export[column].apply(clean_display_value)

    if "bron_id" in df_export.columns:
        df_export = df_export.rename(columns={"bron_id": "id"})

    return df_export


def _successful_changed_cells(change_log: Iterable[dict[str, Any]]) -> set[tuple[int, str]]:
    """
    Verzamel unieke gewijzigde cellen uit het logboek.

    Meerdere wijzigingen op hetzelfde object/veld tellen als één cel, omdat de
    export alleen de laatste werkwaarde van dat veld meeneemt.
    """
    cells: set[tuple[int, str]] = set()

    for entry in change_log:
        if str(entry.get("Status", "Succes")).lower() not in {"succes", "success", ""}:
            continue

        object_id = safe_int(entry.get("ID"))
        field = str(entry.get("Veld", "")).strip()

        if object_id is None or not field:
            continue

        cells.add((object_id, field))

    return cells


def summarize_export_profile(
    gdf: gpd.GeoDataFrame,
    change_log: Iterable[dict[str, Any]],
    profile_name: str | None = None,
) -> ExportSummary:
    """
    Maak zichtbaar wat het gekozen iASSET-exportprofiel precies doet.

    Waarom?
    iASSET importeert per bestand één vaste kolommenset voor alle objecten. Als
    object A alleen een verhardingssoortwijziging heeft en object B alleen een
    onderhoudsprojectwijziging, dan worden in één gecombineerde export toch alle
    profielkolommen voor beide objecten meegeschreven.
    """
    selected_profile = profile_name or DEFAULT_EXPORT_PROFILE
    profile_columns = get_export_profile_columns(selected_profile)

    changed_ids = collect_changed_ids(change_log)
    valid_ids = {object_id for object_id in changed_ids if object_id in gdf.index}
    valid_columns = [column for column in profile_columns if column in gdf.columns]

    changed_cells = _successful_changed_cells(change_log)
    valid_changed_cells = {
        (object_id, field)
        for object_id, field in changed_cells
        if object_id in valid_ids
    }

    changed_cells_in_profile = {
        (object_id, field)
        for object_id, field in valid_changed_cells
        if field in valid_columns
    }

    changed_fields = {field for _, field in valid_changed_cells}
    omitted_changed_fields = sorted(field for field in changed_fields if field not in valid_columns)

    written_value_count = len(valid_ids) * len(valid_columns)
    unchanged_written_value_count = max(0, written_value_count - len(changed_cells_in_profile))

    return ExportSummary(
        profile_name=selected_profile,
        changed_object_count=len(valid_ids),
        changed_cell_count=len(valid_changed_cells),
        changed_cells_in_profile_count=len(changed_cells_in_profile),
        written_value_count=written_value_count,
        unchanged_written_value_count=unchanged_written_value_count,
        valid_export_columns=valid_columns,
        omitted_changed_fields=omitted_changed_fields,
    )
