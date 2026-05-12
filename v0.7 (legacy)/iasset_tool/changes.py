"""
Wijzigingslogboek, autosave en export.

Deze module bevat bestands- en mutatielogica. Streamlit bepaalt alleen wanneer
een knop wordt ingedrukt.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd

from .config import EXPORT_COLUMNS
from .utils import clean_display_value, safe_int


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


def build_export_dataframe(
    gdf: gpd.GeoDataFrame,
    changed_ids: Iterable[int],
    export_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bouw de exporttabel met alleen gewijzigde objecten.
    """
    export_columns = export_columns or EXPORT_COLUMNS
    valid_ids = [object_id for object_id in changed_ids if object_id in gdf.index]
    valid_columns = [column for column in export_columns if column in gdf.columns]

    if not valid_ids:
        return pd.DataFrame(columns=valid_columns)

    df_export = gdf.loc[valid_ids, valid_columns].copy()

    for column in ["Jaar aanleg", "Jaar deklaag"]:
        if column in df_export.columns:
            df_export[column] = df_export[column].apply(clean_display_value)

    if "bron_id" in df_export.columns:
        df_export = df_export.rename(columns={"bron_id": "id"})

    return df_export
