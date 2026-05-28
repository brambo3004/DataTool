"""
Kaartweergave voor Onderhoudscontrolepunten.

Deze module visualiseert alleen betrokken paspoortobjecten. Objecten die alleen
in de onderhoudsexport staan, kunnen zonder paspoortgeometrie niet op kaart
worden getekend; die blijven wel zichtbaar in de detailtabel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Iterable

import folium
import geopandas as gpd
import pandas as pd

from .maintenance_control import normalize_object_number
from .utils import clean_display_value


@dataclass
class MaintenanceControlMapResult:
    """Resultaat van een controlepuntkaart."""

    folium_map: folium.Map | None
    mapped_object_count: int = 0
    missing_passport_object_count: int = 0
    missing_geometry_count: int = 0
    message: str = ""


def _json_safe_value(value: Any) -> str | int | float | bool:
    """Maak waarden veilig voor Folium/GeoJSON."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if numeric == numeric else ""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return clean_display_value(value)
    return clean_display_value(value)


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kolom terug."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _passport_rows_for_objects(passport_df: pd.DataFrame, object_details: pd.DataFrame) -> gpd.GeoDataFrame:
    """Selecteer paspoortregels die horen bij de objecten uit de detailtabel."""
    if passport_df is None or passport_df.empty or object_details is None or object_details.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")

    if "geometry" not in passport_df.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")

    object_column = _first_existing_column(passport_df, ["nummer", "bron_id", "objectnummer", "sys_id"])
    if object_column is None or "objectnummer_norm" not in object_details.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(passport_df, "crs", None))

    wanted = {
        normalize_object_number(value)
        for value in object_details["objectnummer_norm"]
        if normalize_object_number(value)
    }
    if not wanted:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(passport_df, "crs", None))

    working = passport_df.copy()
    working["_object_norm_for_map"] = working[object_column].map(normalize_object_number)
    selected = working[working["_object_norm_for_map"].isin(wanted)].copy()

    if selected.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(passport_df, "crs", None))

    if not isinstance(selected, gpd.GeoDataFrame):
        selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=getattr(passport_df, "crs", None))

    return selected


def _style_for_difference(value: Any) -> dict[str, Any]:
    """Kies een eenvoudige kaartstijl op basis van verschiltype."""
    text = clean_display_value(value).upper()
    if "OBJECT_WEGNUMMER_VERDACHT" in text:
        return {"fillColor": "#FF0000", "color": "#8B0000", "weight": 3, "fillOpacity": 0.75}
    if "ALLEEN_IN_PASPOORT" in text:
        return {"fillColor": "#FFA500", "color": "#B36B00", "weight": 3, "fillOpacity": 0.75}
    if "ONGELDIGE_METRERING" in text:
        return {"fillColor": "#800080", "color": "#4B004B", "weight": 3, "fillOpacity": 0.75}
    return {"fillColor": "#00BFFF", "color": "#005F7F", "weight": 3, "fillOpacity": 0.7}


def build_maintenance_control_map(
    passport_df: pd.DataFrame,
    object_details: pd.DataFrame,
) -> MaintenanceControlMapResult:
    """
    Bouw een kaart voor de paspoortobjecten van één controlepunt.

    Veiligheidsregel: deze functie visualiseert alleen. Er worden geen wijzigingen
    in iASSET-data of exports doorgevoerd.
    """
    selected = _passport_rows_for_objects(passport_df, object_details)
    if selected.empty:
        missing_count = int(len(object_details)) if object_details is not None else 0
        return MaintenanceControlMapResult(
            folium_map=None,
            missing_passport_object_count=missing_count,
            message="Geen paspoortgeometrie gevonden voor de betrokken objecten.",
        )

    valid_geometry = selected["geometry"].notna()
    try:
        valid_geometry &= ~selected["geometry"].is_empty
    except Exception:
        pass
    selected = selected.loc[valid_geometry].copy()

    if selected.empty:
        return MaintenanceControlMapResult(
            folium_map=None,
            missing_geometry_count=int(len(object_details)),
            message="De betrokken paspoortobjecten hebben geen bruikbare geometrie.",
        )

    detail_by_norm = {}
    if object_details is not None and not object_details.empty and "objectnummer_norm" in object_details.columns:
        for _, row in object_details.iterrows():
            detail_by_norm[normalize_object_number(row.get("objectnummer_norm", ""))] = row.to_dict()

    object_column = _first_existing_column(selected, ["nummer", "bron_id", "objectnummer", "sys_id"])
    selected["_object_norm_for_map"] = selected[object_column].map(normalize_object_number) if object_column else ""

    selected["_verschiltype_kaart"] = selected["_object_norm_for_map"].map(
        lambda value: clean_display_value(detail_by_norm.get(value, {}).get("verschiltype", ""))
    )
    selected["_controle_opmerking"] = selected["_object_norm_for_map"].map(
        lambda value: clean_display_value(detail_by_norm.get(value, {}).get("opmerking", ""))
    )

    try:
        web = selected.to_crs(epsg=4326)
    except Exception:
        web = selected.copy()
        if getattr(web, "crs", None) is None:
            # iASSET RD-geometrie is de normale bron. Als CRS ontbreekt, proberen we
            # voorzichtig RD aan te nemen voordat we de kaart opgeven.
            try:
                web = web.set_crs(epsg=28992).to_crs(epsg=4326)
            except Exception:
                return MaintenanceControlMapResult(
                    folium_map=None,
                    mapped_object_count=0,
                    message="Kon de geometrie niet naar WGS84 omzetten voor kaartweergave.",
                )

    bounds = web.total_bounds
    minx, miny, maxx, maxy = bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    folium_map = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")
    folium_map.fit_bounds([[miny, minx], [maxy, maxx]])

    tooltip_fields = [
        column
        for column in [
            object_column,
            "Wegnummer",
            "subthema",
            "Metrering",
            "Situering",
            "Onderhoudsproject",
            "_verschiltype_kaart",
            "_controle_opmerking",
        ]
        if column and column in web.columns
    ]

    property_columns = list(dict.fromkeys(["geometry", "_object_norm_for_map", "_verschiltype_kaart", *tooltip_fields]))
    map_gdf = gpd.GeoDataFrame(web[property_columns].copy(), geometry="geometry", crs=web.crs)
    for column in map_gdf.columns:
        if column == "geometry":
            continue
        map_gdf[column] = map_gdf[column].map(_json_safe_value)

    def style_fn(feature):
        return _style_for_difference(feature["properties"].get("_verschiltype_kaart", ""))

    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;") if tooltip_fields else None
    folium.GeoJson(map_gdf, style_function=style_fn, tooltip=tooltip).add_to(folium_map)

    for _, row in web.iterrows():
        try:
            point = row.geometry.centroid
        except Exception:
            continue
        object_label = clean_display_value(row.get(object_column, "")) if object_column else clean_display_value(row.get("_object_norm_for_map", ""))
        folium.Marker(
            [point.y, point.x],
            tooltip=object_label,
            icon=folium.DivIcon(
                icon_size=(140, 18),
                icon_anchor=(0, 0),
                html=(
                    '<div style="font-size:10px;font-weight:bold;color:#111;'
                    'background:rgba(255,255,255,0.75);padding:1px 3px;border-radius:3px;">'
                    f'{object_label}</div>'
                ),
            ),
        ).add_to(folium_map)

    detail_norms = {
        normalize_object_number(value)
        for value in object_details.get("objectnummer_norm", pd.Series(dtype=str))
    } if object_details is not None and not object_details.empty else set()
    mapped_norms = set(web["_object_norm_for_map"].map(normalize_object_number))
    missing_passport = len(detail_norms - mapped_norms)

    return MaintenanceControlMapResult(
        folium_map=folium_map,
        mapped_object_count=int(len(web)),
        missing_passport_object_count=int(missing_passport),
        missing_geometry_count=0,
        message="",
    )
