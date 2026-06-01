"""
Kaartweergave voor Onderhoudscontrolepunten.

Deze module visualiseert alleen betrokken paspoortobjecten. Objecten die alleen
in de onderhoudsexport staan, kunnen zonder paspoortgeometrie niet op kaart
worden getekend; die blijven wel zichtbaar in de detailtabel.

v0.25 verdiept de kaartcontrole:
- verschiltypen krijgen herkenbare kleuren;
- primaire ruggengraatobjecten worden zwaarder getekend dan secundaire objecten;
- uitgezonderde objecten krijgen een gestippelde lijn;
- pop-ups tonen de oude projectcontext, mogelijke vervanger en prioriteit;
- de kaart bevat een kleine legenda, zodat het beeld zonder codekennis te lezen is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from html import escape
from numbers import Integral, Real
from typing import Any, Iterable

import folium
import geopandas as gpd
import pandas as pd

from .config import BACKBONE_TYPES
from .domain import is_maintenance_project_exempt
from .maintenance_control import normalize_object_number
from .utils import clean_display_value, normalize_text


@dataclass
class MaintenanceControlMapResult:
    """Resultaat van een controlepuntkaart."""

    folium_map: folium.Map | None
    mapped_object_count: int = 0
    missing_passport_object_count: int = 0
    missing_geometry_count: int = 0
    primary_object_count: int = 0
    secondary_object_count: int = 0
    exempt_object_count: int = 0
    difference_type_counts: dict[str, int] = field(default_factory=dict)
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
    """Kies een eenvoudige basisstijl op basis van verschiltype."""
    text = clean_display_value(value).upper()
    if "OBJECT_WEGNUMMER_VERDACHT" in text:
        return {"fillColor": "#E31A1C", "color": "#8B0000", "fillOpacity": 0.78}
    if "ALLEEN_IN_PASPOORT" in text:
        return {"fillColor": "#FFB000", "color": "#B36B00", "fillOpacity": 0.75}
    if "ALLEEN_IN_ONDERHOUD" in text:
        return {"fillColor": "#999999", "color": "#555555", "fillOpacity": 0.55}
    if "ONGELDIGE_METRERING" in text:
        return {"fillColor": "#7B3294", "color": "#4B004B", "fillOpacity": 0.75}
    if "OBJECTSET" in text or "VERSCHIL" in text:
        return {"fillColor": "#1F78B4", "color": "#005F7F", "fillOpacity": 0.72}
    return {"fillColor": "#33A02C", "color": "#1B6E1B", "fillOpacity": 0.62}


def _classify_object_layer(row: pd.Series | dict[str, Any]) -> str:
    """
    Bepaal hoe een object op de kaart moet worden benadrukt.

    Waarom?
    Bij onderhoudscomplexen vormen primaire objecten de ruggengraat. Op de kaart
    moeten die daarom direct herkenbaar zijn, terwijl secundaire objecten wel
    zichtbaar blijven maar minder zwaar wegen in de interpretatie.
    """
    try:
        if is_maintenance_project_exempt(row):
            return "uitzondering"
    except Exception:
        pass

    subthema = normalize_text(clean_display_value(row.get("subthema", row.get("Subthema", ""))))
    primary_types = {normalize_text(value) for value in BACKBONE_TYPES}
    if subthema in primary_types:
        return "primair"
    return "secundair"


def _style_for_feature(feature: dict[str, Any]) -> dict[str, Any]:
    """Combineer verschilkleur met objectlaag, zodat kleur én ruggengraat zichtbaar zijn."""
    properties = feature.get("properties", {})
    style = _style_for_difference(properties.get("_verschiltype_kaart", ""))
    object_layer = clean_display_value(properties.get("_kaartlaag", "")).lower()

    if object_layer == "primair":
        style.update({"weight": 5, "fillOpacity": max(float(style.get("fillOpacity", 0.7)), 0.78)})
    elif object_layer == "uitzondering":
        style.update({"weight": 2, "dashArray": "3,5", "fillOpacity": min(float(style.get("fillOpacity", 0.7)), 0.45)})
    else:
        style.update({"weight": 2.5})

    return style


def _map_context_from_action_row(action_row: dict[str, Any] | pd.Series | None) -> dict[str, str]:
    """Haal kaartcontext uit de geselecteerde actieregel."""
    if action_row is None:
        return {}
    row = action_row.to_dict() if isinstance(action_row, pd.Series) else dict(action_row)
    return {
        "_oude_projectnaam_kaart": clean_display_value(row.get("onderhoudsproject", "")),
        "_mogelijke_vervanger_kaart": clean_display_value(row.get("mogelijke_vervangende_projectnaam", "")),
        "_prioriteit_kaart": clean_display_value(row.get("prioriteit", "")),
        "_duiding_kaart": clean_display_value(row.get("duiding", "")),
        "_voortgang_kaart": clean_display_value(row.get("voortgang_status", "")),
    }


def _build_legend_html() -> str:
    """Maak een compacte legenda voor de Folium-kaart."""
    return """
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        padding: 10px 12px;
        border: 1px solid #bbb;
        border-radius: 6px;
        font-size: 11px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.18);
        max-width: 260px;
    ">
      <div style="font-weight: 700; margin-bottom: 5px;">Legenda Onderhoudscontrole</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#E31A1C;margin-right:6px;"></span>Verdacht wegnummer</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#FFB000;margin-right:6px;"></span>Alleen in paspoort</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#1F78B4;margin-right:6px;"></span>Objectset-/koppelverschil</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#7B3294;margin-right:6px;"></span>Ongeldige metrering</div>
      <div><span style="display:inline-block;width:24px;border-top:4px solid #333;margin-right:6px;"></span>Primaire ruggengraatobjecten</div>
      <div><span style="display:inline-block;width:24px;border-top:2px dashed #333;margin-right:6px;"></span>Uitgezonderde objecten</div>
      <div style="margin-top:5px;color:#555;">Kaart is controlehulp; geen automatische iASSET-mutatie.</div>
    </div>
    """


def build_maintenance_control_map(
    passport_df: pd.DataFrame,
    object_details: pd.DataFrame,
    action_row: dict[str, Any] | pd.Series | None = None,
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
    missing_geometry_count = int((~valid_geometry).sum())
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
    selected["_kaartlaag"] = selected.apply(_classify_object_layer, axis=1)

    context = _map_context_from_action_row(action_row)
    for column, value in context.items():
        selected[column] = value

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
            "_kaartlaag",
            "_verschiltype_kaart",
            "_prioriteit_kaart",
            "_mogelijke_vervanger_kaart",
        ]
        if column and column in web.columns
    ]
    popup_fields = [
        column
        for column in [
            object_column,
            "Wegnummer",
            "subthema",
            "Metrering",
            "Situering",
            "Onderhoudsproject",
            "_oude_projectnaam_kaart",
            "_mogelijke_vervanger_kaart",
            "_prioriteit_kaart",
            "_duiding_kaart",
            "_voortgang_kaart",
            "_verschiltype_kaart",
            "_controle_opmerking",
        ]
        if column and column in web.columns
    ]

    property_columns = list(
        dict.fromkeys(
            [
                "geometry",
                "_object_norm_for_map",
                "_verschiltype_kaart",
                "_kaartlaag",
                *tooltip_fields,
                *popup_fields,
            ]
        )
    )
    map_gdf = gpd.GeoDataFrame(web[property_columns].copy(), geometry="geometry", crs=web.crs)
    for column in map_gdf.columns:
        if column == "geometry":
            continue
        map_gdf[column] = map_gdf[column].map(_json_safe_value)

    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;") if tooltip_fields else None
    popup = folium.GeoJsonPopup(fields=popup_fields, labels=True, max_width=420) if popup_fields else None
    folium.GeoJson(
        map_gdf,
        name="Betrokken objecten",
        style_function=_style_for_feature,
        tooltip=tooltip,
        popup=popup,
    ).add_to(folium_map)

    for _, row in web.iterrows():
        try:
            point = row.geometry.centroid
        except Exception:
            continue
        object_label = clean_display_value(row.get(object_column, "")) if object_column else clean_display_value(row.get("_object_norm_for_map", ""))
        layer = clean_display_value(row.get("_kaartlaag", ""))
        label_prefix = "P" if layer == "primair" else "S" if layer == "secundair" else "U"
        folium.Marker(
            [point.y, point.x],
            tooltip=f"{label_prefix} - {object_label}",
            icon=folium.DivIcon(
                icon_size=(150, 18),
                icon_anchor=(0, 0),
                html=(
                    '<div style="font-size:10px;font-weight:bold;color:#111;'
                    'background:rgba(255,255,255,0.78);padding:1px 3px;border-radius:3px;">'
                    f'{escape(label_prefix)} {escape(object_label)}</div>'
                ),
            ),
        ).add_to(folium_map)

    folium_map.get_root().html.add_child(folium.Element(_build_legend_html()))

    detail_norms = {
        normalize_object_number(value)
        for value in object_details.get("objectnummer_norm", pd.Series(dtype=str))
    } if object_details is not None and not object_details.empty else set()
    mapped_norms = set(web["_object_norm_for_map"].map(normalize_object_number))
    missing_passport = len(detail_norms - mapped_norms)

    layer_counts = web["_kaartlaag"].value_counts().to_dict()
    difference_counts = {
        clean_display_value(key) or "geen_verschiltype": int(value)
        for key, value in web["_verschiltype_kaart"].value_counts(dropna=False).to_dict().items()
    }

    return MaintenanceControlMapResult(
        folium_map=folium_map,
        mapped_object_count=int(len(web)),
        missing_passport_object_count=int(missing_passport),
        missing_geometry_count=int(missing_geometry_count),
        primary_object_count=int(layer_counts.get("primair", 0)),
        secondary_object_count=int(layer_counts.get("secundair", 0)),
        exempt_object_count=int(layer_counts.get("uitzondering", 0)),
        difference_type_counts=difference_counts,
        message="",
    )
