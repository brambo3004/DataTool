"""
Folium-kaartopbouw.

De kaartlogica staat los van Streamlit. Streamlit rendert alleen het eindresultaat
met `st_folium`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import folium
import geopandas as gpd
import pandas as pd
import networkx as nx

from .config import SEGMENTATION_ATTRIBUTES
from .utils import clean_display_value


@dataclass
class MapBuildResult:
    """Resultaat van kaartopbouw."""
    folium_map: folium.Map
    network_node_count: int = 0
    network_edge_count: int = 0


def _valid_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    """
    Zet een bounds-object veilig om naar vier floats.

    Streamlit-state kan tuples, lijsten of numpy-arrays bevatten. Door hier
    expliciet te valideren voorkomen we dat de kaart crasht op een lege of
    corrupte selectie.
    """
    if bounds is None:
        return None

    try:
        values = tuple(float(value) for value in bounds)
    except (TypeError, ValueError):
        return None

    if len(values) != 4:
        return None

    minx, miny, maxx, maxy = values
    if any(pd.isna(value) for value in values):
        return None

    if minx == maxx and miny == maxy:
        # Een puntobject heeft geen echte omvang. Geef het een klein venster,
        # zodat `fit_bounds` toch bruikbaar blijft.
        delta = 0.0002
        return minx - delta, miny - delta, maxx + delta, maxy + delta

    return minx, miny, maxx, maxy


def _base_map(road_web: gpd.GeoDataFrame, zoom_bounds: tuple | None) -> folium.Map:
    """
    Maak de basiskaart met of zonder zoom naar selectie.
    """
    selected_bounds = _valid_bounds(zoom_bounds)
    if selected_bounds is not None:
        minx, miny, maxx, maxy = selected_bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")
        m.fit_bounds([[miny, minx], [maxy, maxx]])
        return m

    fallback_bounds = _valid_bounds(road_web.total_bounds)
    if fallback_bounds is None:
        # Uiterste noodfallback: midden van Fryslân. In normale werking komt de
        # app hier niet, omdat de loader lege geometrieën al overslaat.
        return folium.Map(location=[53.1, 5.8], zoom_start=10, tiles="CartoDB positron")

    minx, miny, maxx, maxy = fallback_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    return folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")


def _selected_group_ids(computed_groups: dict | None, selected_group_id: str | None) -> set[int]:
    """Geef de object-id's terug van de geselecteerde adviesgroep."""
    if not computed_groups or not selected_group_id:
        return set()

    group = computed_groups.get(selected_group_id)
    if not group:
        return set()

    return set(group.get("ids", []))


def _suggested_ids(
    computed_groups: dict | None,
    processed_groups: Iterable[str] | None,
    ignored_groups: Iterable[str] | None,
) -> set[int]:
    """Bepaal welke objecten onderdeel zijn van een openstaand advies."""
    if not computed_groups:
        return set()

    processed = set(processed_groups or [])
    ignored = set(ignored_groups or [])

    ids: set[int] = set()
    for group_id, group_data in computed_groups.items():
        if group_id in processed or group_id in ignored:
            continue
        ids.update(group_data.get("ids", []))

    return ids


def _add_network_layer(m: folium.Map, road_web: gpd.GeoDataFrame, graph: nx.Graph) -> tuple[int, int]:
    """Teken rode verbindingslijnen en blauwe node-bollen."""
    line_count = 0

    lines_coords = []
    for left, right in graph.edges():
        if left not in road_web.index or right not in road_web.index:
            continue

        geom_left = road_web.loc[left].geometry
        geom_right = road_web.loc[right].geometry

        if geom_left is None or geom_right is None:
            continue

        point_left = geom_left.centroid
        point_right = geom_right.centroid
        lines_coords.append([[point_left.y, point_left.x], [point_right.y, point_right.x]])

    if lines_coords:
        folium.PolyLine(lines_coords, color="red", weight=2, opacity=1.0).add_to(m)
        line_count = len(lines_coords)

    node_count = 0
    for node_id in graph.nodes():
        if node_id not in road_web.index:
            continue

        geom = road_web.loc[node_id].geometry
        if geom is None:
            continue

        point = geom.centroid
        folium.CircleMarker(
            [point.y, point.x],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=1.0,
            tooltip=f"Node ID: {node_id}",
        ).add_to(m)
        node_count += 1

    return node_count, line_count


def _add_hectometer_layer(m: folium.Map, pdok_hm: gpd.GeoDataFrame | None) -> None:
    """Teken hectometerlabels op de kaart."""
    if pdok_hm is None or pdok_hm.empty:
        return

    try:
        pdok_web = pdok_hm.to_crs(epsg=4326)
    except Exception:
        return

    for _, row in pdok_web.iterrows():
        if row.geometry is None:
            continue

        geom = row.geometry.centroid
        try:
            value = float(row.get("hm_val", 0)) / 10
        except (TypeError, ValueError):
            value = 0.0

        icon_html = (
            '<div style="font-size: 9pt; font-weight: bold; color:black; '
            f'text-shadow: 1px 1px 0 #fff;">{value:.1f}</div>'
        )

        folium.Marker(
            [geom.y, geom.x],
            icon=folium.DivIcon(icon_size=(30, 15), icon_anchor=(15, 7), html=icon_html),
        ).add_to(m)


def _safe_property_value(value: Any) -> Any:
    """
    Maak attribuutwaarden veilig voor Folium/GeoJSON.

    iASSET-exports bevatten soms `pd.NA`, NaN of andere niet-JSON-waarden.
    Zonder opschoning kan Folium dan een TypeError geven tijdens kaartopbouw.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return clean_display_value(value)


def _prepare_geojson_layer(road_web: gpd.GeoDataFrame, cols_to_select: list[str]) -> gpd.GeoDataFrame:
    """
    Maak een lichte, veilige kaartlaag voor Folium.

    We sturen alleen noodzakelijke attributen naar de browser en schonen
    ontbrekende waarden op. De geometrie blijft onaangetast.
    """
    layer = road_web[cols_to_select].copy()

    for column in layer.columns:
        if column == layer.geometry.name:
            continue

        if column == "sys_id":
            # `sys_id` wordt door de style-functie gebruikt en moet numeriek
            # blijven waar dat kan.
            layer[column] = pd.to_numeric(layer[column], errors="coerce").astype("Int64")
            continue

        layer[column] = layer[column].map(_safe_property_value)

    return layer


def build_road_map(
    road_gdf: gpd.GeoDataFrame,
    graph: nx.Graph | None = None,
    *,
    zoom_bounds: tuple | None = None,
    selected_error_id: int | None = None,
    selected_group_id: str | None = None,
    selected_object_id: int | None = None,
    computed_groups: dict | None = None,
    processed_groups: Iterable[str] | None = None,
    ignored_groups: Iterable[str] | None = None,
    error_ids: Iterable[int] | None = None,
    show_network: bool = False,
    pdok_hm: gpd.GeoDataFrame | None = None,
    **_ignored_options: Any,
) -> MapBuildResult:
    """
    Bouw de volledige kaart voor één geselecteerde weg.

    Extra keyword-argumenten worden bewust genegeerd. Daardoor blijft de
    kaartfunctie robuust als `app.py` en deze module bij een lokale update even
    niet exact dezelfde versie hebben.
    """
    road_web = road_gdf.to_crs(epsg=4326)
    m = _base_map(road_web, zoom_bounds)

    error_id_set = set(error_ids or [])
    selected_group_object_ids = _selected_group_ids(computed_groups, selected_group_id)
    open_suggested_ids = _suggested_ids(computed_groups, processed_groups, ignored_groups)

    network_node_count = 0
    network_edge_count = 0

    if show_network and graph is not None:
        network_node_count, network_edge_count = _add_network_layer(m, road_web, graph)

    def style_fn(feature):
        object_id = feature["properties"]["sys_id"]
        props = feature["properties"]

        # Selectie wint altijd van andere stijlen.
        if object_id == selected_error_id or object_id == selected_object_id or object_id in selected_group_object_ids:
            return {"fillColor": "#00FFFF", "color": "black", "weight": 3, "fillOpacity": 0.9}

        if object_id in error_id_set:
            return {"fillColor": "#FFA500", "color": "#cc8400", "weight": 2, "fillOpacity": 0.7}

        if object_id in open_suggested_ids:
            return {"fillColor": "#FFFF00", "color": "black", "weight": 1, "fillOpacity": 0.6}

        if clean_display_value(props.get("Onderhoudsproject", "")):
            return {"fillColor": "#00CC00", "color": "gray", "weight": 0.5, "fillOpacity": 0.5}

        return {"fillColor": "#808080", "color": "gray", "weight": 0.5, "fillOpacity": 0.3}

    tooltip_fields = [
        column
        for column in ["subthema", "Onderhoudsproject", *SEGMENTATION_ATTRIBUTES]
        if column in road_web.columns
    ]

    # Stuur alleen de attributen naar de browser die de kaart écht nodig heeft.
    # In v0.8 gingen alle meta-kolommen mee, inclusief WKT-teksten. Dat maakt
    # de Folium/GeoJSON-laag onnodig zwaar bij grote iASSET-exports.
    property_cols = ["sys_id", "Onderhoudsproject", *tooltip_fields]
    cols_to_select = list(dict.fromkeys(["geometry", *property_cols]))

    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;") if tooltip_fields else None

    folium.GeoJson(
        _prepare_geojson_layer(road_web, cols_to_select),
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    _add_hectometer_layer(m, pdok_hm)

    return MapBuildResult(
        folium_map=m,
        network_node_count=network_node_count,
        network_edge_count=network_edge_count,
    )
