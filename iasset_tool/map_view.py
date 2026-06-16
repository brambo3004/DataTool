"""
Folium-kaartopbouw.

De kaartlogica staat los van Streamlit. Streamlit rendert alleen het eindresultaat
met `st_folium`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Iterable

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd

from .config import SEGMENTATION_ATTRIBUTES
from .utils import clean_display_value


@dataclass
class MapBuildResult:
    """Resultaat van kaartopbouw."""
    folium_map: folium.Map
    network_node_count: int = 0
    network_edge_count: int = 0


def _json_safe_object_id(value: object) -> int | str:
    """
    Maak een object-id geschikt voor GeoJSON én voor vergelijkingen in de stijlfunctie.

    Folium serialiseert kaartattributen via JSON. Numpy/pandas-getaltypen kunnen
    daarbij een TypeError geven, terwijl ze voor ons gewoon een technisch object-id
    voorstellen. We zetten hele getallen daarom om naar een normale Python ``int``.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return int(numeric_value)
        return clean_display_value(value)

    text = clean_display_value(value)
    try:
        numeric_value = float(text)
    except (TypeError, ValueError):
        return text

    if numeric_value.is_integer():
        return int(numeric_value)

    return text


def _object_id_key(value: Any) -> str:
    """
    Maak een stabiele vergelijkingssleutel voor object-id's.

    Waarom?
    De iASSET-exports leveren ``sys_id`` soms als getal, soms als tekst en na een
    CSV-ronde soms als ``1.0``. Voor kaartselecties willen we dezelfde objecten
    toch herkennen zonder dat de styling wegvalt.
    """
    safe_value = _json_safe_object_id(value)
    return clean_display_value(safe_value).strip()


def _object_id_key_set(values: Iterable[Any] | None) -> set[str]:
    """Normaliseer een lijst/set object-id's naar veilige vergelijkingssleutels."""
    if values is None:
        return set()
    return {key for key in (_object_id_key(value) for value in values) if key}


def _add_project_proposal_legend(m: folium.Map) -> None:
    """Voeg een compacte legenda toe voor de projectvoorstel-inspectie."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: white;
        padding: 8px 10px;
        border: 1px solid #999;
        border-radius: 4px;
        font-size: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    ">
      <b>Projectvoorstel-inspectie</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#8A2BE2;border:1px solid #333;"></span>
      geselecteerd voorstel<br>
      <span style="display:inline-block;width:12px;height:12px;background:#1E90FF;border:1px solid #004C99;"></span>
      bestaand iASSET-overlap<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _json_safe_property_value(value: object) -> str | int | float | bool:
    """
    Zet kaartattributen om naar JSON-veilige waarden.

    Waarom dit nodig is:
    iASSET-exports en Excelbestanden kunnen pandas-typen bevatten zoals
    ``Timestamp`` of ``NA``. Folium kan die niet rechtstreeks als GeoJSON
    serialiseren. Omdat deze waarden alleen bedoeld zijn voor tooltip/kaartweergave,
    is een nette tekstrepresentatie hier veiliger dan de app laten crashen.
    """
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
        numeric_value = float(value)
        return numeric_value if numeric_value == numeric_value else ""

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return clean_display_value(value)

    return clean_display_value(value)


def _prepare_geojson_properties(
    road_web: gpd.GeoDataFrame,
    cols_to_select: list[str],
) -> gpd.GeoDataFrame:
    """
    Maak een lichte, GeoJSON-veilige kopie voor de Folium-laag.

    De originele GeoDataFrame blijft ongemoeid. Alleen de beperkte set
    kaartkolommen wordt opgeschoond, zodat paspoortdata met datums, NA-waarden of
    pandas/numpy-scalar-typen geen kaartcrash veroorzaakt.
    """
    present_cols = [column for column in cols_to_select if column in road_web.columns]
    map_gdf = gpd.GeoDataFrame(
        road_web[present_cols].copy(),
        geometry="geometry",
        crs=road_web.crs,
    )

    if "sys_id" in map_gdf.columns:
        map_gdf["sys_id"] = map_gdf["sys_id"].map(_json_safe_object_id)

    for column in map_gdf.columns:
        if column == "geometry" or column == "sys_id":
            continue
        map_gdf[column] = map_gdf[column].map(_json_safe_property_value)

    return map_gdf


def _base_map(road_web: gpd.GeoDataFrame, zoom_bounds: tuple | None) -> folium.Map:
    """
    Maak de basiskaart met of zonder zoom naar selectie.
    """
    if zoom_bounds:
        minx, miny, maxx, maxy = zoom_bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")
        m.fit_bounds([[miny, minx], [maxy, maxx]])
        return m

    minx, miny, maxx, maxy = road_web.total_bounds
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
        value = float(row.get("hm_val", 0)) / 10
        icon_html = (
            '<div style="font-size: 9pt; font-weight: bold; color:black; '
            f'text-shadow: 1px 1px 0 #fff;">{value:.1f}</div>'
        )

        folium.Marker(
            [geom.y, geom.x],
            icon=folium.DivIcon(icon_size=(30, 15), icon_anchor=(15, 7), html=icon_html),
        ).add_to(m)


def build_road_map(
    road_gdf: gpd.GeoDataFrame,
    graph: nx.Graph | None = None,
    *,
    zoom_bounds: tuple | None = None,
    selected_error_id: int | None = None,
    selected_group_id: str | None = None,
    selected_object_id: int | None = None,
    selected_project_proposal_object_ids: Iterable[Any] | None = None,
    selected_project_proposal_existing_object_ids: Iterable[Any] | None = None,
    computed_groups: dict | None = None,
    processed_groups: Iterable[str] | None = None,
    ignored_groups: Iterable[str] | None = None,
    error_ids: Iterable[int] | None = None,
    show_network: bool = False,
    pdok_hm: gpd.GeoDataFrame | None = None,
) -> MapBuildResult:
    """
    Bouw de volledige kaart voor één geselecteerde weg.

    v0.35.3 kan optioneel een groenveld-projectvoorstel uitlichten. De
    berekening van het voorstel gebeurt elders; deze kaartfunctie blijft alleen
    verantwoordelijk voor de visuele inspectie.
    """
    road_web = road_gdf.to_crs(epsg=4326)
    m = _base_map(road_web, zoom_bounds)

    error_id_set = _object_id_key_set(error_ids)
    selected_error_key = _object_id_key(selected_error_id) if selected_error_id is not None else ""
    selected_object_key = _object_id_key(selected_object_id) if selected_object_id is not None else ""
    selected_group_object_ids = _selected_group_ids(computed_groups, selected_group_id)
    selected_group_object_keys = _object_id_key_set(selected_group_object_ids)
    selected_proposal_object_keys = _object_id_key_set(selected_project_proposal_object_ids)
    selected_proposal_existing_object_keys = (
        _object_id_key_set(selected_project_proposal_existing_object_ids)
        - selected_proposal_object_keys
    )
    open_suggested_ids = _suggested_ids(computed_groups, processed_groups, ignored_groups)
    open_suggested_keys = _object_id_key_set(open_suggested_ids)

    network_node_count = 0
    network_edge_count = 0

    if show_network and graph is not None:
        network_node_count, network_edge_count = _add_network_layer(m, road_web, graph)

    def style_fn(feature):
        object_id = feature["properties"]["sys_id"]
        object_key = _object_id_key(object_id)
        props = feature["properties"]

        # Handmatige selectie uit de bestaande schermen wint altijd van andere stijlen.
        if (
            object_key == selected_error_key
            or object_key == selected_object_key
            or object_key in selected_group_object_keys
        ):
            return {"fillColor": "#00FFFF", "color": "black", "weight": 3, "fillOpacity": 0.9}

        # v0.35.3: inspectie van groenveld-projectvoorstellen. Objecten uit het
        # geselecteerde voorstel krijgen de meest opvallende projectkleur.
        if object_key in selected_proposal_object_keys:
            return {"fillColor": "#8A2BE2", "color": "black", "weight": 3, "fillOpacity": 0.85}

        # Objecten uit bestaande iASSET-projecten die met het voorstel overlappen,
        # maar zelf niet in het voorstel zitten, tonen we als vergelijkingscontext.
        if object_key in selected_proposal_existing_object_keys:
            return {"fillColor": "#1E90FF", "color": "#004C99", "weight": 2, "fillOpacity": 0.65}

        if object_key in error_id_set:
            return {"fillColor": "#FFA500", "color": "#cc8400", "weight": 2, "fillOpacity": 0.7}

        if object_key in open_suggested_keys:
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
    map_gdf = _prepare_geojson_properties(road_web, cols_to_select)

    folium.GeoJson(
        map_gdf,
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    _add_hectometer_layer(m, pdok_hm)

    if selected_proposal_object_keys or selected_proposal_existing_object_keys:
        _add_project_proposal_legend(m)

    return MapBuildResult(
        folium_map=m,
        network_node_count=network_node_count,
        network_edge_count=network_edge_count,
    )
