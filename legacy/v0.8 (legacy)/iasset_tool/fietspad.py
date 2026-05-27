"""
Classificatie van fietspaden voor de Project Adviseur.

Domeinregel:
- Een parallelfietspad krijgt in principe een eigen onderhoudsproject.
- Een fietspad dat haaks kruist, rondom een rotonde ligt of onderdeel is van
  een kruispunt, hoort bij het onderhoudsproject van de hoofdrijbaan of
  parallelweg.

Deze module maakt die regel expliciet en voorzichtig. Bij twijfel blijft het
fietspad voorlopig een eigen adviesgroep, zodat de databeheerder het handmatig
kan controleren in plaats van dat de app het stilzwijgend verkeerd koppelt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees, hypot
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry.base import BaseGeometry

from .config import HIERARCHY_RANK
from .utils import clean_display_value


class FietspadProjectRole(str, Enum):
    """Rol van een fietspad in de onderhoudsprojectlogica."""

    PARALLEL_OWN_PROJECT = "parallel_own_project"
    ATTACHED_TO_MAIN_PROJECT = "attached_to_main_project"
    UNKNOWN_KEEP_OWN = "unknown_keep_own"


@dataclass(frozen=True)
class FietspadClassification:
    """
    Uitlegbare classificatie voor één fietspadobject.

    `confidence` is bewust indicatief. De app gebruikt hem vooral voor uitleg en
    debugging, niet als harde waarheid richting iASSET.
    """

    node_id: int
    role: FietspadProjectRole
    reason: str
    confidence: str = "laag"
    nearest_primary_id: int | None = None
    nearest_primary_subthema: str | None = None
    distance_m: float | None = None
    angle_delta_deg: float | None = None
    fietspad_angle_deg: float | None = None
    primary_angle_deg: float | None = None


# --- Instelbare drempelwaarden --------------------------------------------

PRIMARY_REFERENCE_TYPES = {"rijstrook", "parallelweg", "landbouwpad", "busbaan"}

PARALLEL_ANGLE_MAX_DEG = 25.0
CROSSING_ANGLE_MIN_DEG = 60.0
MAX_REFERENCE_DISTANCE_M = 30.0
MAX_MARKER_DISTANCE_M = 20.0
GRAPH_SEARCH_DEPTH = 4

# Bij polygonen gebruiken we de verhouding lengte/breedte van de georiënteerde
# bounding box. Een lang, smal vlak heeft een betrouwbare hoofdrichting; een
# bijna rond of vierkant vlak niet.
MIN_POLYGON_ELONGATION_FOR_DIRECTION = 2.5

# Bij lijnen gebruiken we de verhouding tussen hemelsbrede afstand en lijnlengte.
# Een bijna rechte lijn heeft een betrouwbare richting; een slinger of lus niet.
MIN_LINE_STRAIGHTNESS_FOR_DIRECTION = 0.55

ROUNDABOUT_OR_INTERSECTION_MARKERS = {
    "rotonde",
    "rotonderand",
    "kruispunt",
    "verkeerseiland",
    "middengeleider",
    "verkeersplein",
}


def angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """
    Bepaal het kleinste hoekverschil tussen twee wegassen.

    Wegen hebben geen 'kop' en 'staart' voor deze analyse: 0° en 180° betekenen
    allebei parallel. Daarom normaliseren we het verschil naar 0-90 graden.
    """
    delta = abs((angle_a - angle_b) % 180.0)
    if delta > 90.0:
        delta = 180.0 - delta
    return float(delta)


def _iter_line_coords(geometry: BaseGeometry) -> Iterable[list[tuple[float, float]]]:
    """Geef coördinaatreeksen terug voor LineString/MultiLineString-achtige geometrie."""
    geom_type = getattr(geometry, "geom_type", "")

    if geom_type == "LineString":
        yield [(float(x), float(y)) for x, y, *_ in geometry.coords]
        return

    if geom_type == "LinearRing":
        yield [(float(x), float(y)) for x, y, *_ in geometry.coords]
        return

    if geom_type == "MultiLineString":
        for part in geometry.geoms:
            yield from _iter_line_coords(part)
        return

    if geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_line_coords(part)


def _line_orientation(geometry: BaseGeometry) -> tuple[float | None, float, str]:
    """
    Bepaal de richting van een lijngeometrie.

    We gebruiken de langste LineString-part. De betrouwbaarheid komt uit de
    'straightness': hemelsbrede afstand gedeeld door echte lijnlengte.
    """
    best_coords: list[tuple[float, float]] | None = None
    best_length = -1.0

    for coords in _iter_line_coords(geometry):
        if len(coords) < 2:
            continue

        length = 0.0
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            length += hypot(x2 - x1, y2 - y1)

        if length > best_length:
            best_length = length
            best_coords = coords

    if not best_coords or best_length <= 0:
        return None, 0.0, "geen bruikbare lijnrichting"

    start_x, start_y = best_coords[0]
    end_x, end_y = best_coords[-1]
    displacement = hypot(end_x - start_x, end_y - start_y)
    straightness = displacement / best_length if best_length else 0.0

    if displacement <= 0:
        return None, straightness, "lijn heeft geen netto richting"

    angle = degrees(atan2(end_y - start_y, end_x - start_x)) % 180.0
    return float(angle), float(straightness), f"lijnrichting, rechtheid {straightness:.2f}"


def _rectangle_edges(geometry: BaseGeometry) -> list[tuple[float, float, float]]:
    """
    Geef de randen van de minimum rotated rectangle als (dx, dy, lengte).

    Bij onbruikbare geometrie komt een lege lijst terug.
    """
    try:
        rectangle = geometry.minimum_rotated_rectangle
    except Exception:
        return []

    if rectangle is None or rectangle.is_empty or getattr(rectangle, "geom_type", "") != "Polygon":
        return []

    coords = list(rectangle.exterior.coords)
    if len(coords) < 4:
        return []

    edges: list[tuple[float, float, float]] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = hypot(dx, dy)
        if length > 0:
            edges.append((dx, dy, length))

    return edges


def _polygon_orientation(geometry: BaseGeometry) -> tuple[float | None, float, str]:
    """
    Bepaal de hoofdrichting van een vlak via de georiënteerde bounding box.

    Voor lange smalle vlakken werkt dit goed. Voor ronde rotonde-achtige vlakken
    is de richting juist onbetrouwbaar; dat signaal gebruiken we verderop.
    """
    edges = _rectangle_edges(geometry)
    if not edges:
        return None, 0.0, "geen bruikbare vlakrichting"

    lengths = sorted((edge[2] for edge in edges), reverse=True)
    longest = lengths[0]
    shortest = min(length for length in lengths if length > 0)
    elongation = longest / shortest if shortest > 0 else 999.0

    dx, dy, _ = max(edges, key=lambda edge: edge[2])
    angle = degrees(atan2(dy, dx)) % 180.0

    return float(angle), float(elongation), f"vlakrichting, lengte/breedte {elongation:.2f}"


def geometry_orientation_deg(geometry: BaseGeometry | None) -> tuple[float | None, float, str]:
    """
    Bepaal een uitlegbare hoofdrichting voor lijn- of vlakgeometrie.

    Retourneert: (hoek in graden, betrouwbaarheidsscore, uitleg).
    Bij lijnen is score = rechtheid. Bij vlakken is score = lengte/breedte.
    """
    if geometry is None or geometry.is_empty:
        return None, 0.0, "lege geometrie"

    geom_type = getattr(geometry, "geom_type", "")

    if "LineString" in geom_type or geom_type in {"LinearRing", "GeometryCollection"}:
        return _line_orientation(geometry)

    if "Polygon" in geom_type:
        return _polygon_orientation(geometry)

    # Fallback: probeer de georiënteerde bounding box.
    return _polygon_orientation(geometry)


def _orientation_is_reliable(geometry: BaseGeometry, score: float) -> bool:
    """Bepaal of de gevonden richting sterk genoeg is om automatisch te gebruiken."""
    geom_type = getattr(geometry, "geom_type", "")

    if "LineString" in geom_type or geom_type in {"LinearRing", "GeometryCollection"}:
        return score >= MIN_LINE_STRAIGHTNESS_FOR_DIRECTION

    if "Polygon" in geom_type:
        return score >= MIN_POLYGON_ELONGATION_FOR_DIRECTION

    return score >= MIN_POLYGON_ELONGATION_FOR_DIRECTION


def _geometry_is_compact_or_loop_like(geometry: BaseGeometry, score: float) -> bool:
    """
    Herken objecten zonder duidelijke hoofdrichting.

    Dat zijn vaak rotonde-/kruispuntvlakken of korte verbindingsstukken. We
    noemen dit geen harde rotondedetectie, maar wel een reden om voorzichtig te
    zijn met 'parallelfietspad'.
    """
    if geometry is None or geometry.is_empty:
        return False

    geom_type = getattr(geometry, "geom_type", "")

    if "Polygon" in geom_type:
        return 0 < score < MIN_POLYGON_ELONGATION_FOR_DIRECTION

    if "LineString" in geom_type or geom_type in {"LinearRing", "GeometryCollection"}:
        return 0 <= score < MIN_LINE_STRAIGHTNESS_FOR_DIRECTION

    return False


def _row_text(row: Any) -> str:
    """Maak één genormaliseerde tekststring van relevante velden in een rij."""
    preferred_fields = [
        "subthema",
        "naam",
        "Object naam",
        "objectnaam",
        "nummer",
        "Gebruikersfunctie",
        "Type onderdeel",
        "Onderhoudsproject",
        "Situering",
    ]

    values: list[str] = []

    for field in preferred_fields:
        try:
            if field in row.index:
                values.append(clean_display_value(row.get(field, "")))
        except AttributeError:
            # Dict-achtige rij.
            value = row.get(field, "") if hasattr(row, "get") else ""
            values.append(clean_display_value(value))

    return " ".join(value.lower() for value in values if value)


def _contains_marker(text: str) -> bool:
    """Controleer of tekst wijst op rotonde/kruispuntcontext."""
    return any(marker in text for marker in ROUNDABOUT_OR_INTERSECTION_MARKERS)


def _nearby_marker_present(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph | None,
    node_id: int,
    max_marker_distance_m: float = MAX_MARKER_DISTANCE_M,
) -> bool:
    """
    Kijk of er rondom het fietspad rotonde-/kruispuntobjecten liggen.

    We combineren graafburen en geometrische nabijheid. De graaf vangt objecten
    die via BGT-vlakken aansluiten; de afstand vangt objecten die net niet raken.
    """
    if node_id not in gdf.index or "geometry" not in gdf.columns:
        return False

    candidates: set[int] = set()

    if graph is not None and node_id in graph:
        try:
            lengths = nx.single_source_shortest_path_length(graph, node_id, cutoff=2)
            candidates.update(n for n, distance in lengths.items() if distance > 0)
        except Exception:
            pass

    try:
        geom = gdf.loc[node_id].geometry
        if geom is not None and not geom.is_empty:
            distances = gdf.geometry.distance(geom)
            close_ids = distances[distances <= max_marker_distance_m].index
            candidates.update(int(idx) for idx in close_ids if idx != node_id)
    except Exception:
        pass

    for candidate_id in candidates:
        if candidate_id in gdf.index and _contains_marker(_row_text(gdf.loc[candidate_id])):
            return True

    return False


def _candidate_primary_nodes(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph | None,
    node_id: int,
    max_reference_distance_m: float = MAX_REFERENCE_DISTANCE_M,
) -> list[tuple[int, float, int | None]]:
    """
    Zoek mogelijke hoofdrijbaan-/parallelwegreferenties voor een fietspad.

    Retourneert tuples: (node_id, geometrische afstand in meters, graafstand).
    """
    if node_id not in gdf.index or "geometry" not in gdf.columns:
        return []

    primary_nodes = [
        int(idx)
        for idx, row in gdf.iterrows()
        if clean_display_value(row.get("subthema_clean", "")).lower() in PRIMARY_REFERENCE_TYPES
    ]

    if not primary_nodes:
        return []

    start_geom = gdf.loc[node_id].geometry
    if start_geom is None or start_geom.is_empty:
        return []

    graph_distances: dict[int, int] = {}

    if graph is not None and node_id in graph:
        try:
            lengths = nx.single_source_shortest_path_length(graph, node_id, cutoff=GRAPH_SEARCH_DEPTH)
            graph_distances = {
                int(node): int(distance)
                for node, distance in lengths.items()
                if node in primary_nodes and distance > 0
            }
        except Exception:
            graph_distances = {}

    candidates: dict[int, tuple[float, int | None]] = {}

    for primary_id in primary_nodes:
        try:
            distance = float(start_geom.distance(gdf.loc[primary_id].geometry))
        except Exception:
            continue

        graph_distance = graph_distances.get(primary_id)

        if graph_distance is not None or distance <= max_reference_distance_m:
            candidates[primary_id] = (distance, graph_distance)

    def sort_key(item: tuple[int, tuple[float, int | None]]) -> tuple[int, float, int, int]:
        candidate_id, (distance, graph_distance) = item
        subthema = clean_display_value(gdf.loc[candidate_id].get("subthema_clean", "")).lower()
        hierarchy_rank = HIERARCHY_RANK.get(subthema, 99)
        return (
            graph_distance if graph_distance is not None else 999,
            distance,
            hierarchy_rank,
            candidate_id,
        )

    return [
        (candidate_id, distance, graph_distance)
        for candidate_id, (distance, graph_distance) in sorted(candidates.items(), key=sort_key)
    ]


def classify_single_fietspad(
    gdf: gpd.GeoDataFrame,
    node_id: int,
    graph: nx.Graph | None = None,
) -> FietspadClassification:
    """
    Classificeer één fietspadobject.

    Bij twijfel kiezen we `UNKNOWN_KEEP_OWN`: het fietspad blijft dan als eigen
    voorstelgroep zichtbaar. Dat is veiliger dan een parallelfietspad per
    ongeluk aan de hoofdrijbaan koppelen.
    """
    if node_id not in gdf.index:
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
            reason="Object-id niet gevonden in GeoDataFrame.",
            confidence="laag",
        )

    row = gdf.loc[node_id]
    subthema = clean_display_value(row.get("subthema_clean", "")).lower()
    if subthema != "fietspad":
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
            reason="Object is geen fietspad.",
            confidence="laag",
        )

    geometry = row.geometry
    fietspad_angle, fietspad_score, fietspad_orientation_reason = geometry_orientation_deg(geometry)
    row_has_marker = _contains_marker(_row_text(row))
    nearby_has_marker = _nearby_marker_present(gdf, graph, node_id)
    compact_or_loop = _geometry_is_compact_or_loop_like(geometry, fietspad_score)

    candidates = _candidate_primary_nodes(gdf, graph, node_id)
    if not candidates:
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
            reason="Geen hoofdrijbaan/parallelweg binnen de zoekafstand gevonden; fietspad blijft eigen voorstelgroep.",
            confidence="laag",
            fietspad_angle_deg=fietspad_angle,
        )

    primary_id, distance_m, _graph_distance = candidates[0]
    primary_row = gdf.loc[primary_id]
    primary_subthema = clean_display_value(primary_row.get("subthema_clean", "")).lower()
    primary_angle, primary_score, primary_orientation_reason = geometry_orientation_deg(primary_row.geometry)

    if row_has_marker or nearby_has_marker:
        # Een duidelijke rotonde-/kruispuntcontext is sterker dan een lokale
        # hoekmeting, omdat een fietspad rond een rotonde plaatselijk parallel
        # kan lijken.
        marker_reason = "tekst/omgeving wijst op rotonde of kruispunt"
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT,
            reason=f"{marker_reason}; fietspad hoort bij het hoofdproject.",
            confidence="middel",
            nearest_primary_id=primary_id,
            nearest_primary_subthema=primary_subthema,
            distance_m=distance_m,
            fietspad_angle_deg=fietspad_angle,
            primary_angle_deg=primary_angle,
        )

    if fietspad_angle is None or primary_angle is None:
        if compact_or_loop and distance_m <= MAX_REFERENCE_DISTANCE_M:
            return FietspadClassification(
                node_id=node_id,
                role=FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT,
                reason="Geen betrouwbare richting en compact/lusachtig vlak nabij hoofdroute; waarschijnlijk kruispunt/rotondecontext.",
                confidence="laag",
                nearest_primary_id=primary_id,
                nearest_primary_subthema=primary_subthema,
                distance_m=distance_m,
                fietspad_angle_deg=fietspad_angle,
                primary_angle_deg=primary_angle,
            )

        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
            reason=f"Richting onvoldoende betrouwbaar ({fietspad_orientation_reason}; {primary_orientation_reason}); handmatige controle nodig.",
            confidence="laag",
            nearest_primary_id=primary_id,
            nearest_primary_subthema=primary_subthema,
            distance_m=distance_m,
            fietspad_angle_deg=fietspad_angle,
            primary_angle_deg=primary_angle,
        )

    angle_delta = angle_difference_deg(fietspad_angle, primary_angle)
    fietspad_direction_ok = _orientation_is_reliable(geometry, fietspad_score)
    primary_direction_ok = _orientation_is_reliable(primary_row.geometry, primary_score)

    if not fietspad_direction_ok or not primary_direction_ok:
        if compact_or_loop and distance_m <= MAX_REFERENCE_DISTANCE_M:
            return FietspadClassification(
                node_id=node_id,
                role=FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT,
                reason="Geometrie heeft geen duidelijke hoofdrichting en ligt nabij hoofdroute; waarschijnlijk kruispunt/rotondecontext.",
                confidence="laag",
                nearest_primary_id=primary_id,
                nearest_primary_subthema=primary_subthema,
                distance_m=distance_m,
                angle_delta_deg=angle_delta,
                fietspad_angle_deg=fietspad_angle,
                primary_angle_deg=primary_angle,
            )

        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
            reason=f"Hoek {angle_delta:.1f}°, maar richting is onvoldoende betrouwbaar; handmatige controle nodig.",
            confidence="laag",
            nearest_primary_id=primary_id,
            nearest_primary_subthema=primary_subthema,
            distance_m=distance_m,
            angle_delta_deg=angle_delta,
            fietspad_angle_deg=fietspad_angle,
            primary_angle_deg=primary_angle,
        )

    if angle_delta <= PARALLEL_ANGLE_MAX_DEG:
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.PARALLEL_OWN_PROJECT,
            reason=f"Fietspad loopt parallel aan {primary_subthema} (hoekverschil {angle_delta:.1f}°).",
            confidence="hoog",
            nearest_primary_id=primary_id,
            nearest_primary_subthema=primary_subthema,
            distance_m=distance_m,
            angle_delta_deg=angle_delta,
            fietspad_angle_deg=fietspad_angle,
            primary_angle_deg=primary_angle,
        )

    if angle_delta >= CROSSING_ANGLE_MIN_DEG:
        return FietspadClassification(
            node_id=node_id,
            role=FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT,
            reason=f"Fietspad kruist {primary_subthema} haaks/schuin (hoekverschil {angle_delta:.1f}°).",
            confidence="hoog",
            nearest_primary_id=primary_id,
            nearest_primary_subthema=primary_subthema,
            distance_m=distance_m,
            angle_delta_deg=angle_delta,
            fietspad_angle_deg=fietspad_angle,
            primary_angle_deg=primary_angle,
        )

    return FietspadClassification(
        node_id=node_id,
        role=FietspadProjectRole.UNKNOWN_KEEP_OWN,
        reason=f"Hoekverschil {angle_delta:.1f}° ligt tussen parallel en haaks; handmatige controle nodig.",
        confidence="laag",
        nearest_primary_id=primary_id,
        nearest_primary_subthema=primary_subthema,
        distance_m=distance_m,
        angle_delta_deg=angle_delta,
        fietspad_angle_deg=fietspad_angle,
        primary_angle_deg=primary_angle,
    )


def classify_fietspaden(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph | None = None,
) -> dict[int, FietspadClassification]:
    """
    Classificeer alle fietspaden in een GeoDataFrame.

    Retourneert een dictionary met node-id als sleutel.
    """
    if gdf is None or gdf.empty or "subthema_clean" not in gdf.columns:
        return {}

    classifications: dict[int, FietspadClassification] = {}

    for node_id, row in gdf.iterrows():
        if clean_display_value(row.get("subthema_clean", "")).lower() != "fietspad":
            continue

        try:
            int_node_id = int(node_id)
        except (TypeError, ValueError):
            int_node_id = node_id

        classifications[int_node_id] = classify_single_fietspad(gdf, int_node_id, graph)

    return classifications
