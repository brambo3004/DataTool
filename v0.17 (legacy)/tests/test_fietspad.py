import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from iasset_tool.advisor import generate_grouped_proposals
from iasset_tool.fietspad import (
    FietspadProjectRole,
    angle_difference_deg,
    classify_fietspaden,
)
from iasset_tool.utils import normalize_text


def _line_gdf(rows):
    """Maak een kleine RD-GeoDataFrame met lijngeometrieën voor fietspadtests."""
    prepared = []

    for idx, row in enumerate(rows):
        item = {
            "sys_id": idx,
            "Wegnummer": "N398",
            "subthema": row["subthema"],
            "naam": row.get("naam", ""),
            "Onderhoudsproject": row.get("Onderhoudsproject", ""),
            "verhardingssoort": row.get("verhardingssoort", "asfalt"),
            "Soort deklaag specifiek": row.get("Soort deklaag specifiek", "deklaag"),
            "Jaar aanleg": row.get("Jaar aanleg", "2020"),
            "Jaar deklaag": row.get("Jaar deklaag", "2020"),
            "Besteknummer": row.get("Besteknummer", "B-001"),
            "hm_sort": row.get("hm_sort", float(idx)),
            "geometry": row["geometry"],
        }
        item["subthema_clean"] = normalize_text(item["subthema"])
        prepared.append(item)

    gdf = gpd.GeoDataFrame(prepared, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_angle_difference_treats_opposite_direction_as_parallel():
    assert angle_difference_deg(0, 180) == 0
    assert angle_difference_deg(10, 170) == 20
    assert angle_difference_deg(0, 90) == 90


def test_parallel_fietspad_gets_own_project_role():
    gdf = _line_gdf(
        [
            {"subthema": "rijstrook", "geometry": LineString([(0, 0), (100, 0)])},
            {"subthema": "fietspad", "geometry": LineString([(0, 10), (100, 10)])},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)

    classifications = classify_fietspaden(gdf, graph)

    assert classifications[1].role == FietspadProjectRole.PARALLEL_OWN_PROJECT
    assert classifications[1].angle_delta_deg == 0


def test_crossing_fietspad_is_attached_to_main_project():
    gdf = _line_gdf(
        [
            {"subthema": "rijstrook", "geometry": LineString([(0, 0), (100, 0)])},
            {"subthema": "fietspad", "geometry": LineString([(50, -20), (50, 20)])},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 1, type="lateral")

    classifications = classify_fietspaden(gdf, graph)

    assert classifications[1].role == FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT
    assert classifications[1].angle_delta_deg == 90


def test_advisor_keeps_parallel_fietspad_as_own_group():
    gdf = _line_gdf(
        [
            {"subthema": "rijstrook", "geometry": LineString([(0, 0), (100, 0)])},
            {"subthema": "fietspad", "geometry": LineString([(0, 10), (100, 10)])},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)

    groups = generate_grouped_proposals(gdf, graph)

    group_by_primary = {
        primary_id: group
        for group in groups.values()
        for primary_id in group.get("primary_ids", [])
    }

    assert 0 in group_by_primary
    assert 1 in group_by_primary
    assert group_by_primary[1]["prefix"] == "GRP_FIETSPAD"
    assert not group_by_primary[1].get("attached_fietspad_ids")


def test_advisor_attaches_crossing_fietspad_to_rijstrook_group():
    gdf = _line_gdf(
        [
            {"subthema": "rijstrook", "geometry": LineString([(0, 0), (100, 0)])},
            {"subthema": "fietspad", "geometry": LineString([(50, -20), (50, 20)])},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 1, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)

    rijstrook_group = next(group for group in groups.values() if 0 in group.get("primary_ids", []))

    assert 1 in rijstrook_group["ids"]
    assert 1 in rijstrook_group["secondary_ids"]
    assert 1 in rijstrook_group["attached_fietspad_ids"]
    assert not any(1 in group.get("primary_ids", []) for group in groups.values())


def test_advisor_attaches_crossing_fietspad_with_spatial_fallback_when_graph_gap_exists():
    """Een duidelijk haaks fietspad mag niet verdwijnen als de graafrand ontbreekt."""
    gdf = _line_gdf(
        [
            {"subthema": "rijstrook", "geometry": LineString([(0, 0), (100, 0)])},
            {"subthema": "fietspad", "geometry": LineString([(50, -20), (50, 20)])},
        ]
    )

    # Bewust géén edge: dit simuleert een klein topologisch gat in de iASSET/BGT-geometrie.
    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)

    groups = generate_grouped_proposals(gdf, graph)

    rijstrook_group = next(group for group in groups.values() if 0 in group.get("primary_ids", []))

    assert 1 in rijstrook_group["ids"]
    assert 1 in rijstrook_group["attached_fietspad_ids"]
