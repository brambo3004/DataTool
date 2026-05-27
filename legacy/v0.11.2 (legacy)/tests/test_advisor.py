import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

from iasset_tool.advisor import generate_grouped_proposals
from iasset_tool.utils import normalize_text


def _gdf(rows):
    prepared = []
    for idx, row in enumerate(rows):
        item = {
            "sys_id": idx,
            "Wegnummer": "N398",
            "subthema": row.get("subthema", ""),
            "Onderhoudsproject": row.get("Onderhoudsproject", ""),
            "verhardingssoort": row.get("verhardingssoort", "asfalt"),
            "Soort deklaag specifiek": row.get("Soort deklaag specifiek", "deklaag"),
            "Jaar aanleg": row.get("Jaar aanleg", "2020"),
            "Jaar deklaag": row.get("Jaar deklaag", "2020"),
            "Besteknummer": row.get("Besteknummer", "B-001"),
            "hm_sort": row.get("hm_sort", 1.0),
            "geometry": Point(idx, 0),
        }
        item["subthema_clean"] = normalize_text(item["subthema"])
        prepared.append(item)

    gdf = gpd.GeoDataFrame(prepared, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_project_advisor_does_not_absorb_documented_exception():
    gdf = _gdf(
        [
            {"subthema": "rijstrook"},
            {"subthema": "perron"},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 1, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)

    assert groups
    all_ids = {object_id for group in groups.values() for object_id in group["ids"]}
    assert 0 in all_ids
    assert 1 not in all_ids


def _group_with_primary(groups, primary_id):
    """Zoek de adviesgroep waarvan het primaire object overeenkomt."""
    for group_id, group_data in groups.items():
        if primary_id in group_data.get("primary_ids", group_data.get("ids", [])):
            return group_id, group_data
    raise AssertionError(f"Geen groep gevonden met primair object {primary_id}")


def test_secondary_direct_between_rijstrook_and_parallelweg_goes_to_rijstrook():
    gdf = _gdf(
        [
            {"subthema": "rijstrook"},
            {"subthema": "parallelweg"},
            {"subthema": "bermverharding"},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 2, type="lateral")
    graph.add_edge(1, 2, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)

    _, rijstrook_group = _group_with_primary(groups, 0)
    _, parallel_group = _group_with_primary(groups, 1)

    assert 2 in rijstrook_group["ids"]
    assert 2 in rijstrook_group["secondary_ids"]
    assert 2 not in parallel_group["ids"]


def test_secondary_uses_shortest_topological_distance_before_hierarchy_for_indirect_chains():
    gdf = _gdf(
        [
            {"subthema": "rijstrook"},
            {"subthema": "fietspad"},
            {"subthema": "bermverharding"},
            {"subthema": "goot"},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 3, type="lateral")
    graph.add_edge(3, 2, type="lateral")
    graph.add_edge(2, 1, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)

    _, rijstrook_group = _group_with_primary(groups, 0)
    _, fietspad_group = _group_with_primary(groups, 1)

    assert 3 in rijstrook_group["ids"]
    assert 2 in fietspad_group["ids"]
    assert 2 not in rijstrook_group["ids"]


def test_secondary_chain_between_two_rijstrook_groups_is_split_by_shortest_distance():
    gdf = _gdf(
        [
            {"subthema": "rijstrook", "Jaar aanleg": "2020", "hm_sort": 1.0},
            {"subthema": "rijstrook", "Jaar aanleg": "2021", "hm_sort": 2.0},
            {"subthema": "bermverharding"},
            {"subthema": "goot"},
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 2, type="lateral")
    graph.add_edge(2, 3, type="lateral")
    graph.add_edge(3, 1, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)

    _, first_rijstrook_group = _group_with_primary(groups, 0)
    _, second_rijstrook_group = _group_with_primary(groups, 1)

    assert 2 in first_rijstrook_group["ids"]
    assert 3 in second_rijstrook_group["ids"]
