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
