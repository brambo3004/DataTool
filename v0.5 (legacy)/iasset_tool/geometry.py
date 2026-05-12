\
"""
Ruimtelijke netwerklogica.

Hier bouwen we van losse objectgeometrieën een NetworkX-graaf:
- elk object is een node;
- objecten die elkaar raken of bijna raken krijgen een edge.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx


def build_graph_from_geometry(gdf: gpd.GeoDataFrame, buffer_meters: float = 0.5) -> nx.Graph:
    """
    Bouw een ruimtelijk netwerk op basis van geometrische nabijheid.

    We bufferen de linkerkant met 0,5 meter. Dat is bewust: in BGT/iASSET-data
    raken vlakken of lijnen elkaar soms net niet door afronding of geometrische
    toleranties. De buffer maakt de koppeling praktischer voor databeheer.

    Bij fouten wordt een graaf met alleen nodes teruggegeven. De app blijft dan
    bruikbaar, alleen zonder ruimtelijke relaties.
    """
    graph = nx.Graph()

    if gdf is None or gdf.empty:
        return graph

    graph.add_nodes_from(gdf.index)

    required = {"geometry", "subthema_clean", "Rank", "sys_id"}
    missing = required.difference(gdf.columns)
    if missing:
        graph.graph["warning"] = f"Netwerk niet volledig: ontbrekende kolommen {sorted(missing)}"
        return graph

    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if valid.empty:
        graph.graph["warning"] = "Netwerk niet opgebouwd: geen geldige geometrieën."
        return graph

    try:
        buffered = valid[["geometry", "subthema_clean", "Rank", "sys_id"]].copy()
        buffered["geometry"] = buffered.geometry.buffer(buffer_meters)

        left_df = buffered.copy()
        left_df.index.name = None

        right_df = valid[["geometry", "subthema_clean", "Rank", "sys_id"]].copy()
        right_df.index.name = None

        joined = gpd.sjoin(
            left_df,
            right_df,
            how="inner",
            predicate="intersects",
            lsuffix="left",
            rsuffix="right",
        )
    except Exception as exc:
        graph.graph["warning"] = f"Spatial join mislukt: {exc}"
        return graph

    if joined.empty:
        return graph

    joined = joined[joined["sys_id_left"] != joined["sys_id_right"]]

    edges = []
    for _, row in joined.iterrows():
        idx_left = row["sys_id_left"]
        idx_right = row["sys_id_right"]
        sub_left = row["subthema_clean_left"]
        sub_right = row["subthema_clean_right"]

        relation_type = "longitudinal" if sub_left == sub_right else "lateral"
        edges.append((idx_left, idx_right, {"type": relation_type}))

    graph.add_edges_from(edges)
    return graph
