\
"""
Project Adviseur.

Deze module maakt voorstelgroepen voor onderhoudscomplexen.
De Streamlit UI bepaalt alleen hoe de gebruiker zo'n groep bekijkt en accepteert.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx

from .config import (
    FRIENDLY_LABELS,
    HIERARCHY_CONFIG,
    ROAD_DIRECTIONS,
    SEGMENTATION_ATTRIBUTES,
    SUBTHEMA_EXCEPTIONS,
)
from .utils import clean_display_value


def _get_segmentation_hash(gdf: gpd.GeoDataFrame, node_id: int) -> tuple[str, ...]:
    """
    Maak een 'vingerafdruk' van de kenmerken waarop we onderhoudscomplexen knippen.
    """
    row = gdf.loc[node_id]
    return tuple(clean_display_value(row.get(column, "")) for column in SEGMENTATION_ATTRIBUTES)


def _reason_text(segmentation_hash: tuple[str, ...]) -> str:
    """Maak een leesbare tekst waarom objecten bij elkaar horen."""
    specs: list[str] = []

    for index, attribute in enumerate(SEGMENTATION_ATTRIBUTES):
        value = segmentation_hash[index]
        if value:
            specs.append(f"{FRIENDLY_LABELS.get(attribute, attribute)}: {value}")

    return ", ".join(specs) if specs else "Basis kenmerken"


def _axis_tie_breaker(gdf: gpd.GeoDataFrame, group_ids: list[int], direction_code: str) -> float:
    """
    Bepaal een fallback-sortering op basis van de richting van de weg.

    Dit is geen vervanging voor echte hectometrering of een centrale wegas.
    Het is alleen een stabiele tie-breaker wanneer hm_sort ontbreekt of gelijk is.
    """
    group_nodes = gdf.loc[group_ids]

    if group_nodes.empty:
        return 0.0

    center = group_nodes.geometry.unary_union.centroid

    if direction_code == "WTE":      # West -> Oost
        return float(center.x)
    if direction_code == "ETW":      # Oost -> West
        return float(-center.x)
    if direction_code == "STN":      # Zuid -> Noord
        return float(center.y)
    if direction_code == "NTS":      # Noord -> Zuid
        return float(-center.y)

    # Fallback: West -> Oost.
    return float(center.x)


def generate_grouped_proposals(gdf: gpd.GeoDataFrame, graph: nx.Graph) -> dict[str, dict[str, Any]]:
    """
    Genereer onderhoudsprojectvoorstellen.

    Werkwijze:
    1. Bouw per primaire laag een ruggengraat;
    2. knip die ruggengraat bij veranderende segmentatiekenmerken;
    3. laat de groep secundaire objecten absorberen;
    4. sorteer de groepen op rang, hectometrering en ruimtelijke tie-breaker.
    """
    if gdf is None or gdf.empty or graph is None:
        return {}

    groups: dict[str, dict[str, Any]] = {}
    node_to_group: dict[int, str] = {}
    processed_ids: set[int] = set()
    exceptions_clean = {value.lower() for value in SUBTHEMA_EXCEPTIONS}

    for layer in HIERARCHY_CONFIG:
        rank = layer["rank"]
        target_types = layer["types"]
        prefix = layer["prefix"]

        candidates = [
            node
            for node in graph.nodes
            if node in gdf.index
            and gdf.loc[node, "subthema_clean"] in target_types
            and node not in processed_ids
        ]

        if not candidates:
            continue

        graph_sub = graph.subgraph(candidates).copy()

        edges_to_remove = []
        for left, right in graph_sub.edges():
            if _get_segmentation_hash(gdf, left) != _get_segmentation_hash(gdf, right):
                edges_to_remove.append((left, right))

        graph_sub.remove_edges_from(edges_to_remove)

        components = list(nx.connected_components(graph_sub))
        current_layer_groups: list[str] = []

        for index, component in enumerate(components):
            group_id = f"{prefix}_{rank}_{index}"
            node_list = list(component)

            first_node = gdf.loc[node_list[0]]
            seg_props = _get_segmentation_hash(gdf, node_list[0])

            groups[group_id] = {
                "ids": node_list,
                "subthema": target_types[0],
                "rank": rank,
                "prefix": prefix,
                "reason": _reason_text(seg_props),
                "current_project": clean_display_value(first_node.get("Onderhoudsproject", "")),
                "seg_props": seg_props,
                "spatial_sort_val": 0,
            }

            for node in node_list:
                processed_ids.add(node)
                node_to_group[node] = group_id

            current_layer_groups.append(group_id)

        # Expansie: secundaire objecten koppelen aan de ruggengraat.
        all_backbone_types = {
            backbone_type
            for config in HIERARCHY_CONFIG
            for backbone_type in config["types"]
        }

        for group_id in current_layer_groups:
            queue = list(groups[group_id]["ids"])
            pointer = 0

            while pointer < len(queue):
                current_node = queue[pointer]
                pointer += 1

                for neighbor in graph.neighbors(current_node):
                    if neighbor in processed_ids or neighbor not in gdf.index:
                        continue

                    neighbor_sub = gdf.loc[neighbor, "subthema_clean"]

                    if neighbor_sub in exceptions_clean:
                        continue

                    # Een groep mag geen andere primaire ruggengraat opslokken.
                    if neighbor_sub in all_backbone_types:
                        continue

                    groups[group_id]["ids"].append(neighbor)
                    node_to_group[neighbor] = group_id
                    processed_ids.add(neighbor)
                    queue.append(neighbor)

    if not groups:
        return {}

    road_label = str(gdf["Wegnummer"].iloc[0]) if "Wegnummer" in gdf.columns and not gdf.empty else "Onbekend"
    direction_code = ROAD_DIRECTIONS.get(road_label, "UNKNOWN")

    for group_id, group_data in groups.items():
        group_nodes = gdf.loc[group_data["ids"]]
        tie_breaker_value = _axis_tie_breaker(gdf, group_data["ids"], direction_code)

        group_data["tie_breaker_dist"] = tie_breaker_value

        min_hm = group_nodes["hm_sort"].min() if "hm_sort" in group_nodes.columns else 99999.9

        if min_hm < 90000.0:
            group_data["sort_value"] = float(min_hm)
            group_data["sort_mode"] = "hm"
        else:
            group_data["sort_value"] = tie_breaker_value
            group_data["sort_mode"] = "axis"

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[1]["rank"],
            item[1]["sort_value"],
            item[1]["tie_breaker_dist"],
        ),
    )

    final_groups: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}

    for _, data in sorted_groups:
        prefix = data["prefix"]
        counters[prefix] = counters.get(prefix, 0) + 1
        new_id = f"{prefix}_{counters[prefix]}"
        final_groups[new_id] = data

    return final_groups
