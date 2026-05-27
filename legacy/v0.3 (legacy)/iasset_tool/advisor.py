"""
Project Adviseur.

Deze module maakt voorstelgroepen voor onderhoudscomplexen.
De Streamlit UI bepaalt alleen hoe de gebruiker zo'n groep bekijkt en accepteert.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import geopandas as gpd
import networkx as nx

from .config import (
    FRIENDLY_LABELS,
    HIERARCHY_CONFIG,
    ROAD_DIRECTIONS,
    SEGMENTATION_ATTRIBUTES,
)
from .domain import is_maintenance_project_exempt
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

    # GeoPandas heeft `union_all()` als moderne vervanger van `unary_union`.
    # De fallback houdt de code bruikbaar op oudere GeoPandas-versies.
    try:
        merged_geometry = group_nodes.geometry.union_all()
    except AttributeError:
        merged_geometry = group_nodes.geometry.unary_union

    center = merged_geometry.centroid

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


def _all_backbone_types() -> set[str]:
    """
    Geef alle primaire subthema's terug.

    Primaire objecten vormen de ruggengraat van een onderhoudscomplex. Ze mogen
    daarom niet door een andere groep als secundair object worden opgeslokt.
    """
    return {
        backbone_type
        for config in HIERARCHY_CONFIG
        for backbone_type in config["types"]
    }


def _layer_rank_for_subthema(subthema_clean: str) -> int | None:
    """
    Geef de hiërarchische rang terug voor een primair subthema.

    Lager getal betekent belangrijker: rijstrook gaat bijvoorbeeld vóór
    parallelweg, en parallelweg gaat vóór fietspad.
    """
    for layer in HIERARCHY_CONFIG:
        if subthema_clean in layer["types"]:
            return int(layer["rank"])
    return None


def _group_axis_order(gdf: gpd.GeoDataFrame, group_ids: list[int]) -> tuple[float, float]:
    """
    Maak een stabiele ruimtelijke fallback voor gelijkwaardige groepen.

    Deze functie wordt alleen gebruikt als twee kandidaten dezelfde afstand én
    dezelfde hiërarchische rang hebben. De exacte waarde is minder belangrijk dan
    de stabiliteit: dezelfde inputdata moet dezelfde toewijzing geven.
    """
    if not group_ids:
        return (99999.9, 0.0)

    group_nodes = gdf.loc[group_ids]
    min_hm = group_nodes["hm_sort"].min() if "hm_sort" in group_nodes.columns else 99999.9
    if min_hm >= 90000.0:
        min_hm = 99999.9

    try:
        center = group_nodes.geometry.union_all().centroid
    except AttributeError:
        center = group_nodes.geometry.unary_union.centroid

    return (float(min_hm), float(center.x))


def _build_primary_groups(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """
    Bouw eerst alléén de primaire ruggengraatgroepen.

    In de oude app groeide iedere groep direct door naar secundaire objecten.
    Daardoor kon een secundair object toevallig terechtkomen bij de groep die als
    eerste door de loop kwam. Deze functie doet dat bewust niet: eerst worden alle
    primaire groepen vastgesteld, daarna pas worden secundaire objecten toegewezen.
    """
    groups: dict[str, dict[str, Any]] = {}
    node_to_group: dict[int, str] = {}

    for layer in HIERARCHY_CONFIG:
        rank = int(layer["rank"])
        target_types = layer["types"]
        prefix = layer["prefix"]

        candidates = [
            node
            for node in graph.nodes
            if node in gdf.index
            and gdf.loc[node, "subthema_clean"] in target_types
            and node not in node_to_group
            and not is_maintenance_project_exempt(gdf.loc[node])
        ]

        if not candidates:
            continue

        graph_sub = graph.subgraph(candidates).copy()

        # Knip de ruggengraat zodra fundamentele segmentatiekenmerken wijzigen.
        edges_to_remove = []
        for left, right in graph_sub.edges():
            if _get_segmentation_hash(gdf, left) != _get_segmentation_hash(gdf, right):
                edges_to_remove.append((left, right))

        graph_sub.remove_edges_from(edges_to_remove)

        # Connected components zijn sets. We sorteren ze voor voorspelbare output.
        components = sorted(nx.connected_components(graph_sub), key=lambda component: min(component))

        for index, component in enumerate(components):
            group_id = f"{prefix}_{rank}_{index}"
            primary_ids = sorted(component)

            first_node = gdf.loc[primary_ids[0]]
            seg_props = _get_segmentation_hash(gdf, primary_ids[0])

            groups[group_id] = {
                "ids": list(primary_ids),
                "primary_ids": list(primary_ids),
                "secondary_ids": [],
                "subthema": target_types[0],
                "rank": rank,
                "prefix": prefix,
                "reason": _reason_text(seg_props),
                "current_project": clean_display_value(first_node.get("Onderhoudsproject", "")),
                "seg_props": seg_props,
                "spatial_sort_val": 0,
                "assignment_note": "Primaire ruggengraatgroep; secundaire objecten apart toegewezen.",
            }

            for node in primary_ids:
                node_to_group[node] = group_id

    return groups, node_to_group


def _is_assignable_secondary(gdf: gpd.GeoDataFrame, node_id: int, backbone_types: set[str]) -> bool:
    """
    Bepaal of een object als secundair object aan een onderhoudscomplex mag hangen.
    """
    if node_id not in gdf.index:
        return False

    row = gdf.loc[node_id]
    if is_maintenance_project_exempt(row):
        return False

    return row.get("subthema_clean", "") not in backbone_types


def _find_best_group_for_secondary(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    start_node: int,
    primary_node_to_group: dict[int, str],
    groups: dict[str, dict[str, Any]],
    backbone_types: set[str],
) -> str | None:
    """
    Zoek de beste primaire groep voor één secundair object.

    Toewijzingsregel:
    1. kortste graafafstand naar een primaire ruggengraat;
    2. bij gelijke afstand wint de hiërarchie: rijstrook > parallelweg/busbaan/
       landbouwpad > fietspad;
    3. bij volledig gelijke kandidaten gebruiken we een stabiele ruimtelijke
       fallback, zodat het resultaat niet afhankelijk is van dictionary-volgorde.

    Waarom afstand vóór rang? De hiërarchie uit het werkproces gaat over objecten
    die aan meerdere primaire objecten grenzen. Voor indirecte ketens van
    secundaire objecten is de kortste topologische afstand veiliger; anders kan
    een ver weg gelegen rijstrook een secundair object bij een direct aangrenzend
    fietspad wegtrekken.
    """
    visited: set[int] = {start_node}
    queue: deque[tuple[int, int]] = deque([(start_node, 0)])
    candidates: list[tuple[int, int, float, float, str]] = []

    while queue:
        current_node, distance = queue.popleft()

        for neighbor in graph.neighbors(current_node):
            if neighbor not in gdf.index or neighbor in visited:
                continue

            neighbor_group = primary_node_to_group.get(neighbor)
            if neighbor_group:
                group = groups[neighbor_group]
                stable_hm, stable_x = _group_axis_order(gdf, group.get("primary_ids", group["ids"]))
                candidates.append(
                    (
                        distance + 1,
                        int(group.get("rank", 99)),
                        stable_hm,
                        stable_x,
                        neighbor_group,
                    )
                )
                # Primaire groepen zijn eindpunten: we lopen niet door een
                # ruggengraat heen naar een ander onderhoudscomplex.
                visited.add(neighbor)
                continue

            if not _is_assignable_secondary(gdf, neighbor, backbone_types):
                visited.add(neighbor)
                continue

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][4]


def _assign_secondary_objects(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    groups: dict[str, dict[str, Any]],
    node_to_group: dict[int, str],
) -> None:
    """
    Wijs secundaire objecten toe nadat alle primaire groepen bekend zijn.

    Deze werkwijze voorkomt dat de uitkomst afhankelijk wordt van de volgorde
    waarin groepen door de code worden afgelopen.
    """
    backbone_types = _all_backbone_types()

    # Bewaar een onveranderlijke kaart van primaire knopen naar groepen.
    # Secundaire objecten die al eerder zijn toegewezen mogen namelijk niet als
    # nieuwe 'bron' fungeren voor andere secundaire objecten; anders ontstaat
    # alsnog volgorde-afhankelijk gedrag.
    primary_node_to_group = dict(node_to_group)

    secondary_nodes = sorted(
        node
        for node in graph.nodes
        if node in gdf.index
        and node not in node_to_group
        and _is_assignable_secondary(gdf, node, backbone_types)
    )

    for node in secondary_nodes:
        best_group_id = _find_best_group_for_secondary(
            gdf,
            graph,
            node,
            primary_node_to_group,
            groups,
            backbone_types,
        )

        if best_group_id is None:
            continue

        groups[best_group_id]["ids"].append(node)
        groups[best_group_id].setdefault("secondary_ids", []).append(node)
        node_to_group[node] = best_group_id


def generate_grouped_proposals(gdf: gpd.GeoDataFrame, graph: nx.Graph) -> dict[str, dict[str, Any]]:
    """
    Genereer onderhoudsprojectvoorstellen.

    Werkwijze:
    1. Bouw per primaire laag de ruggengraatgroepen;
    2. knip die ruggengraat bij veranderende segmentatiekenmerken;
    3. wijs secundaire objecten toe via afstand en hiërarchie;
    4. sorteer de groepen op rang, hectometrering en ruimtelijke tie-breaker.
    """
    if gdf is None or gdf.empty or graph is None:
        return {}

    groups, node_to_group = _build_primary_groups(gdf, graph)

    if not groups:
        return {}

    _assign_secondary_objects(gdf, graph, groups, node_to_group)

    road_label = str(gdf["Wegnummer"].iloc[0]) if "Wegnummer" in gdf.columns and not gdf.empty else "Onbekend"
    direction_code = ROAD_DIRECTIONS.get(road_label, "UNKNOWN")

    for group_id, group_data in groups.items():
        # Houd de volgorde stabiel en voorkom dubbele objecten.
        group_data["ids"] = list(dict.fromkeys(group_data["ids"]))
        group_data["primary_ids"] = list(dict.fromkeys(group_data.get("primary_ids", [])))
        group_data["secondary_ids"] = list(dict.fromkeys(group_data.get("secondary_ids", [])))

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
