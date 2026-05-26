"""
Project Adviseur.

Deze module maakt voorstelgroepen voor onderhoudscomplexen.
De Streamlit UI bepaalt alleen hoe de gebruiker zo'n groep bekijkt en accepteert.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd

from .config import (
    FRIENDLY_LABELS,
    HIERARCHY_CONFIG,
    ROAD_DIRECTIONS,
    SEGMENTATION_ATTRIBUTES,
)
from .domain import is_maintenance_project_exempt
from .fietspad import FietspadClassification, FietspadProjectRole, classify_fietspaden
from .sorting_diagnostics import build_local_axis, project_geometry_range_on_axis
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


def _dominant_subthema_for_nodes(gdf: gpd.GeoDataFrame, node_ids: list[int], fallback: str) -> str:
    """
    Bepaal het dominante primaire subthema binnen een groep.

    De parallel-laag bevat parallelweg, landbouwpad en busbaan. Voor de gebruiker
    is het verwarrend als een landbouwpadgroep als 'parallelweg' wordt getoond.
    Daarom bepalen we het label uit de objecten zelf.
    """
    counts: dict[str, int] = {}

    for node_id in node_ids:
        if node_id not in gdf.index:
            continue
        subthema = str(gdf.loc[node_id].get("subthema_clean", "")).lower().strip()
        if not subthema:
            continue
        counts[subthema] = counts.get(subthema, 0) + 1

    if not counts:
        return fallback

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _subthema_list_for_nodes(gdf: gpd.GeoDataFrame, node_ids: list[int]) -> list[str]:
    """Geef een stabiele lijst met primaire subthema's in een groep."""
    values = {
        str(gdf.loc[node_id].get("subthema_clean", "")).lower().strip()
        for node_id in node_ids
        if node_id in gdf.index and str(gdf.loc[node_id].get("subthema_clean", "")).strip()
    }
    return sorted(values)


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


def _route_axis_tie_breaker(
    gdf: gpd.GeoDataFrame,
    group_ids: list[int],
    axis,
) -> tuple[float | None, float | None, float | None]:
    """
    Bepaal de positie van een groep langs de lokaal afgeleide route-as.

    We projecteren de geometrieën van de primaire objecten op de route-as en
    gebruiken de mediane positie als tie-breaker binnen dezelfde hectometrering.
    De start/eindpositie bewaren we voor diagnose en debug. Als projectie niet
    lukt, geeft de functie ``None`` terug zodat de oude globale X/Y-fallback
    gebruikt blijft worden.
    """
    if axis is None or not group_ids:
        return None, None, None

    positions_start: list[float] = []
    positions_mid: list[float] = []
    positions_end: list[float] = []

    for object_id in group_ids:
        if object_id not in gdf.index:
            continue

        start_m, mid_m, end_m, _ = project_geometry_range_on_axis(gdf.loc[object_id].geometry, axis)
        if start_m is None or mid_m is None or end_m is None:
            continue

        positions_start.append(float(start_m))
        positions_mid.append(float(mid_m))
        positions_end.append(float(end_m))

    if not positions_mid:
        return None, None, None

    positions_mid.sort()
    median_index = len(positions_mid) // 2
    if len(positions_mid) % 2:
        route_mid = positions_mid[median_index]
    else:
        route_mid = (positions_mid[median_index - 1] + positions_mid[median_index]) / 2

    return min(positions_start), float(route_mid), max(positions_end)


def _route_tie_breaker_is_usable(route_mid: float | None) -> bool:
    """
    Controleer of de lokale routepositie bruikbaar is als sorteerwaarde.

    We eisen bewust alleen dat de projectie beschikbaar is. Als twee groepen
    exact dezelfde routepositie krijgen, blijft de globale X/Y-fallback als
    tweede tie-breaker in de sort-key staan.
    """
    return route_mid is not None



def _hm_range_for_group(gdf: gpd.GeoDataFrame, group_ids: list[int]) -> tuple[float, float]:
    """
    Bepaal het geldige hectometerbereik van een adviesgroep.

    We gebruiken dit vanaf v0.13 om overlappende groepen te herkennen. Lege of
    corrupte hm-waarden krijgen een hoge fallback, zodat de app blijft draaien
    bij wisselende iASSET-exports.
    """
    if not group_ids or "hm_sort" not in gdf.columns:
        return 99999.9, 99999.9

    values = pd.to_numeric(gdf.loc[group_ids, "hm_sort"], errors="coerce")
    valid_values = values[values < 90000.0].dropna()
    if valid_values.empty:
        return 99999.9, 99999.9

    return float(valid_values.min()), float(valid_values.max())


def _safe_route_value(value: Any) -> float | None:
    """Zet een routepositie om naar float of None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _overlap_cluster_route_key(group_data: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Sorteersleutel binnen een overlappend hm-cluster.

    Binnen overlappende hectometerbereiken is ``hm_min`` te grof: een lang
    segment met een lage begin-hectometrering kan ruimtelijk later liggen dan
    een korte knip binnen dezelfde omgeving. Daarom gebruiken we dan de positie
    langs de lokale route-as, bij voorkeur het beginpunt van de groep.
    """
    route_start = _safe_route_value(group_data.get("route_start_m"))
    route_mid = _safe_route_value(group_data.get("route_mid_m"))
    hm_min = float(group_data.get("hm_min_sort", group_data.get("sort_value", 99999.9)) or 99999.9)
    fallback = float(group_data.get("fallback_tie_breaker_dist", group_data.get("axis_tie_breaker_dist", 0.0)) or 0.0)

    route_start_sort = route_start if route_start is not None else float("inf")
    route_mid_sort = route_mid if route_mid is not None else float("inf")
    return (route_start_sort, route_mid_sort, hm_min, fallback)


def _regular_group_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float, float]:
    """Sorteersleutel voor niet-overlappende groepen."""
    _, group_data = item
    return (
        int(group_data.get("rank", 99)),
        float(group_data.get("hm_min_sort", group_data.get("sort_value", 99999.9)) or 99999.9),
        float(group_data.get("tie_breaker_dist", 0.0) or 0.0),
        float(group_data.get("fallback_tie_breaker_dist", 0.0) or 0.0),
    )


def _sort_groups_with_overlap_clusters(
    group_items: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Sorteer adviesgroepen met speciale behandeling voor overlappende hm-bereiken.

    v0.12 gebruikte de lokale route-as alleen als tie-breaker bij exact dezelfde
    ``hm_min``. De N354 liet zien dat groepen ook overlappende hm-bereiken kunnen
    hebben, bijvoorbeeld een lang segment dat over een korte extra knip heen valt.
    In zo'n overlapcluster bepaalt de lokale routepositie daarom de volgorde.
    """
    if not group_items:
        return []

    result: list[tuple[str, dict[str, Any]]] = []
    items_by_rank: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    for item in group_items:
        _, group_data = item
        items_by_rank[int(group_data.get("rank", 99))].append(item)

    for rank in sorted(items_by_rank):
        rank_items = sorted(
            items_by_rank[rank],
            key=lambda item: (
                float(item[1].get("hm_min_sort", 99999.9) or 99999.9),
                float(item[1].get("hm_max_sort", 99999.9) or 99999.9),
                float(item[1].get("tie_breaker_dist", 0.0) or 0.0),
                item[0],
            ),
        )

        cluster: list[tuple[str, dict[str, Any]]] = []
        cluster_hm_max = float("-inf")
        cluster_number = 0

        def flush_cluster() -> None:
            nonlocal cluster, cluster_hm_max, cluster_number
            if not cluster:
                return

            cluster_number += 1
            has_overlap = len(cluster) > 1
            if has_overlap:
                sorted_cluster = sorted(cluster, key=lambda item: _overlap_cluster_route_key(item[1]))
                cluster_id = f"R{rank}-{cluster_number}"

                for _, data in sorted_cluster:
                    data["overlap_cluster_id"] = cluster_id
                    data["overlap_sort_applied"] = True
                    data["overlap_cluster_size"] = len(sorted_cluster)

                    route_key = _overlap_cluster_route_key(data)
                    if route_key[0] != float("inf") or route_key[1] != float("inf"):
                        data["sort_mode"] = "hm_overlap_route"
                        data["tie_breaker_source"] = "lokale_route_as_overlapcluster"
                        # Bewaar de gebruikte routewaarde ook in de algemene
                        # tie-breaker, zodat debugtabellen dezelfde richting tonen.
                        data["tie_breaker_dist"] = route_key[0] if route_key[0] != float("inf") else route_key[1]
                    else:
                        data["tie_breaker_source"] = "globale_richting_fallback"
            else:
                sorted_cluster = sorted(cluster, key=_regular_group_sort_key)
                for _, data in sorted_cluster:
                    data["overlap_cluster_id"] = ""
                    data["overlap_sort_applied"] = False
                    data["overlap_cluster_size"] = 1

            result.extend(sorted_cluster)
            cluster = []
            cluster_hm_max = float("-inf")

        for item in rank_items:
            _, data = item
            hm_min = float(data.get("hm_min_sort", data.get("sort_value", 99999.9)) or 99999.9)
            hm_max = float(data.get("hm_max_sort", hm_min) or hm_min)

            same_hm_start_as_cluster = any(
                abs(
                    hm_min
                    - float(existing_data.get("hm_min_sort", existing_data.get("sort_value", 99999.9)) or 99999.9)
                )
                <= 0.0001
                for _, existing_data in cluster
            )
            overlaps_cluster = hm_min < cluster_hm_max - 0.0001

            if cluster and not (overlaps_cluster or same_hm_start_as_cluster):
                flush_cluster()

            cluster.append(item)
            cluster_hm_max = max(cluster_hm_max, hm_max)

        flush_cluster()

    return result


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


def _fietspad_is_attached_to_main_project(
    gdf: gpd.GeoDataFrame,
    node_id: int,
    fietspad_classes: dict[int, FietspadClassification],
) -> bool:
    """
    Bepaal of een fietspad géén eigen ruggengraatgroep moet vormen.

    Alleen fietspaden die de classifier voldoende duidelijk als kruisend,
    rotonde-/kruispuntgebonden of anderszins gekoppeld aan de hoofdrijbaan ziet,
    worden als secundair object behandeld. Bij twijfel blijft het fietspad een
    eigen voorstelgroep zodat de databeheerder het kan controleren.
    """
    if node_id not in gdf.index:
        return False

    subthema = str(gdf.loc[node_id].get("subthema_clean", "")).lower().strip()
    if subthema != "fietspad":
        return False

    classification = fietspad_classes.get(int(node_id))
    return (
        classification is not None
        and classification.role == FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT
    )


def _node_is_primary_candidate_for_layer(
    gdf: gpd.GeoDataFrame,
    node_id: int,
    target_types: list[str],
    fietspad_classes: dict[int, FietspadClassification],
) -> bool:
    """
    Bepaal of een object ruggengraatkandidaat is voor de huidige laag.

    Dit is vooral relevant voor fietspaden. Niet elk object met subthema
    'fietspad' krijgt een eigen onderhoudsproject: haakse of rotondegebonden
    fietspaden worden later als secundair object aan de hoofdrijbaan gekoppeld.
    """
    if node_id not in gdf.index:
        return False

    subthema = str(gdf.loc[node_id].get("subthema_clean", "")).lower().strip()
    if subthema not in target_types:
        return False

    if subthema == "fietspad" and _fietspad_is_attached_to_main_project(gdf, node_id, fietspad_classes):
        return False

    return True


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
    fietspad_classes: dict[int, FietspadClassification],
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
            and _node_is_primary_candidate_for_layer(gdf, node, target_types, fietspad_classes)
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

            assignment_note = "Primaire ruggengraatgroep; secundaire objecten apart toegewezen."
            review_needed = False

            if target_types == ["fietspad"]:
                fietspad_notes = [
                    fietspad_classes[int(node)].reason
                    for node in primary_ids
                    if int(node) in fietspad_classes
                ]
                if any(
                    fietspad_classes[int(node)].role == FietspadProjectRole.UNKNOWN_KEEP_OWN
                    for node in primary_ids
                    if int(node) in fietspad_classes
                ):
                    review_needed = True
                    assignment_note = (
                        "Fietspad blijft voorlopig een eigen voorstelgroep, "
                        "maar de parallel-/kruisingclassificatie is onzeker."
                    )
                elif fietspad_notes:
                    assignment_note = fietspad_notes[0]

            dominant_subthema = _dominant_subthema_for_nodes(gdf, primary_ids, target_types[0])

            groups[group_id] = {
                "ids": list(primary_ids),
                "primary_ids": list(primary_ids),
                "secondary_ids": [],
                "attached_fietspad_ids": [],
                "subthema": dominant_subthema,
                "layer_label": " / ".join(target_types),
                "subthema_lijst": _subthema_list_for_nodes(gdf, primary_ids),
                "rank": rank,
                "prefix": prefix,
                "reason": _reason_text(seg_props),
                "current_project": clean_display_value(first_node.get("Onderhoudsproject", "")),
                "seg_props": seg_props,
                "spatial_sort_val": 0,
                "assignment_note": assignment_note,
                "review_needed": review_needed,
            }

            for node in primary_ids:
                node_to_group[node] = group_id

    return groups, node_to_group


def _is_assignable_secondary(
    gdf: gpd.GeoDataFrame,
    node_id: int,
    backbone_types: set[str],
    fietspad_classes: dict[int, FietspadClassification],
) -> bool:
    """
    Bepaal of een object als secundair object aan een onderhoudscomplex mag hangen.
    """
    if node_id not in gdf.index:
        return False

    row = gdf.loc[node_id]
    if is_maintenance_project_exempt(row):
        return False

    subthema = str(row.get("subthema_clean", "")).lower().strip()

    if subthema == "fietspad":
        return _fietspad_is_attached_to_main_project(gdf, node_id, fietspad_classes)

    return subthema not in backbone_types


def _find_best_group_for_secondary(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    start_node: int,
    primary_node_to_group: dict[int, str],
    groups: dict[str, dict[str, Any]],
    backbone_types: set[str],
    fietspad_classes: dict[int, FietspadClassification],
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

            if not _is_assignable_secondary(gdf, neighbor, backbone_types, fietspad_classes):
                visited.add(neighbor)
                continue

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    if not candidates:
        # Fallback voor duidelijke haakse/rotondegebonden fietspaden die wel
        # geometrisch aan een hoofdroute zijn gekoppeld, maar door kleine
        # topologische gaten geen graafpad hebben. Dit voorkomt dat zo'n
        # fietspad uit alle adviesgroepen verdwijnt.
        classification = fietspad_classes.get(int(start_node))
        if (
            classification is not None
            and classification.role == FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT
            and classification.nearest_primary_id in primary_node_to_group
        ):
            return primary_node_to_group[classification.nearest_primary_id]

        return None

    candidates.sort()
    return candidates[0][4]


def _assign_secondary_objects(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    groups: dict[str, dict[str, Any]],
    node_to_group: dict[int, str],
    fietspad_classes: dict[int, FietspadClassification],
) -> None:
    """
    Wijs secundaire objecten toe nadat alle primaire groepen bekend zijn.

    Performance:
    v0.9 zocht per secundair object met een losse BFS naar de beste ruggengraat.
    Bij grotere wegen betekent dat veel herhaald netwerkwerk. Deze versie doet
    één multi-source BFS vanaf alle primaire ruggengraatknopen tegelijk. De
    sorteerregel blijft hetzelfde: kortste afstand wint, daarna de hiërarchie
    rijstrook > parallelweg/busbaan/landbouwpad > fietspad, daarna een stabiele
    ruimtelijke fallback.
    """
    backbone_types = _all_backbone_types()

    # Bewaar een onveranderlijke kaart van primaire knopen naar groepen.
    primary_node_to_group = dict(node_to_group)

    if not primary_node_to_group:
        return

    secondary_nodes = {
        node
        for node in graph.nodes
        if node in gdf.index
        and node not in node_to_group
        and _is_assignable_secondary(gdf, node, backbone_types, fietspad_classes)
    }

    if not secondary_nodes:
        return

    group_order: dict[str, tuple[int, float, float, str]] = {}
    for group_id, group in groups.items():
        stable_hm, stable_x = _group_axis_order(gdf, group.get("primary_ids", group.get("ids", [])))
        group_order[group_id] = (
            int(group.get("rank", 99)),
            stable_hm,
            stable_x,
            group_id,
        )

    best_candidate_by_node: dict[int, tuple[int, int, float, float, str]] = {}
    queue: deque[tuple[int, int, str]] = deque()

    for primary_node, group_id in primary_node_to_group.items():
        if primary_node in graph:
            queue.append((primary_node, 0, group_id))

    while queue:
        current_node, distance, group_id = queue.popleft()
        rank, stable_hm, stable_x, stable_group_id = group_order.get(group_id, (99, 99999.9, 0.0, group_id))

        try:
            neighbors = graph.neighbors(current_node)
        except Exception:
            continue

        for neighbor in neighbors:
            if neighbor not in secondary_nodes:
                continue

            candidate = (
                distance + 1,
                rank,
                stable_hm,
                stable_x,
                stable_group_id,
            )

            previous = best_candidate_by_node.get(neighbor)
            if previous is not None and previous <= candidate:
                continue

            best_candidate_by_node[neighbor] = candidate
            queue.append((neighbor, distance + 1, group_id))

    for node in sorted(secondary_nodes):
        best_candidate = best_candidate_by_node.get(node)

        if best_candidate is None:
            # Fallback voor fietspaden die door de classifier aan het hoofdproject
            # zijn gekoppeld, maar door kleine topologische gaten geen graafpad
            # hebben naar de primaire ruggengraat.
            classification = fietspad_classes.get(int(node))
            if (
                classification is not None
                and classification.role == FietspadProjectRole.ATTACHED_TO_MAIN_PROJECT
                and classification.nearest_primary_id in primary_node_to_group
            ):
                best_group_id = primary_node_to_group[classification.nearest_primary_id]
            else:
                continue
        else:
            best_group_id = best_candidate[4]

        groups[best_group_id]["ids"].append(node)
        groups[best_group_id].setdefault("secondary_ids", []).append(node)

        if str(gdf.loc[node].get("subthema_clean", "")).lower().strip() == "fietspad":
            groups[best_group_id].setdefault("attached_fietspad_ids", []).append(node)

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

    fietspad_classes = classify_fietspaden(gdf, graph)
    groups, node_to_group = _build_primary_groups(gdf, graph, fietspad_classes)

    if not groups:
        return {}

    _assign_secondary_objects(gdf, graph, groups, node_to_group, fietspad_classes)

    road_label = str(gdf["Wegnummer"].iloc[0]) if "Wegnummer" in gdf.columns and not gdf.empty else "Onbekend"
    direction_code = ROAD_DIRECTIONS.get(road_label, "UNKNOWN")
    axis_result = build_local_axis(gdf, selected_road=road_label)
    local_axis = axis_result.axis

    for group_id, group_data in groups.items():
        # Houd de volgorde stabiel en voorkom dubbele objecten.
        group_data["ids"] = list(dict.fromkeys(group_data["ids"]))
        group_data["primary_ids"] = list(dict.fromkeys(group_data.get("primary_ids", [])))
        group_data["secondary_ids"] = list(dict.fromkeys(group_data.get("secondary_ids", [])))
        group_data["attached_fietspad_ids"] = list(dict.fromkeys(group_data.get("attached_fietspad_ids", [])))

        group_nodes = gdf.loc[group_data["ids"]]
        global_tie_breaker_value = _axis_tie_breaker(gdf, group_data["ids"], direction_code)

        # v0.12: gebruik de lokale route-as als gecontroleerde tie-breaker
        # binnen dezelfde hectometrering. De globale X/Y-richting blijft als
        # fallback bestaan, zodat eindpunt- of rotondegevallen niet instabiel
        # worden wanneer meerdere groepen dezelfde routepositie krijgen.
        route_start, route_mid, route_end = _route_axis_tie_breaker(
            gdf,
            group_data.get("primary_ids") or group_data["ids"],
            local_axis,
        )

        group_data["route_tie_breaker_dist"] = route_mid
        group_data["route_start_m"] = route_start
        group_data["route_mid_m"] = route_mid
        group_data["route_end_m"] = route_end
        group_data["axis_tie_breaker_dist"] = global_tie_breaker_value
        group_data["axis_source"] = axis_result.source

        hm_ids = group_data.get("primary_ids") or group_data["ids"]
        min_hm, max_hm = _hm_range_for_group(gdf, hm_ids)
        group_data["hm_min_sort"] = min_hm
        group_data["hm_max_sort"] = max_hm

        if min_hm < 90000.0:
            group_data["sort_value"] = float(min_hm)
            if _route_tie_breaker_is_usable(route_mid):
                group_data["tie_breaker_dist"] = float(route_mid)
                group_data["fallback_tie_breaker_dist"] = global_tie_breaker_value
                group_data["tie_breaker_source"] = "lokale_route_as"
                group_data["sort_mode"] = "hm_route"
            else:
                group_data["tie_breaker_dist"] = global_tie_breaker_value
                group_data["fallback_tie_breaker_dist"] = 0.0
                group_data["tie_breaker_source"] = "globale_richting_fallback"
                group_data["sort_mode"] = "hm"
        else:
            group_data["sort_value"] = global_tie_breaker_value
            group_data["tie_breaker_dist"] = 0.0
            group_data["fallback_tie_breaker_dist"] = 0.0
            group_data["tie_breaker_source"] = "globale_richting_fallback"
            group_data["sort_mode"] = "axis"

    sorted_groups = _sort_groups_with_overlap_clusters(list(groups.items()))

    final_groups: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}

    for volgorde_nr, (_, data) in enumerate(sorted_groups, start=1):
        prefix = data["prefix"]
        counters[prefix] = counters.get(prefix, 0) + 1
        new_id = f"{prefix}_{counters[prefix]}"
        data["volgorde_nr"] = volgorde_nr
        data["advies_volgorde"] = volgorde_nr
        final_groups[new_id] = data

    return final_groups
