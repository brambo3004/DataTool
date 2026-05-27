import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

from iasset_tool.advisor import generate_grouped_proposals, _resolve_route_backtracking_conflicts, _sort_groups_with_overlap_clusters
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
            "Wegvaknum": row.get("Wegvaknum", ""),
            "Metrering": row.get("Metrering", ""),
            "Situering": row.get("Situering", ""),
            "hm_sort": row.get("hm_sort", 1.0),
            "geometry": row.get("geometry", Point(idx, 0)),
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



def test_project_advisor_uses_local_route_axis_as_tie_breaker_with_same_metrering():
    """
    Bij gelijke hectometrering moet de lokale route-as winnen van de globale X/Y-richting.

    N398 heeft in de oude richtingentabel ETW (oost naar west). Zonder route-as
    zou het object met de hoogste X-waarde eerst komen. De lokale as wordt hier
    opgebouwd uit de metrering en moet daarom object 0 vóór object 1 zetten.
    """
    gdf = _gdf(
        [
            {"subthema": "rijstrook", "Wegvaknum": "1", "Metrering": "1.0", "hm_sort": 1.0, "Jaar aanleg": "2020", "geometry": Point(0, 0)},
            {"subthema": "rijstrook", "Wegvaknum": "1", "Metrering": "1.0", "hm_sort": 1.0, "Jaar aanleg": "2021", "geometry": Point(100, 0)},
            {"subthema": "rijstrook", "Wegvaknum": "1", "Metrering": "1.1", "hm_sort": 1.1, "Jaar aanleg": "2022", "geometry": Point(200, 0)},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)

    groups = generate_grouped_proposals(gdf, graph)
    ordered_primary_ids = [group["primary_ids"][0] for group in groups.values()]

    assert ordered_primary_ids[:2] == [0, 1]
    assert list(groups.values())[0]["sort_mode"] in {"hm_route", "hm_overlap_route"}
    assert list(groups.values())[0]["tie_breaker_source"] in {"lokale_route_as", "lokale_route_as_overlapcluster"}


def test_overlap_cluster_uses_route_position_before_hm_min_order():
    """
    Bij overlappende hm-bereiken mag de laagste hm_min niet blind winnen.

    Dit test de kern van v0.13: binnen een overlapcluster sorteert de app op de
    lokale routepositie. Daardoor komt een korte knip die ruimtelijk eerder ligt
    vóór een lang segment dat op hm_min eerder lijkt te beginnen.
    """
    items = [
        (
            "lange_groep",
            {
                "rank": 1,
                "hm_min_sort": 11.3,
                "hm_max_sort": 13.6,
                "route_start_m": 23665.0,
                "route_mid_m": 23908.0,
                "sort_value": 11.3,
                "tie_breaker_dist": 23908.0,
                "fallback_tie_breaker_dist": 0.0,
            },
        ),
        (
            "korte_knip",
            {
                "rank": 1,
                "hm_min_sort": 11.5,
                "hm_max_sort": 11.5,
                "route_start_m": 22127.0,
                "route_mid_m": 22129.0,
                "sort_value": 11.5,
                "tie_breaker_dist": 22129.0,
                "fallback_tie_breaker_dist": 0.0,
            },
        ),
    ]

    sorted_items = _sort_groups_with_overlap_clusters(items)

    assert [group_id for group_id, _ in sorted_items] == ["korte_knip", "lange_groep"]
    assert sorted_items[0][1]["sort_mode"] == "hm_overlap_route"
    assert sorted_items[0][1]["tie_breaker_source"] == "lokale_route_as_overlapcluster"
    assert sorted_items[0][1]["overlap_sort_applied"] is True


def test_overlap_cluster_marks_stable_fallback_when_route_position_is_not_distinguishing():
    """
    Als twee overlapgroepen exact dezelfde routepositie krijgen, moet de diagnose
    zichtbaar maken dat de lokale route-as niet meer onderscheidend is.

    Dit komt voor bij wegeinden of compacte kruispunt/rotonde-situaties. De app
    mag dan niet doen alsof de lokale route-as de onderlinge volgorde volledig
    verklaart.
    """
    items = [
        (
            "eindpunt_a",
            {
                "rank": 1,
                "hm_min_sort": 6.3,
                "hm_max_sort": 6.3,
                "route_start_m": 6231.0,
                "route_mid_m": 6231.0,
                "route_end_m": 6231.0,
                "sort_value": 6.3,
                "tie_breaker_dist": 6231.0,
                "fallback_tie_breaker_dist": 10.0,
            },
        ),
        (
            "eindpunt_b",
            {
                "rank": 1,
                "hm_min_sort": 6.3,
                "hm_max_sort": 6.3,
                "route_start_m": 6231.0,
                "route_mid_m": 6231.0,
                "route_end_m": 6231.0,
                "sort_value": 6.3,
                "tie_breaker_dist": 6231.0,
                "fallback_tie_breaker_dist": 20.0,
            },
        ),
    ]

    sorted_items = _sort_groups_with_overlap_clusters(items)

    for _, group_data in sorted_items:
        assert group_data["sort_mode"] == "hm_overlap_route_fallback"
        assert group_data["tie_breaker_source"] == "stabiele_fallback"
        assert group_data["route_sort_m"] == 6231.0
        assert group_data["route_sort_bron"] == "route_start_m"
        assert group_data["route_sort_verklaarbaar"] is True
        assert group_data["routepositie_onderscheidend"] is False



def test_route_backtracking_conflict_is_resorted_when_hm_ranges_are_related():
    """
    v0.14 corrigeert compacte plekken waar hm-volgorde en route-as elkaar tegenspreken.

    Dit bootst een N354-achtige situatie na: een lang segment met lagere hm_min
    staat eerst, maar een korte knip binnen hetzelfde hm-bereik ligt ruimtelijk
    eerder langs de lokale route-as.
    """
    items = [
        (
            "lange_groep",
            {
                "rank": 1,
                "hm_min_sort": 26.9,
                "hm_max_sort": 29.6,
                "route_sort_m": 45012.0,
                "route_start_m": 45012.0,
                "route_mid_m": 45031.0,
                "route_end_m": 45049.0,
                "fallback_tie_breaker_dist": 0.0,
                "sort_mode": "hm_route",
                "tie_breaker_source": "lokale_route_as",
            },
        ),
        (
            "korte_knip",
            {
                "rank": 1,
                "hm_min_sort": 28.0,
                "hm_max_sort": 28.1,
                "route_sort_m": 43540.0,
                "route_start_m": 43540.0,
                "route_mid_m": 43549.0,
                "route_end_m": 43560.0,
                "fallback_tie_breaker_dist": 0.0,
                "sort_mode": "hm_route",
                "tie_breaker_source": "lokale_route_as",
            },
        ),
    ]

    sorted_items = _resolve_route_backtracking_conflicts(items)

    assert [group_id for group_id, _ in sorted_items] == ["korte_knip", "lange_groep"]
    assert all(group_data["route_conflict_sort_applied"] for _, group_data in sorted_items)
    assert all(group_data["tie_breaker_source"] == "lokale_route_as_conflictcluster" for _, group_data in sorted_items)


def test_route_backtracking_conflict_does_not_reorder_unrelated_hm_ranges():
    """
    Een route-terugval zonder hm-relatie kan ook een onbetrouwbare lokale as zijn.

    Die situatie moet de app niet automatisch over de hele weg hersorteren.
    """
    items = [
        (
            "groep_a",
            {
                "rank": 1,
                "hm_min_sort": 10.0,
                "hm_max_sort": 10.1,
                "route_sort_m": 5000.0,
                "sort_mode": "hm_route",
            },
        ),
        (
            "groep_b",
            {
                "rank": 1,
                "hm_min_sort": 12.0,
                "hm_max_sort": 12.1,
                "route_sort_m": 4000.0,
                "sort_mode": "hm_route",
            },
        ),
    ]

    sorted_items = _resolve_route_backtracking_conflicts(items)

    assert [group_id for group_id, _ in sorted_items] == ["groep_a", "groep_b"]
    assert "route_conflict_sort_applied" not in sorted_items[0][1]
    assert "route_conflict_sort_applied" not in sorted_items[1][1]


def test_v015_secondary_route_outlier_does_not_drive_project_order():
    """
    v0.15: de zichtbare projectvolgorde wordt bepaald door de primaire
    ruggengraat. Een secundair object dat verderop langs de route ligt, mag de
    groep niet achter een volgende primaire groep trekken.
    """
    gdf = _gdf(
        [
            {
                "subthema": "rijstrook",
                "Wegvaknum": "1",
                "Metrering": "1.0",
                "hm_sort": 1.0,
                "Jaar aanleg": "2020",
                "geometry": Point(0, 0),
            },
            {
                "subthema": "rijstrook",
                "Wegvaknum": "1",
                "Metrering": "1.0",
                "hm_sort": 1.0,
                "Jaar aanleg": "2021",
                "geometry": Point(25, 0),
            },
            {
                "subthema": "rijstrook",
                "Wegvaknum": "1",
                "Metrering": "1.1",
                "hm_sort": 1.1,
                "Jaar aanleg": "2022",
                "geometry": Point(100, 0),
            },
            {
                "subthema": "bermverharding",
                "Wegvaknum": "1",
                "Metrering": "1.0",
                "hm_sort": 1.0,
                "geometry": Point(100, 0),
            },
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf.index)
    graph.add_edge(0, 3, type="lateral")

    groups = generate_grouped_proposals(gdf, graph)
    ordered_primary_ids = [group["primary_ids"][0] for group in groups.values()]

    assert ordered_primary_ids[:2] == [0, 1]

    _, first_group = _group_with_primary(groups, 0)
    assert 3 in first_group["secondary_ids"]
    assert first_group["advisor_sort_basis"] == "primary_route_sort_m"
    assert first_group["advisor_sort_m"] == first_group["route_sort_m"]
    assert first_group["all_route_mid_m"] > first_group["primary_route_mid_m"]

