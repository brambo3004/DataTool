
import geopandas as gpd
from shapely.geometry import Point

from iasset_tool.sorting_diagnostics import build_local_axis, build_sort_diagnostics
from iasset_tool.utils import normalize_text


def _gdf(rows):
    """Maak een kleine GeoDataFrame voor sorteerdiagnosetests."""
    prepared = []
    for idx, row in enumerate(rows):
        item = {
            "sys_id": idx,
            "Wegnummer": row.get("Wegnummer", "N398"),
            "nummer": row.get("nummer", f"OBJ-{idx}"),
            "subthema": row.get("subthema", "rijstrook"),
            "Wegvaknum": row.get("Wegvaknum", ""),
            "Metrering": row.get("Metrering", ""),
            "Situering": row.get("Situering", ""),
            "hm_sort": row.get("hm_sort", 99999.9),
            "Onderhoudsproject": row.get("Onderhoudsproject", ""),
            "geometry": row.get("geometry", Point(idx * 10, 0)),
        }
        item["subthema_clean"] = normalize_text(item["subthema"])
        prepared.append(item)

    gdf = gpd.GeoDataFrame(prepared, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_local_axis_groups_parallel_lanes_by_wegvak_and_metrering():
    gdf = _gdf(
        [
            {"Wegvaknum": "1", "Metrering": "1.0", "hm_sort": 1.0, "geometry": Point(0, 0)},
            {"Wegvaknum": "1", "Metrering": "1.0", "hm_sort": 1.0, "geometry": Point(0, 5)},
            {"Wegvaknum": "1", "Metrering": "1.1", "hm_sort": 1.1, "geometry": Point(100, 0)},
            {"Wegvaknum": "1", "Metrering": "1.1", "hm_sort": 1.1, "geometry": Point(100, 5)},
        ]
    )

    axis_result = build_local_axis(gdf, selected_road="N398")

    assert axis_result.axis is not None
    assert axis_result.anchor_count == 2
    assert axis_result.axis.length > 0


def test_sort_diagnostics_flags_multiple_primary_objects_in_same_bucket():
    gdf = _gdf(
        [
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(0, 0)},
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(0, 4)},
            {"Wegvaknum": "7", "Metrering": "1.1", "hm_sort": 1.1, "Situering": "Rechts", "geometry": Point(100, 0)},
        ]
    )
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0, 1],
            "primary_ids": [0, 1],
            "rank": 1,
            "subthema": "rijstrook",
            "sort_mode": "hm",
            "tie_breaker_dist": 0,
        }
    }

    object_df, group_df, axis_result = build_sort_diagnostics(gdf, groups, selected_road="N398")

    duplicate_rows = object_df[object_df["sys_id"].isin([0, 1])]
    assert set(duplicate_rows["bucket_count"]) == {2}
    assert duplicate_rows["sort_warning"].str.contains("meerdere objecten").all()
    assert group_df.loc[0, "sort_quality"] == "middel"
    assert "binnenvakvolgorde" in group_df.loc[0, "waarschuwing"]
    assert axis_result.axis is not None


def test_sort_diagnostics_handles_missing_wegvak_and_metrering_columns():
    gdf = _gdf(
        [
            {"geometry": Point(0, 0)},
            {"geometry": Point(100, 0)},
        ]
    ).drop(columns=["Wegvaknum", "Metrering"])

    object_df, group_df, axis_result = build_sort_diagnostics(gdf, {}, selected_road="N398")

    assert not object_df.empty
    assert group_df.empty
    assert "mist metrering" in object_df.loc[0, "sort_warning"]
    assert axis_result.source in {"primaire_objecten", "onvoldoende_ankerpunten"}


def test_sort_diagnostics_marks_duplicate_bucket_as_info_when_route_position_distinguishes():
    gdf = _gdf(
        [
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(0, 0)},
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(40, 0)},
            {"Wegvaknum": "7", "Metrering": "1.1", "hm_sort": 1.1, "Situering": "Rechts", "geometry": Point(100, 0)},
        ]
    )
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0, 1],
            "primary_ids": [0, 1],
            "rank": 1,
            "subthema": "rijstrook",
            "sort_mode": "hm",
            "tie_breaker_dist": 0,
        }
    }

    object_df, group_df, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")

    duplicate_rows = object_df[object_df["sys_id"].isin([0, 1])]
    assert duplicate_rows["sort_warning"].str.contains("INFO: meerdere primaire objecten").all()
    assert "lokale routepositie lijkt bruikbaar" in group_df.loc[0, "waarschuwing"]



def test_sort_diagnostics_separates_severity_from_warning_text():
    gdf = _gdf(
        [
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(0, 0)},
            {"Wegvaknum": "7", "Metrering": "1.0", "hm_sort": 1.0, "Situering": "Rechts", "geometry": Point(40, 0)},
            {"Wegvaknum": "7", "Metrering": "1.1", "hm_sort": 1.1, "Situering": "Rechts", "geometry": Point(100, 0)},
        ]
    )

    object_df, _, _ = build_sort_diagnostics(gdf, {}, selected_road="N398")
    duplicate_rows = object_df[object_df["sys_id"].isin([0, 1])]

    assert set(duplicate_rows["sort_severity"]) == {"info"}
    assert set(duplicate_rows["Wegnummer"]) == {"N398"}


def test_sort_diagnostics_shows_dominant_subthema_for_parallel_layer():
    gdf = _gdf(
        [
            {
                "subthema": "landbouwpad",
                "Wegvaknum": "1",
                "Metrering": "1.0",
                "hm_sort": 1.0,
                "geometry": Point(0, 0),
            }
        ]
    )
    groups = {
        "GRP_PARALLEL_1": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 2,
            "subthema": "parallelweg",
            "layer_label": "parallelweg / landbouwpad / busbaan",
            "sort_mode": "hm",
            "tie_breaker_dist": 0,
        }
    }

    _, group_df, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")

    assert group_df.loc[0, "dominant_subthema"] == "landbouwpad"
    assert group_df.loc[0, "project_prefix"] == "LBP"
