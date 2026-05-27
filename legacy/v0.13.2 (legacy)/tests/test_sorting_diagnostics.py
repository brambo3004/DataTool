import geopandas as gpd
from shapely.geometry import Point

from iasset_tool.sorting_diagnostics import build_sort_diagnostics
from iasset_tool.utils import normalize_text


def _diagnose_gdf():
    rows = []
    for idx, hm in enumerate(["1.0", "1.1"]):
        rows.append(
            {
                "sys_id": idx,
                "Wegnummer": "N398",
                "nummer": f"VV-N398-{idx}",
                "subthema": "rijstrook",
                "subthema_clean": normalize_text("rijstrook"),
                "Wegvaknum": "1",
                "Metrering": hm,
                "Situering": "R",
                "hm_sort": float(hm),
                "geometry": Point(idx * 100, 0),
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_group_diagnostics_exposes_exact_route_sort_fields():
    """
    De groepsdiagnose moet dezelfde route-sortwaarde tonen als de Project
    Adviseur gebruikt. Daardoor kan een databeheerder verklaren waarom een groep
    in deze volgorde staat.
    """
    gdf = _diagnose_gdf()
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 0.0,
            "route_mid_m": 0.0,
            "route_end_m": 0.0,
            "route_sort_m": 0.0,
            "route_sort_bron": "route_mid_m",
            "route_sort_verklaarbaar": True,
            "fallback_sort_m": 10.0,
            "tie_breaker_dist": 0.0,
            "tie_breaker_source": "lokale_route_as",
            "sort_mode": "hm_route",
            "hm_min_sort": 1.0,
            "hm_max_sort": 1.0,
        }
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")

    assert len(group_diag) == 1
    row = group_diag.iloc[0]
    assert row["route_sort_m"] == 0.0
    assert row["route_sort_bron"] == "route_mid_m"
    assert row["route_sort_verklaarbaar"] is True or bool(row["route_sort_verklaarbaar"]) is True
    assert row["fallback_sort_m"] == 10.0
    assert row["tie_breaker_source"] == "lokale_route_as"
