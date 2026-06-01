import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from iasset_tool.reference_axis import (
    build_reference_axis_diagnostics,
    build_reference_axis_from_hectopoints,
)
from iasset_tool.trajectory import round_km_to_nearest_5m


def _hectopoints():
    return gpd.GeoDataFrame(
        {
            "hm_val": [140, 141, 142],
            "wegnummer": ["N354", "N354", "N354"],
        },
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0)],
        crs="EPSG:28992",
    )


def test_reference_axis_builds_from_pdok_hectopoints():
    """PDOK-hectometerpunten vormen een experimentele as met hm-kilometers."""
    axis_result = build_reference_axis_from_hectopoints(_hectopoints(), selected_road="N354")

    assert axis_result.axis is not None
    assert axis_result.anchor_count == 3
    assert axis_result.source == "pdok_hectometerpunten_experimenteel"
    assert axis_result.anchors["hm_km"].tolist() == pytest.approx([14.0, 14.1, 14.2])


def test_reference_axis_diagnostics_projects_rijstrook_without_changing_project_logic():
    """De diagnose berekent referentiemetrering naast de bestaande projectnaam."""
    road = gpd.GeoDataFrame(
        {
            "sys_id": [1],
            "Wegnummer": ["N354"],
            "nummer": ["VV001"],
            "subthema": ["rijstrook"],
            "subthema_clean": ["rijstrook"],
            "Onderhoudsproject": ["N354-HRB-14.0-14.2"],
            "Metrering": ["14.0"],
        },
        geometry=[LineString([(0, 10), (100, 10)])],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)

    result = build_reference_axis_diagnostics(
        road,
        _hectopoints(),
        selected_road="N354",
        max_offset_m=25,
    )

    assert result.axis_result.anchor_count == 3
    assert len(result.object_diagnostics) == 1

    row = result.object_diagnostics.iloc[0]
    assert row["referentie_begin_km"] == pytest.approx(14.0)
    assert row["referentie_eind_km"] == pytest.approx(14.1)
    assert row["referentie_lengte_m"] == pytest.approx(100.0)
    assert row["afstand_tot_as_m"] == pytest.approx(10.0)
    assert row["bronkwaliteit"] == "experimenteel"
    assert row["status"] == "projectie"

    summary = result.project_summary.iloc[0]
    assert summary["Onderhoudsproject"] == "N354-HRB-14.0-14.2"
    assert summary["referentie_lengte_m"] == pytest.approx(100.0)
    assert summary["project_lengte_m"] == pytest.approx(200.0)


def test_reference_axis_diagnostics_does_not_crash_without_hectopoints():
    """Ontbrekende PDOK-data levert lege diagnoseframes en een waarschuwing op."""
    road = gpd.GeoDataFrame(
        {"sys_id": [1], "subthema_clean": ["rijstrook"]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)

    result = build_reference_axis_diagnostics(road, None, selected_road="N354")

    assert result.axis_result.axis is None
    assert result.object_diagnostics.empty
    assert result.project_summary.empty
    assert "Geen PDOK-hectometerpunten" in result.warning or "Geen PDOK-hectometerpunten" in result.axis_result.warning


def test_five_meter_management_rule_boundaries_from_v032_context():
    """Leg de beheerregel uit de v0.32-richting expliciet vast."""
    assert round_km_to_nearest_5m(14.000) == pytest.approx(14.000)
    assert round_km_to_nearest_5m(14.001) == pytest.approx(14.005)
    assert round_km_to_nearest_5m(14.007) == pytest.approx(14.005)
    assert round_km_to_nearest_5m(14.008) == pytest.approx(14.010)
    assert round_km_to_nearest_5m(14.012) == pytest.approx(14.010)
    assert round_km_to_nearest_5m(14.013) == pytest.approx(14.015)
    assert round_km_to_nearest_5m(14.017) == pytest.approx(14.015)
