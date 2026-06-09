import geopandas as gpd
import pytest
from shapely.geometry import LineString

from iasset_tool.trajectory import (
    format_name_hm,
    round_km_to_name_tenth_up,
    round_km_to_nearest_5m,
    trajectory_quantity_for_group,
)


def _base_rows(metreringen, project="N398-HRB-00.0-00.3"):
    return gpd.GeoDataFrame(
        {
            "sys_id": list(range(1, len(metreringen) + 1)),
            "Metrering": metreringen,
            "Onderhoudsproject": [project] * len(metreringen),
            "__overview_value": ["2009"] * len(metreringen),
        },
        geometry=[LineString([(0, i), (1, i)]) for i in range(len(metreringen))],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)


def test_grove_objectmetrering_does_not_override_project_name():
    """Losse paspoortmetrering is vaak grof en mag de projectnaam niet zomaar overstemmen."""
    gdf = _base_rows(["0,1", "0,3"], project="N398-HRB-00.0-00.3")

    quantity = trajectory_quantity_for_group(gdf, gdf)

    assert quantity.length_m == pytest.approx(300.0)
    assert quantity.source == "onderhoudsprojectnaam"
    assert quantity.source_quality == "administratief"
    assert quantity.object_metrering_length_m == pytest.approx(200.0)
    assert quantity.object_metrering_quality == "grof"
    assert quantity.difference_m == pytest.approx(100.0)
    assert "grof afgerond" in quantity.warning


def test_explicit_begin_eind_metrering_remains_preferred():
    """Echte begin/eind-kolommen zijn sterker dan de afgeronde projectnaam."""
    gdf = gpd.GeoDataFrame(
        {
            "sys_id": [1, 2],
            "Metrering begin": [12.300, 13.200],
            "Metrering einde": [13.200, 14.405],
            "Onderhoudsproject": ["N354-HRB-12.3-14.5", "N354-HRB-12.3-14.5"],
            "__overview_value": ["2010", "2010"],
        },
        geometry=[LineString([(0, 0), (900, 0)]), LineString([(900, 0), (2105, 0)])],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)

    quantity = trajectory_quantity_for_group(gdf, gdf)

    assert quantity.length_m == pytest.approx(2105.0)
    assert quantity.source == "objectmetrering 'Metrering begin'-'Metrering einde'"
    assert quantity.source_quality == "precies"
    assert quantity.name_length_m == pytest.approx(2200.0)


def test_rounding_helpers_for_future_project_name_logic():
    """Leg de vijfmeterregel en projectnaamafronding vast voor latere Project Adviseur-logica."""
    assert round_km_to_nearest_5m(14.302) == pytest.approx(14.305)
    assert round_km_to_nearest_5m(14.303) == pytest.approx(14.305)
    assert round_km_to_nearest_5m(14.398) == pytest.approx(14.395)

    assert round_km_to_name_tenth_up(3.800) == pytest.approx(3.8)
    assert round_km_to_name_tenth_up(3.805) == pytest.approx(3.9)
    assert round_km_to_name_tenth_up(3.900) == pytest.approx(3.9)
    assert format_name_hm(4.205) == "04.3"
