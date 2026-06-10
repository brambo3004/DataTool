import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from iasset_tool.nwb import (
    build_nwb_source_summary,
    compare_iasset_wegassen_to_nwb,
    compare_iasset_wegassen_to_nwb_detail,
    build_nwb_wegas_deviation_zones,
    build_nwb_wegas_detail_map,
    filter_iasset_wegassen_for_road,
    filter_nwb_hectopunten_for_wegvakken,
    filter_nwb_wegvakken_for_road,
    read_wegassen_geojson_bytes,
)


def _nwb_wegvakken():
    return gpd.GeoDataFrame(
        {
            "wvk_id": [1001.0, 1002.0, 2001.0],
            "wegnummer": ["N354", "N354", "N398"],
            "wegnr_hmp": ["N354", "N354", "N398"],
            "routenr": [354, 354, 398],
            "beginkm": [8.6, 8.7, 0.1],
            "eindkm": [8.7, 8.8, 0.2],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(1000, 0), (1100, 0)]),
        ],
        crs="EPSG:28992",
    )


def _nwb_hectopunten():
    return gpd.GeoDataFrame(
        {
            "wvk_id": [1001, 1002, 2001, 9999],
            "hectomtrng": [86, 87, 1, 999],
            "afstand": [0, 0, 0, 0],
            "zijde": ["M", "M", "M", "M"],
        },
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(1000, 0),
            Point(9999, 9999),
        ],
        crs="EPSG:28992",
    )


def test_filter_nwb_wegvakken_for_road_uses_multiple_road_columns():
    """NWB kan wegnummer/routenummer in meerdere kolommen hebben."""
    filtered = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")

    assert len(filtered) == 2
    assert set(filtered["wvk_id"].astype(int).tolist()) == {1001, 1002}


def test_filter_hectopunten_uses_wvk_id_coupling():
    """Hectopunten worden via wvk_id aan de geselecteerde wegvakken gekoppeld."""
    wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    filtered = filter_nwb_hectopunten_for_wegvakken(_nwb_hectopunten(), wegvakken)

    assert len(filtered) == 2
    assert filtered["hectomtrng"].tolist() == [86, 87]


def test_nwb_source_summary_reports_counts_and_hm_range():
    """De bronsamenvatting legt de eerste NWB-verkenning uitlegbaar vast."""
    wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    hectopunten = filter_nwb_hectopunten_for_wegvakken(_nwb_hectopunten(), wegvakken)

    summary = build_nwb_source_summary("N354", wegvakken, hectopunten)

    row = summary.iloc[0]
    assert row["nwb_wegvakken"] == 2
    assert row["nwb_hectopunten"] == 2
    assert row["unieke_wvk_ids"] == 2
    assert row["hectopunt_min_km"] == pytest.approx(8.6)
    assert row["hectopunt_max_km"] == pytest.approx(8.7)
    assert row["status"] == "bron_gevonden"
    assert row["bronkwaliteit"] == "extern_nwb_ogc_api_experimenteel"


def test_iasset_wegassen_geojson_can_be_read_and_filtered():
    """De iASSET-wegassen-GeoJSON is de veilige lijnbron voor de interne as."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"nummer": "WA-N354", "naam": "N354", "Wegnummer": "N354"},
                "geometry": {"type": "LineString", "coordinates": [(5.0, 53.0), (5.001, 53.0)]},
            },
            {
                "type": "Feature",
                "properties": {"nummer": "WA-N398", "naam": "N398", "Wegnummer": "N398"},
                "geometry": {"type": "LineString", "coordinates": [(5.1, 53.0), (5.101, 53.0)]},
            },
        ],
    }

    import json

    wegassen = read_wegassen_geojson_bytes(json.dumps(payload).encode("utf-8"))
    filtered = filter_iasset_wegassen_for_road(wegassen, "N354")

    assert len(wegassen) == 2
    assert len(filtered) == 1
    assert filtered.iloc[0]["nummer"] == "WA-N354"




def test_iasset_wegassen_geojson_with_utf8_bom_can_be_read():
    """iASSET/GeoJSON-exports kunnen een UTF-8 BOM bevatten."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"nummer": "WA-N354", "naam": "N354"},
                "geometry": {"type": "LineString", "coordinates": [(5.0, 53.0), (5.001, 53.0)]},
            }
        ],
    }

    import json

    data = "\ufeff" + json.dumps(payload)
    wegassen = read_wegassen_geojson_bytes(data.encode("utf-8"))

    assert len(wegassen) == 1
    assert wegassen.iloc[0]["nummer"] == "WA-N354"


def test_filter_iasset_wegassen_accepts_suffix_variants():
    """Wegassen zoals N354_1 horen bij N354, niet bij N3541."""
    wegassen = gpd.GeoDataFrame(
        {
            "nummer": ["WA-N354", "WA-N354_1", "WA-N398"],
            "naam": ["N354", "N354_1", "N398"],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(1000, 0), (1100, 0)]),
        ],
        crs="EPSG:28992",
    )

    filtered = filter_iasset_wegassen_for_road(wegassen, "N354")

    assert set(filtered["nummer"].tolist()) == {"WA-N354", "WA-N354_1"}


def test_compare_iasset_wegas_to_nwb_marks_near_axis_as_comparison():
    """Een iASSET-wegas dicht bij NWB krijgt geen controlewaarschuwing."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"], "Wegnummer": ["N354"]},
        geometry=[LineString([(0, 2), (200, 2)])],
        crs="EPSG:28992",
    )

    comparison = compare_iasset_wegassen_to_nwb(wegassen, nwb_wegvakken, "N354", max_distance_m=25)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["status"] == "vergelijking"
    assert row["afstand_max_sample_tot_nwb_m"] == pytest.approx(2.0)


def test_compare_iasset_wegas_to_nwb_marks_far_axis_as_control():
    """Een wegas die ruim naast NWB ligt blijft zichtbaar als controlepunt."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"], "Wegnummer": ["N354"]},
        geometry=[LineString([(0, 80), (200, 80)])],
        crs="EPSG:28992",
    )

    comparison = compare_iasset_wegassen_to_nwb(wegassen, nwb_wegvakken, "N354", max_distance_m=25)

    assert comparison.iloc[0]["status"] == "controleer"
    assert "wijkt ruimtelijk af" in comparison.iloc[0]["waarschuwing"]



def test_compare_iasset_wegas_to_nwb_detail_localises_far_sample():
    """De detail-export laat zien waar langs de as de afwijking zit."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"], "Wegnummer": ["N354"]},
        geometry=[LineString([(0, 2), (100, 2), (200, 80)])],
        crs="EPSG:28992",
    )

    detail = compare_iasset_wegassen_to_nwb_detail(
        wegassen,
        nwb_wegvakken,
        "N354",
        max_distance_m=25,
        sample_step_m=50,
    )

    assert not detail.empty
    assert {"afstand_langs_iasset_wegas_m", "x_rd", "y_rd", "afstand_tot_nwb_m", "status"}.issubset(detail.columns)
    assert detail["afstand_tot_nwb_m"].max() > 25
    assert "controleer" in set(detail["status"].tolist())


def test_compare_iasset_wegas_to_nwb_detail_links_nearest_wegvak():
    """Elk detailpunt krijgt het dichtstbijzijnde NWB-wegvak als context mee."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"], "Wegnummer": ["N354"]},
        geometry=[LineString([(0, 1), (200, 1)])],
        crs="EPSG:28992",
    )

    detail = compare_iasset_wegassen_to_nwb_detail(
        wegassen,
        nwb_wegvakken,
        "N354",
        max_distance_m=25,
        sample_step_m=100,
    )

    assert not detail.empty
    assert set(detail["dichtstbijzijnde_nwb_wvk_id"].astype(str)).issubset({"1001.0", "1002.0", "1001", "1002"})
    assert set(detail["status"].tolist()) == {"vergelijking"}



def test_build_nwb_wegas_detail_map_returns_folium_map():
    """De detailkaart maakt lokale afwijkingen visueel vindbaar."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"], "Wegnummer": ["N354"]},
        geometry=[LineString([(0, 2), (100, 2), (200, 80)])],
        crs="EPSG:28992",
    )
    detail = compare_iasset_wegassen_to_nwb_detail(
        wegassen,
        nwb_wegvakken,
        "N354",
        max_distance_m=25,
        sample_step_m=100,
    )

    folium_map = build_nwb_wegas_detail_map(
        wegassen,
        nwb_wegvakken,
        detail,
        "N354",
        max_distance_m=25,
    )

    assert folium_map is not None
    assert hasattr(folium_map, "_repr_html_")


def test_build_nwb_wegas_detail_map_returns_none_without_detail():
    """Zonder detailregels wordt geen lege kaart getoond."""
    nwb_wegvakken = filter_nwb_wegvakken_for_road(_nwb_wegvakken(), "N354")
    wegassen = gpd.GeoDataFrame(
        {"nummer": ["WA-N354"], "naam": ["N354"]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:28992",
    )

    assert build_nwb_wegas_detail_map(wegassen, nwb_wegvakken, pd.DataFrame(), "N354") is None


def test_build_nwb_wegas_deviation_zones_clusters_local_attention():
    """Aandachtspunten worden samengevat tot zones voor beheercontrole."""
    detail = pd.DataFrame(
        {
            "nummer": ["WA-N354"] * 6,
            "naam": ["N354"] * 6,
            "Wegnummer": ["N354"] * 6,
            "afstand_langs_iasset_wegas_m": [0, 250, 400, 500, 900, 1000],
            "afstand_tot_nwb_m": [1, 2, 12, 14, 30, 3],
            "status": ["vergelijking", "vergelijking", "vergelijking", "vergelijking", "controleer", "vergelijking"],
            "waarschuwing": ["", "", "", "", "samplepunt ligt verder dan de maximale afstand tot NWB", ""],
        }
    )

    zones = build_nwb_wegas_deviation_zones(detail, "N354", max_distance_m=25)

    assert len(zones) == 2
    assert zones.iloc[0]["kleurklasse"] == "oranje"
    assert zones.iloc[0]["zone_type"] == "lokale_afwijking"
    assert zones.iloc[1]["kleurklasse"] == "rood"
    assert zones.iloc[1]["status"] == "controleer"
    assert "projectgrens" in zones.iloc[1]["relevantie_projectgrenzen"]


def test_build_nwb_wegas_deviation_zones_marks_end_zone():
    """Een rode afwijking aan het einde wordt als eindzone geduid."""
    detail = pd.DataFrame(
        {
            "nummer": ["WA-N354_1"] * 6,
            "naam": ["N354_1"] * 6,
            "Wegnummer": ["N354"] * 6,
            "afstand_langs_iasset_wegas_m": [0, 1000, 2000, 3000, 3900, 4000],
            "afstand_tot_nwb_m": [1, 2, 3, 4, 108, 131],
            "status": ["vergelijking", "vergelijking", "vergelijking", "vergelijking", "controleer", "controleer"],
            "waarschuwing": ["", "", "", "", "samplepunt ligt verder dan de maximale afstand tot NWB", "samplepunt ligt verder dan de maximale afstand tot NWB"],
        }
    )

    zones = build_nwb_wegas_deviation_zones(detail, "N354", max_distance_m=25)

    assert len(zones) == 1
    row = zones.iloc[0]
    assert row["zone_type"] == "eindzone"
    assert row["kleurklasse"] == "rood"
    assert "einde" in row["advies"].lower()
    assert row["relevantie_projectgrenzen"].startswith("hoog")
