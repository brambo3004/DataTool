from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from iasset_tool.project_axis import build_project_axis_diagnostics


def _wegas() -> gpd.GeoDataFrame:
    """Maak een simpele iASSET-wegas van 1,0 tot 2,0 km."""
    return gpd.GeoDataFrame(
        [
            {
                "nummer": "WA-N398",
                "naam": "N398",
                "Wegnummer": "N398",
                "geometry": LineString([(0, 0), (1000, 0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )


def _hectopoints() -> gpd.GeoDataFrame:
    """Maak NWB-hectopunten; hectomtrng 10 betekent hm 1,0."""
    rows = []
    for hm in range(10, 21):
        rows.append(
            {
                "hectomtrng": hm,
                "objectid": hm,
                "wvk_id": 1000 + hm,
                "geometry": Point((hm - 10) * 100, 2),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992")


def test_projectgrens_wordt_geijkt_op_iasset_wegas_en_zone_wordt_gemeld() -> None:
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.2-01.8",
                "geometry": LineString([(200, 0), (500, 0)]),
            },
            {
                "sys_id": "b",
                "nummer": "RS-2",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.2-01.8",
                "geometry": LineString([(500, 0), (800, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )

    zones = pd.DataFrame(
        [
            {
                "nummer": "WA-N398",
                "zone_id": "WA-N398-Z01",
                "afstand_van_m": 200.0,
                "afstand_tot_m": 200.0,
                "kleurklasse": "oranje",
            }
        ]
    )

    result = build_project_axis_diagnostics(
        road_gdf,
        _wegas(),
        _hectopoints(),
        zones,
        "N398",
        boundary_zone_buffer_m=5.0,
        length_tolerance_m=10.0,
    )

    assert not result.calibration_anchors.empty
    assert not result.project_boundaries.empty

    boundary = result.project_boundaries.iloc[0]
    assert boundary["Onderhoudsproject"] == "N398-HRB-01.2-01.8"
    assert boundary["axis_id"] == "WA-N398"
    assert boundary["as_begin_m"] == 200.0
    assert boundary["as_eind_m"] == 800.0
    assert boundary["begin_zone_kleur"] == "oranje"
    assert boundary["begin_zone_id"] == "WA-N398-Z01"
    assert boundary["status"] == "aandacht"
    assert boundary["fysiek_object_begin_km"] == 1.2
    assert boundary["fysiek_object_eind_km"] == 1.8


def test_projectdekking_signaleert_gat_tussen_projecten() -> None:
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.0-01.4",
                "geometry": LineString([(0, 0), (400, 0)]),
            },
            {
                "sys_id": "b",
                "nummer": "RS-2",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.5-02.0",
                "geometry": LineString([(500, 0), (1000, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )

    result = build_project_axis_diagnostics(
        road_gdf,
        _wegas(),
        _hectopoints(),
        pd.DataFrame(),
        "N398",
        gap_tolerance_m=5.0,
        length_tolerance_m=10.0,
    )

    gap_rows = result.project_coverage[result.project_coverage["controle_type"] == "gat"]
    assert len(gap_rows) == 1
    gap = gap_rows.iloc[0]
    assert gap["van_m"] == 400.0
    assert gap["tot_m"] == 500.0
    assert gap["lengte_m"] == 100.0
    assert gap["status"] == "controleer"


def test_lege_en_corrupte_bronnen_crashen_niet() -> None:
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "bad",
                "nummer": "RS-X",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.0-01.2",
                "geometry": None,
            }
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )

    wegassen = gpd.GeoDataFrame(
        [
            {
                "nummer": "WA-N398",
                "naam": "N398",
                "Wegnummer": "N398",
                "geometry": None,
            }
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )

    result = build_project_axis_diagnostics(
        road_gdf,
        wegassen,
        gpd.GeoDataFrame(geometry=[], crs="EPSG:28992"),
        None,
        "N398",
    )

    assert not result.project_boundaries.empty
    assert result.project_boundaries.iloc[0]["status"] == "controleer"
    assert "geen bruikbare ijking" in result.project_boundaries.iloc[0]["waarschuwing"]
    assert "Geen bruikbare iASSET-wegasgeometrie" in result.warning
