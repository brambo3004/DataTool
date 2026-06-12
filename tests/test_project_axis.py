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


def test_projectdekking_negeert_gat_zonder_primaire_objecten() -> None:
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
    assert gap_rows.empty




def test_projectdekking_signaleert_gat_met_primaire_objecten_zonder_project() -> None:
    """Een gat wordt pas controlepunt als daar fysiek een primair object ligt."""
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
                "sys_id": "gap",
                "nummer": "RS-GAT",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "",
                "geometry": LineString([(420, 0), (480, 0)]),
            },
            {
                "sys_id": "b",
                "nummer": "RS-2",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.7-02.0",
                "geometry": LineString([(700, 0), (1000, 0)]),
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
    assert gap["tot_m"] == 700.0
    assert gap["lengte_m"] == 300.0
    assert gap["hard_gat_van_m"] == 402.5
    assert gap["hard_gat_tot_m"] == 602.5
    assert gap["hard_gat_lengte_m"] == 200.0
    assert gap["status"] == "controleer"
    assert "RS-GAT" in gap["advies"]


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


def test_overlap_wordt_alleen_binnen_hetzelfde_projecttype_gemeld() -> None:
    """HRB en FPR mogen dezelfde hectometrering hebben zonder overlapmelding."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "hrb",
                "nummer": "RS-HRB",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.0-01.5",
                "geometry": LineString([(0, 0), (500, 0)]),
            },
            {
                "sys_id": "fpr",
                "nummer": "FP-R",
                "subthema": "fietspad",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-FPR-01.2-01.6",
                "geometry": LineString([(200, 40), (600, 40)]),
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
        max_object_offset_m=60.0,
        gap_tolerance_m=5.0,
        length_tolerance_m=10.0,
    )

    overlap_rows = result.project_coverage[result.project_coverage["controle_type"] == "overlap"]
    assert overlap_rows.empty


def test_overlap_binnen_hetzelfde_projecttype_blijft_zichtbaar() -> None:
    """Twee HRB-projecten met overlappende hm-range blijven een controlepunt."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.0-01.6",
                "geometry": LineString([(0, 0), (600, 0)]),
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

    overlap_rows = result.project_coverage[result.project_coverage["controle_type"] == "overlap"]
    assert len(overlap_rows) == 1
    assert overlap_rows.iloc[0]["project_type"] == "HRB"
    assert overlap_rows.iloc[0]["status"] == "controleer"


def test_projectnaam_validatie_en_afrondregel_zijn_apart_zichtbaar() -> None:
    """Een foutieve naamvorm wordt controleer en de objectnaamregel gebruikt omhoog afronden."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                # Mist voorloopnul in beginmetrering en is daarom geen beheerconforme naamvorm.
                "Onderhoudsproject": "N398-HRB-1.2-01.8",
                "geometry": LineString([(201, 0), (800, 0)]),
            }
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
        length_tolerance_m=10.0,
    )

    boundary = result.project_boundaries.iloc[0]
    assert boundary["naam_validatie_status"] == "controleer"
    assert boundary["status_projectnaam"] == "controleer"
    assert boundary["status"] == "controleer"
    assert "voorloopnul" in boundary["naam_validatie_melding"]
    # 1.201 km ligt binnen 2,5 m van hm 1.2 en wordt daarom naar hm 1.2 gesnapt.
    assert boundary["object_begin_naamregel"] == "01.2"
    assert bool(boundary["object_begin_gesnapt_naar_hm"]) is True


def test_objectligging_overschreeuwt_projectgrensstatus_niet() -> None:
    """Objectafwijking blijft zichtbaar, maar maakt de hoofdstatus niet geel."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "fp",
                "nummer": "FP-1",
                "subthema": "fietspad",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-FPR-01.2-01.8",
                "geometry": LineString([(200, 60), (800, 60)]),
            }
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
        max_object_offset_m=40.0,
        length_tolerance_m=10.0,
    )

    boundary = result.project_boundaries.iloc[0]
    assert boundary["status_projectgrens"] == "ok"
    assert boundary["objectligging_status"] == "controleer"
    assert boundary["status"] == "ok"


def test_bblr_wordt_als_gecombineerde_situering_toegestaan() -> None:
    """BBLR blijft voorlopig toegestaan als gecombineerde links/rechts-situering."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "bb",
                "nummer": "BB-1",
                "subthema": "busbaan",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-BBLR-01.2-01.8",
                "geometry": LineString([(200, 0), (800, 0)]),
            }
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
        length_tolerance_m=10.0,
    )

    boundary = result.project_boundaries.iloc[0]
    assert boundary["project_family"] == "BB"
    assert boundary["situering"] == "LR"
    assert boundary["naam_validatie_status"] == "ok"
    assert boundary["status_projectnaam"] == "ok"



def test_projectnaamregel_snapt_nabije_grens_naar_hectometerpunt() -> None:
    """Een fysieke grens binnen 2,5 m van een hm-punt gebruikt dat hm-punt."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.2-01.8",
                "geometry": LineString([(202, 0), (800, 0)]),
            }
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
        length_tolerance_m=10.0,
    )

    boundary = result.project_boundaries.iloc[0]
    assert boundary["object_begin_naamregel"] == "01.2"
    assert boundary["object_begin_dichtstbijzijnde_hm"] == 1.2
    assert boundary["object_begin_snap_afstand_m"] == 2.0
    assert bool(boundary["object_begin_gesnapt_naar_hm"]) is True


def test_projectnaamregel_rondt_naar_boven_buiten_snap_tolerantie() -> None:
    """Een fysieke grens buiten 2,5 m van een hm-punt gebruikt de naar-boven-regel."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "a",
                "nummer": "RS-1",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.2-01.8",
                "geometry": LineString([(203, 0), (800, 0)]),
            }
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
        length_tolerance_m=10.0,
    )

    boundary = result.project_boundaries.iloc[0]
    assert boundary["object_begin_naamregel"] == "01.3"
    assert boundary["object_begin_snap_afstand_m"] == 3.0
    assert bool(boundary["object_begin_gesnapt_naar_hm"]) is False


def test_gatcontrole_trekt_projectnaamzone_af_voordat_objecten_worden_getoetst() -> None:
    """Objecten in de toegestane naamzone van de rechter projectgrens geven geen hard gat."""
    road_gdf = gpd.GeoDataFrame(
        [
            {
                "sys_id": "links",
                "nummer": "RS-L",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.0-01.4",
                "geometry": LineString([(0, 0), (400, 0)]),
            },
            {
                "sys_id": "rechts",
                "nummer": "RS-R",
                "subthema": "rijstrook",
                "Wegnummer": "N398",
                "Onderhoudsproject": "N398-HRB-01.5-02.0",
                # 1.404 km valt binnen de naamzone die nog als 1.5 wordt geschreven.
                "geometry": LineString([(404, 0), (1000, 0)]),
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
        gap_tolerance_m=1.0,
        length_tolerance_m=10.0,
    )

    gap_rows = result.project_coverage[result.project_coverage["controle_type"] == "gat"]
    assert gap_rows.empty
