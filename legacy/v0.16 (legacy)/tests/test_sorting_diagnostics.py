import geopandas as gpd
from shapely.geometry import LineString, Point

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
            "advisor_sort_m": 0.0,
            "advisor_sort_basis": "primary_route_sort_m",
            "advisor_sort_fallback_m": 10.0,
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
    assert row["advisor_sort_m"] == 0.0
    assert row["advisor_sort_basis"] == "primary_route_sort_m"
    assert row["advisor_sort_fallback_m"] == 10.0
    assert row["advisor_sort_terugval_vorige"] is False or bool(row["advisor_sort_terugval_vorige"]) is False
    assert row["tie_breaker_source"] == "lokale_route_as"



def _primary_secondary_route_gdf():
    """Maak een kleine route-as met één secundair object als uitschieter."""
    rows = [
        {
            "sys_id": 0,
            "Wegnummer": "N398",
            "nummer": "P-0",
            "subthema": "rijstrook",
            "subthema_clean": normalize_text("rijstrook"),
            "Wegvaknum": "1",
            "Metrering": "1.0",
            "Situering": "R",
            "hm_sort": 1.0,
            "geometry": Point(0, 0),
        },
        {
            "sys_id": 1,
            "Wegnummer": "N398",
            "nummer": "P-1",
            "subthema": "rijstrook",
            "subthema_clean": normalize_text("rijstrook"),
            "Wegvaknum": "1",
            "Metrering": "1.1",
            "Situering": "R",
            "hm_sort": 1.1,
            "geometry": Point(10000, 0),
        },
        {
            "sys_id": 2,
            "Wegnummer": "N398",
            "nummer": "S-2",
            "subthema": "bermverharding",
            "subthema_clean": normalize_text("bermverharding"),
            "Wegvaknum": "1",
            "Metrering": "1.0",
            "Situering": "R",
            "hm_sort": 1.0,
            "geometry": Point(5000, 0),
        },
    ]

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_route_mid_backtracking_is_separate_from_route_sort_backtracking():
    """
    v0.14.2 mag een route_mid-terugval niet meer presenteren als sorteerfout
    wanneer route_sort_m zelf netjes oploopt.
    """
    gdf = _diagnose_gdf()
    groups = {
        "GRP_RIJBAAN_30": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 39905.13,
            "route_mid_m": 45031.21,
            "route_end_m": 45049.00,
            "route_sort_m": 39905.13,
            "route_sort_bron": "route_start_m",
            "route_sort_verklaarbaar": True,
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 39905.13,
            "tie_breaker_source": "lokale_route_as_overlapcluster",
            "sort_mode": "hm_overlap_route",
            "hm_min_sort": 26.9,
            "hm_max_sort": 29.6,
        },
        "GRP_RIJBAAN_31": {
            "ids": [1],
            "primary_ids": [1],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 43469.95,
            "route_mid_m": 43549.88,
            "route_end_m": 43560.00,
            "route_sort_m": 43469.95,
            "route_sort_bron": "route_start_m",
            "route_sort_verklaarbaar": True,
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 43469.95,
            "tie_breaker_source": "lokale_route_as_overlapcluster",
            "sort_mode": "hm_overlap_route",
            "hm_min_sort": 28.0,
            "hm_max_sort": 28.1,
        },
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    row = group_diag[group_diag["groep"] == "GRP_RIJBAAN_31"].iloc[0]

    assert row["route_mid_terugval_vorige"] is True or bool(row["route_mid_terugval_vorige"]) is True
    assert row["route_sort_terugval_vorige"] is False or bool(row["route_sort_terugval_vorige"]) is False
    assert row["route_terugval_vorige"] is False or bool(row["route_terugval_vorige"]) is False
    assert "diagnoseverschil" in row["waarschuwing"]
    assert "WAARSCHUWING: route_sort_m ligt vóór" not in row["waarschuwing"]


def test_compact_group_with_large_route_span_gets_outlier_warning():
    """
    Een compact hm-bereik met kilometers verschil tussen start en mid/eind moet
    als geometrische of gekoppelde-object-anomalie zichtbaar worden.
    """
    gdf = _diagnose_gdf()
    groups = {
        "GRP_RIJBAAN_40": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 39904.60,
            "route_mid_m": 45101.60,
            "route_end_m": 45124.59,
            "route_sort_m": 39904.60,
            "route_sort_bron": "route_start_m",
            "route_sort_verklaarbaar": True,
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 39904.60,
            "tie_breaker_source": "lokale_route_as_overlapcluster",
            "sort_mode": "hm_overlap_route",
            "hm_min_sort": 29.6,
            "hm_max_sort": 29.6,
        }
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    row = group_diag.iloc[0]

    assert "groot verschil tussen route_start_m" in row["route_outlier_warning"]
    assert "groot verschil tussen route_start_m" in row["waarschuwing"]


def test_primary_and_all_object_route_fields_expose_secondary_route_outlier():
    """
    De diagnose moet laten zien of de sorteerruggengraat primair is, terwijl
    secundaire objecten een andere alle-object-route geven.
    """
    gdf = _primary_secondary_route_gdf()
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0, 2],
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
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 0.0,
            "tie_breaker_source": "lokale_route_as",
            "sort_mode": "hm_route",
            "hm_min_sort": 1.0,
            "hm_max_sort": 1.0,
        }
    }

    object_diag, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    row = group_diag.iloc[0]

    assert "object_route_span_m" in object_diag.columns
    assert row["route_basis"] == "primary_ids"
    assert row["primary_route_sort_m"] == 0.0
    assert row["all_route_sort_m"] > 1000.0
    assert row["primary_all_route_delta_m"] > 1000.0
    assert "primaire-route en alle-object-route verschillen" in row["route_outlier_warning"]


def test_sort_diagnostic_warning_texts_are_version_independent():
    """Gebruikersteksten in de diagnose mogen geen verouderde versienummers tonen."""
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
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 0.0,
            "tie_breaker_source": "lokale_route_as_overlapcluster",
            "sort_mode": "hm_overlap_route",
            "hm_min_sort": 1.0,
            "hm_max_sort": 1.0,
            "overlap_sort_applied": True,
            "overlap_cluster_size": 2,
        }
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    warning_text = " ".join(group_diag["waarschuwing"].astype(str).tolist())

    assert "v0.13" not in warning_text
    assert "v0.14" not in warning_text
    assert "overlapcluster-sortering actief" in warning_text


def test_group_diagnostics_reports_advisor_sort_backtracking_separately():
    """
    v0.15 toont een eventuele terugval in de Project Adviseur-sleutel apart,
    zodat route- en advisor-diagnose niet door elkaar lopen.
    """
    gdf = _diagnose_gdf()
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 10.0,
            "route_mid_m": 10.0,
            "route_end_m": 10.0,
            "route_sort_m": 10.0,
            "route_sort_bron": "route_mid_m",
            "advisor_sort_m": 10.0,
            "advisor_sort_basis": "primary_route_sort_m",
            "advisor_sort_fallback_m": 0.0,
            "tie_breaker_source": "lokale_route_as",
            "sort_mode": "hm_route",
            "hm_min_sort": 1.0,
            "hm_max_sort": 1.0,
        },
        "GRP_RIJBAAN_2": {
            "ids": [1],
            "primary_ids": [1],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 9.0,
            "route_mid_m": 9.0,
            "route_end_m": 9.0,
            "route_sort_m": 9.0,
            "route_sort_bron": "route_mid_m",
            "advisor_sort_m": 9.0,
            "advisor_sort_basis": "primary_route_sort_m",
            "advisor_sort_fallback_m": 0.0,
            "tie_breaker_source": "lokale_route_as",
            "sort_mode": "hm_route",
            "hm_min_sort": 1.1,
            "hm_max_sort": 1.1,
        },
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    second = group_diag.iloc[1]

    assert bool(second["advisor_sort_terugval_vorige"]) is True
    assert bool(second["route_sort_terugval_vorige"]) is True
    assert "advisor_sort_m" in group_diag.columns
    assert "advisor_sort_basis" in group_diag.columns



def test_object_route_outlier_is_exported_as_warning_for_primary_object():
    """
    v0.15.1: objecten met een extreem grote route-span moeten niet alleen een
    booleaanse vlag krijgen, maar ook in de waarschuwing/aandachtspunten-export
    terug te vinden zijn.
    """
    rows = [
        {
            "sys_id": 0,
            "Wegnummer": "N398",
            "nummer": "P-anker-0",
            "subthema": "rijstrook",
            "subthema_clean": normalize_text("rijstrook"),
            "Wegvaknum": "1",
            "Metrering": "0.0",
            "Situering": "R",
            "hm_sort": 0.0,
            "geometry": Point(0, 0),
        },
        {
            "sys_id": 1,
            "Wegnummer": "N398",
            "nummer": "P-outlier",
            "subthema": "rijstrook",
            "subthema_clean": normalize_text("rijstrook"),
            "Wegvaknum": "1",
            "Metrering": "1.0",
            "Situering": "R",
            "hm_sort": 1.0,
            "geometry": LineString([(1000, 0), (2500, 0)]),
        },
        {
            "sys_id": 2,
            "Wegnummer": "N398",
            "nummer": "P-anker-2",
            "subthema": "rijstrook",
            "subthema_clean": normalize_text("rijstrook"),
            "Wegvaknum": "1",
            "Metrering": "2.0",
            "Situering": "R",
            "hm_sort": 2.0,
            "geometry": Point(4000, 0),
        },
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992").set_index("sys_id", drop=False)
    gdf.index.name = None

    object_diag, _, _ = build_sort_diagnostics(gdf, {}, selected_road="N398")
    outlier = object_diag[object_diag["nummer"] == "P-outlier"].iloc[0]

    assert bool(outlier["object_route_outlier"]) is True
    assert outlier["sort_severity"] == "waarschuwing"
    assert "primair object heeft extreem grote route-span" in outlier["sort_warning"]


def test_group_diagnostics_exposes_advisor_sort_correction_metadata():
    """De groepsdiagnose toont de ruwe én gecorrigeerde Project Adviseur-sleutel."""
    gdf = _diagnose_gdf()
    groups = {
        "GRP_RIJBAAN_1": {
            "ids": [0],
            "primary_ids": [0],
            "rank": 1,
            "subthema": "rijstrook",
            "layer_label": "rijstrook",
            "route_start_m": 39904.60,
            "route_mid_m": 45101.60,
            "route_end_m": 45124.59,
            "route_sort_m": 45101.60,
            "route_sort_bron": "route_mid_m_gecorrigeerd_start_outlier",
            "advisor_sort_m": 45101.60,
            "advisor_sort_raw_m": 39904.60,
            "advisor_sort_correctie": "compacte_groep_route_start_outlier_gecorrigeerd",
            "advisor_sort_basis": "primary_route_sort_m",
            "advisor_sort_fallback_m": 0.0,
            "fallback_sort_m": 0.0,
            "tie_breaker_dist": 45101.60,
            "tie_breaker_source": "lokale_route_as_overlapcluster",
            "sort_mode": "hm_overlap_route",
            "hm_min_sort": 29.6,
            "hm_max_sort": 29.6,
            "overlap_sort_applied": True,
            "overlap_cluster_size": 2,
        }
    }

    _, group_diag, _ = build_sort_diagnostics(gdf, groups, selected_road="N398")
    row = group_diag.iloc[0]

    assert row["advisor_sort_m"] == 45101.60
    assert row["advisor_sort_raw_m"] == 39904.60
    assert row["advisor_sort_correctie"] == "compacte_groep_route_start_outlier_gecorrigeerd"
    assert "sorteersleutel gecorrigeerd" in row["waarschuwing"]
