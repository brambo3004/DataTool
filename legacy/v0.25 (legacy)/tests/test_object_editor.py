import geopandas as gpd
from shapely.geometry import Point

from iasset_tool.object_editor import (
    editable_fields_for_profile,
    missing_profile_columns,
    object_label,
    object_preview_dataframe,
    search_objects,
)


def _gdf():
    gdf = gpd.GeoDataFrame(
        {
            "sys_id": [0, 1],
            "bron_id": ["bron-a", "bron-b"],
            "nummer": ["OBJ-001", "OBJ-002"],
            "naam": ["Hoofdrijbaan west", "Fietspad oost"],
            "subthema": ["rijstrook", "fietspad"],
            "Onderhoudsproject": ["N398-HRB-00.0-01.0", ""],
            "verhardingssoort": ["asfalt", "beton"],
            "Wegnummer": ["N398", "N398"],
            "Wegvaknum": ["10", "10"],
            "Metrering": ["1.2", "1.3"],
            "gps coordinaten": ["POINT (5 52)", "POINT (5.1 52)"],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:28992",
    )
    return gdf.set_index("sys_id", drop=False)


def test_editable_fields_follow_export_profile_and_skip_geometry_identifiers():
    gdf = _gdf()

    fields = editable_fields_for_profile(gdf, "Onderhoudsprojecten")

    assert fields == ["nummer", "Onderhoudsproject"]
    assert "bron_id" not in fields
    assert "gps coordinaten" not in fields


def test_missing_profile_columns_reports_columns_not_in_active_export():
    gdf = _gdf()

    missing = missing_profile_columns(gdf, "Paspoortdata basis")

    assert "Type onderdeel" in missing
    assert "Onderhoudsproject" not in missing


def test_search_objects_finds_project_and_object_number():
    gdf = _gdf()

    by_project = search_objects(gdf, "HRB-00.0")
    by_number = search_objects(gdf, "OBJ-002")

    assert [result.object_id for result in by_project] == [0]
    assert [result.object_id for result in by_number] == [1]


def test_search_objects_can_limit_to_changed_objects():
    gdf = _gdf()
    change_log = [{"ID": 1, "Veld": "Onderhoudsproject", "Status": "Succes"}]

    results = search_objects(gdf, "", changed_only=True, change_log=change_log)

    assert [result.object_id for result in results] == [1]


def test_object_label_uses_available_passport_fields():
    gdf = _gdf()

    label = object_label(gdf.loc[0])

    assert "ID 0" in label
    assert "OBJ-001" in label
    assert "rijstrook" in label
    assert "wv/hm 10 / 1.2" in label


def test_object_preview_dataframe_skips_missing_fields():
    gdf = _gdf()

    preview = object_preview_dataframe(gdf, 0, ["nummer", "bestaat niet", "Onderhoudsproject"])

    assert list(preview["Veld"]) == ["nummer", "Onderhoudsproject"]
