import geopandas as gpd
from shapely.geometry import Point

from iasset_tool.changes import (
    build_export_dataframe,
    get_export_profile_columns,
    summarize_export_profile,
)


def _gdf():
    gdf = gpd.GeoDataFrame(
        {
            "sys_id": [0, 1],
            "bron_id": ["a", "b"],
            "nummer": ["OBJ-A", "OBJ-B"],
            "Onderhoudsproject": ["oud A", "nieuw B"],
            "verhardingssoort": ["asfalt", "beton"],
            "Wegnummer": ["N398", "N398"],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:28992",
    )
    return gdf.set_index("sys_id", drop=False)


def test_export_profile_columns_falls_back_to_default_for_unknown_profile():
    assert get_export_profile_columns("bestaat niet") == get_export_profile_columns("Onderhoudsprojecten")


def test_build_export_dataframe_uses_selected_profile_columns():
    gdf = _gdf()
    columns = get_export_profile_columns("Onderhoudsprojecten")

    export = build_export_dataframe(gdf, [0], export_columns=columns)

    assert list(export.columns) == ["id", "nummer", "Onderhoudsproject"]
    assert export.iloc[0]["id"] == "a"


def test_export_summary_counts_unchanged_written_values():
    gdf = _gdf()
    change_log = [
        {
            "ID": 0,
            "Veld": "Onderhoudsproject",
            "Oud": "oud A",
            "Nieuw": "nieuw A",
            "Status": "Succes",
        },
        {
            "ID": 1,
            "Veld": "verhardingssoort",
            "Oud": "asfalt",
            "Nieuw": "beton",
            "Status": "Succes",
        },
    ]

    summary = summarize_export_profile(gdf, change_log, "Onderhoudsprojecten")

    assert summary.changed_object_count == 2
    assert summary.changed_cell_count == 2
    assert summary.changed_cells_in_profile_count == 1
    assert summary.written_value_count == 6  # 2 objecten * 3 profielkolommen
    assert summary.unchanged_written_value_count == 5
    assert summary.omitted_changed_fields == ["verhardingssoort"]
