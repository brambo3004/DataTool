import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from iasset_tool.overview_map import (
    UNKNOWN_LABEL,
    available_overview_attributes,
    build_overview_map,
    build_value_color_mapping,
    overview_quantity_dataframe,
    render_overview_map_html,
    resolve_overview_attribute,
)


def make_test_gdf():
    return gpd.GeoDataFrame(
        {
            "sys_id": [1, 2, 3],
            "subthema": ["rijstrook", "rijstrook", "fietspad"],
            "subthema_clean": ["rijstrook", "rijstrook", "fietspad"],
            "Jaar deklaag": ["2023", "2018", "2020"],
            "verhardingssoort": ["SMA", "DAB", "SMA"],
            "Soort deklaag specifiek": ["SMA-NL 11B", "AC 16", "SMA-NL 8"],
            "Besteknummer": ["22-40-WB", "18-01-WB", "20-01-WB"],
            "Onderhoudsproject": ["N359-HRB-01.0-02.0", "", "N359-FP-01.0-02.0"],
            "Wegvaknum": ["221", "222", "223"],
            "nummer": ["R1", "R2", "F1"],
            "Wegnummer": ["N359", "N398", "N359"],
        },
        geometry=[
            LineString([(160000, 560000), (160050, 560050)]),
            LineString([(160100, 560100), (160150, 560150)]),
            LineString([(160200, 560200), (160250, 560250)]),
        ],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)


def test_available_overview_attributes_uses_aliases():
    gdf = make_test_gdf()

    attrs = available_overview_attributes(gdf)

    assert "Jaar deklaag" in attrs
    assert "Soort verharding_N" in attrs
    assert resolve_overview_attribute(gdf, "Soort verharding_N") == "verhardingssoort"


def test_value_color_mapping_sorts_numeric_values():
    _, legend_items = build_value_color_mapping(["2023", "2018", UNKNOWN_LABEL, "2020"])

    assert [item.label for item in legend_items] == ["2018", "2020", "2023", UNKNOWN_LABEL]


def test_build_overview_map_filters_to_rijstroken_only():
    gdf = make_test_gdf()

    result = build_overview_map(gdf, "Jaar deklaag")

    assert result.row_count == 2
    assert result.selected_column == "Jaar deklaag"
    assert [item.label for item in result.legend_items] == ["2018", "2023"]


def test_build_overview_map_returns_empty_result_without_rijstroken():
    gdf = make_test_gdf()
    gdf = gdf[gdf["subthema_clean"] == "fietspad"].copy()

    result = build_overview_map(gdf, "Jaar deklaag")

    assert result.row_count == 0
    assert result.selected_column is None


def test_build_overview_map_can_show_multiple_roads():
    gdf = make_test_gdf()

    result = build_overview_map(gdf, "Jaar deklaag")

    assert result.row_count == 2


def test_render_overview_map_html_contains_export_panel():
    gdf = make_test_gdf()
    result = build_overview_map(gdf, "Jaar deklaag")

    html = render_overview_map_html(
        result,
        title="iASSET Overzicht - alle wegen",
        subtitle="Visualisatie: Jaar deklaag",
    )

    assert "<title>iASSET Overzicht - alle wegen</title>" in html
    assert "Visualisatie: Jaar deklaag" in html
    assert "Alleen-lezen export uit de iASSET Advisor" in html

def test_value_color_mapping_uses_gradual_color_ramp():
    mapping, legend_items = build_value_color_mapping(["2023", "2018", UNKNOWN_LABEL, "2020"])

    assert mapping["2018"] == "#2c7bb6"
    assert mapping["2020"] == "#ffff8c"
    assert mapping["2023"] == "#d7191c"
    assert mapping[UNKNOWN_LABEL] == "#bdbdbd"
    assert all(not item.color.startswith("hsl(") for item in legend_items)


def test_single_known_value_gets_middle_ramp_color():
    mapping, _ = build_value_color_mapping(["2023", "2023"])

    assert mapping["2023"] == "#ffff8c"

def test_build_overview_map_adds_length_quantities_for_line_geometry():
    gdf = make_test_gdf()

    result = build_overview_map(gdf, "Jaar deklaag")
    items = {item.label: item for item in result.legend_items}

    assert items["2018"].object_count == 1
    assert items["2018"].length_m == pytest.approx(70.71, rel=0.01)
    assert result.total_length_m == pytest.approx(141.42, rel=0.01)
    assert result.length_source == "lijngeometrie"

    quantity_df = overview_quantity_dataframe(result)
    assert "Lengte (km)" in quantity_df.columns
    assert quantity_df["Aantal objecten"].sum() == 2


def test_build_overview_map_uses_polygon_area_and_width_for_quantities():
    gdf = gpd.GeoDataFrame(
        {
            "sys_id": [1, 2],
            "subthema": ["rijstrook", "rijstrook"],
            "subthema_clean": ["rijstrook", "rijstrook"],
            "Jaar deklaag": ["2020", "2020"],
            "Administratieve breedte": [4.0, 2.0],
        },
        geometry=[
            Polygon([(0, 0), (40, 0), (40, 4), (0, 4)]),
            Polygon([(50, 0), (100, 0), (100, 2), (50, 2)]),
        ],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)

    result = build_overview_map(gdf, "Jaar deklaag")
    item = result.legend_items[0]

    assert item.object_count == 2
    assert item.area_m2 == pytest.approx(260.0)
    assert item.length_m == pytest.approx(90.0)
    assert result.total_area_m2 == pytest.approx(260.0)
    assert result.area_source == "polygongeometrie"
    assert result.length_source == "geschat uit oppervlakte / kolom 'Administratieve breedte'"

