import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from iasset_tool.map_view import build_road_map


def make_test_gdf():
    """Maak een minimale weglaag voor kaarttests."""
    return gpd.GeoDataFrame(
        {
            "sys_id": [1],
            "subthema": ["rijstrook"],
            "Onderhoudsproject": ["N398-HRB-01.0-02.0"],
            "verhardingssoort": ["asfalt"],
            "Soort deklaag specifiek": [pd.Timestamp("2020-01-01")],
            "Jaar aanleg": [pd.Timestamp("2015-01-01")],
            "Jaar deklaag": [pd.NaT],
            "Besteknummer": ["16-30-WB"],
        },
        geometry=[LineString([(160000, 560000), (160050, 560050)])],
        crs="EPSG:28992",
    ).set_index("sys_id", drop=False)


def test_build_road_map_accepts_non_json_native_paspoort_values():
    """
    Folium kan pandas-waarden zoals Timestamp niet rechtstreeks naar GeoJSON
    serialiseren. De kaartlaag moet die waarden eerst omzetten naar veilige
    tekst, zodat een enkele Excel-datum de app niet onderuit haalt.
    """
    gdf = make_test_gdf()

    result = build_road_map(gdf)

    assert result.folium_map is not None



def test_projectvoorstel_highlight_accepts_csv_like_object_ids():
    """
    De v0.35.2-kaartinspectie krijgt object-id's uit de voorsteltoewijzing.
    Na CSV-export/import kunnen die als tekst of 1.0 binnenkomen; de kaart mag
    dan niet crashen en moet de projectvoorstel-legenda toevoegen.
    """
    gdf = make_test_gdf()

    result = build_road_map(
        gdf,
        selected_project_proposal_object_ids=["1.0"],
        selected_project_proposal_existing_object_ids=["2"],
    )

    html = result.folium_map.get_root().render()
    assert "Projectvoorstel-inspectie" in html
