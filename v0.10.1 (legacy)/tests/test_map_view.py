import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from iasset_tool.map_view import build_road_map
from iasset_tool.utils import normalize_text


def _gdf(rows):
    """Maak een minimale kaartlaag voor kaartopbouw-tests."""
    prepared = []
    for idx, row in enumerate(rows):
        item = {
            "sys_id": idx,
            "subthema": row.get("subthema", "rijstrook"),
            "Onderhoudsproject": row.get("Onderhoudsproject", ""),
            "verhardingssoort": row.get("verhardingssoort", "asfalt"),
            "Soort deklaag specifiek": row.get("Soort deklaag specifiek", "SMA"),
            "Jaar aanleg": row.get("Jaar aanleg", "2020"),
            "Jaar deklaag": row.get("Jaar deklaag", "2020"),
            "Besteknummer": row.get("Besteknummer", "B-001"),
            "geometry": Point(175000 + idx, 565000 + idx),
        }
        item["subthema_clean"] = normalize_text(item["subthema"])
        prepared.append(item)

    gdf = gpd.GeoDataFrame(prepared, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_build_road_map_accepts_selected_object_and_future_kwargs():
    """
    De Streamlit-schil mag extra kaartopties doorgeven zonder TypeError.

    Dit voorkomt regressie als `app.py` en `map_view.py` bij een lokale update
    tijdelijk niet exact in dezelfde versie staan.
    """
    gdf = _gdf([
        {"Onderhoudsproject": "N398-HRB-00.4-01.0"},
        {"Onderhoudsproject": ""},
    ])

    result = build_road_map(
        gdf,
        selected_object_id=1,
        pdok_hm=None,
        toekomstige_optie=True,
    )

    assert result.network_node_count == 0
    assert result.folium_map is not None


def test_build_road_map_sanitizes_missing_iasset_values():
    """
    Ontbrekende iASSET-waarden zoals pd.NA mogen de GeoJSON-laag niet laten crashen.
    """
    gdf = _gdf([
        {
            "Onderhoudsproject": pd.NA,
            "Soort deklaag specifiek": pd.NA,
            "Besteknummer": pd.NA,
        }
    ])

    result = build_road_map(gdf)

    html = result.folium_map.get_root().render()
    assert "GeoJSON" in html or "geo_json" in html


def test_build_road_map_accepts_point_zoom_bounds():
    """
    Een selectie met punt-bounds moet een klein zoomvenster krijgen in plaats van te crashen.
    """
    gdf = _gdf([{"Onderhoudsproject": "N398-HRB-00.4-01.0"}])

    result = build_road_map(gdf, zoom_bounds=(5.0, 53.0, 5.0, 53.0))

    assert result.folium_map is not None
