import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from iasset_tool.rules import check_rules
from iasset_tool.utils import normalize_text


def _gdf(rows):
    """Maak een kleine GeoDataFrame voor regellogica-tests."""
    prepared = []
    for idx, row in enumerate(rows):
        item = {
            "sys_id": idx,
            "subthema": row.get("subthema", ""),
            "Onderhoudsproject": row.get("Onderhoudsproject", ""),
            "Besteknummer": row.get("Besteknummer", ""),
            "geometry": Point(idx, 0),
        }
        item["subthema_clean"] = normalize_text(item["subthema"])
        prepared.append(item)

    gdf = gpd.GeoDataFrame(prepared, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None
    return gdf


def test_exception_without_project_does_not_raise_missing_project():
    gdf = _gdf([{"subthema": "carpoolplaats", "Onderhoudsproject": ""}])

    violations = check_rules(gdf)

    assert violations == []


def test_required_object_with_nan_project_raises_missing_project():
    gdf = _gdf([{"subthema": "rijstrook", "Onderhoudsproject": pd.NA}])

    violations = check_rules(gdf)

    assert len(violations) == 1
    assert violations[0]["msg"] == "Mist verplicht onderhoudsproject"


def test_exception_with_project_raises_wrong_project_warning():
    gdf = _gdf([{"subthema": "perron", "Onderhoudsproject": "N398-HRB-01.2-03.4"}])

    violations = check_rules(gdf)

    assert len(violations) == 1
    assert violations[0]["type"] == "warning"
    assert "uitgezonderd" in violations[0]["msg"].lower()


def test_original_bgt_marker_without_project_is_exempt():
    gdf = _gdf(
        [
            {
                "subthema": "bermverharding",
                "Besteknummer": "Oorspronkelijke BGT-data",
                "Onderhoudsproject": "",
            }
        ]
    )

    violations = check_rules(gdf)

    assert violations == []
