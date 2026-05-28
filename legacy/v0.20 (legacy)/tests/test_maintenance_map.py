
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from iasset_tool.maintenance_map import build_maintenance_control_map


def test_build_maintenance_control_map_maps_only_passport_objects_with_geometry():
    passport = gpd.GeoDataFrame(
        [
            {
                "nummer": "VV-N354-1",
                "Wegnummer": "N354",
                "subthema": "rijstrook",
                "Metrering": "1,0",
                "Situering": "Links",
                "Onderhoudsproject": "N354-HRB-01.0-02.0",
                "geometry": Polygon([(170000, 560000), (170010, 560000), (170010, 560010), (170000, 560010)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )
    details = pd.DataFrame(
        [
            {
                "objectnummer": "VV-N354-1",
                "objectnummer_norm": "VV-N354-1",
                "verschiltype": "ALLEEN_IN_PASPOORT",
                "opmerking": "Controleer object.",
            },
            {
                "objectnummer": "VV-N354-2",
                "objectnummer_norm": "VV-N354-2",
                "verschiltype": "ALLEEN_IN_ONDERHOUD",
                "opmerking": "Geen paspoortgeometrie.",
            },
        ]
    )

    result = build_maintenance_control_map(passport, details)

    assert result.folium_map is not None
    assert result.mapped_object_count == 1
    assert result.missing_passport_object_count == 1


def test_build_maintenance_control_map_returns_message_without_geometry():
    passport = pd.DataFrame([{"nummer": "VV-N354-1"}])
    details = pd.DataFrame([{"objectnummer_norm": "VV-N354-1"}])

    result = build_maintenance_control_map(passport, details)

    assert result.folium_map is None
    assert "Geen paspoortgeometrie" in result.message
