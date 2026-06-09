
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


def test_maintenance_control_map_adds_control_context_legend_and_object_layers():
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
            },
            {
                "nummer": "VV-N354-2",
                "Wegnummer": "N354",
                "subthema": "bermverharding",
                "Metrering": "1,1",
                "Situering": "Links",
                "Onderhoudsproject": "N354-HRB-01.0-02.0",
                "geometry": Polygon([(170020, 560000), (170030, 560000), (170030, 560010), (170020, 560010)]),
            },
            {
                "nummer": "VV-N354-3",
                "Wegnummer": "N354",
                "subthema": "perron",
                "Metrering": "1,2",
                "Situering": "Links",
                "Onderhoudsproject": "N354-HRB-01.0-02.0",
                "geometry": Polygon([(170040, 560000), (170050, 560000), (170050, 560010), (170040, 560010)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:28992",
    )
    details = pd.DataFrame(
        [
            {"objectnummer": "VV-N354-1", "objectnummer_norm": "VV-N354-1", "verschiltype": "ALLEEN_IN_PASPOORT"},
            {"objectnummer": "VV-N354-2", "objectnummer_norm": "VV-N354-2", "verschiltype": "OBJECTSET_VERSCHIL"},
            {"objectnummer": "VV-N354-3", "objectnummer_norm": "VV-N354-3", "verschiltype": "ALLEEN_IN_PASPOORT"},
        ]
    )
    action_row = {
        "onderhoudsproject": "N354-HRB-01.0-02.0",
        "mogelijke_vervangende_projectnaam": "N354-HRBR-01.0-02.0",
        "prioriteit": "hoog",
        "duiding": "oude_projectnaam_of_migratie",
        "voortgang_status": "blijft_open",
    }

    result = build_maintenance_control_map(passport, details, action_row)

    assert result.folium_map is not None
    assert result.primary_object_count == 1
    assert result.secondary_object_count == 1
    assert result.exempt_object_count == 1
    assert result.difference_type_counts["ALLEEN_IN_PASPOORT"] == 2

    html = result.folium_map.get_root().render()
    assert "Legenda Onderhoudscontrole" in html
    assert "N354-HRBR-01.0-02.0" in html
    assert "oude_projectnaam_of_migratie" in html
