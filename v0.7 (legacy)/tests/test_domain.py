import pandas as pd

from iasset_tool.domain import (
    find_exemption_marker,
    is_maintenance_project_exempt,
    is_project_value_empty,
    subthema_is_maintenance_project_exempt,
)


def test_documented_subthema_exception_carpoolplaats():
    assert subthema_is_maintenance_project_exempt(" Carpoolplaats ")


def test_legacy_geleideconstructie_is_not_documented_exception():
    # Geleideconstructie stond in de oude app, maar niet in het werkprocesdocument.
    assert not subthema_is_maintenance_project_exempt("geleideconstructie")


def test_original_bgt_marker_is_found_outside_subthema():
    row = {
        "subthema": "bermverharding",
        "subthema_clean": "bermverharding",
        "Besteknummer": "Oorspronkelijke BGT-data",
        "Onderhoudsproject": "",
    }
    assert find_exemption_marker(row) == "oorspronkelijke bgt-data"
    assert is_maintenance_project_exempt(row)


def test_project_value_empty_handles_nan_text_and_none():
    assert is_project_value_empty(None)
    assert is_project_value_empty(pd.NA)
    assert is_project_value_empty("nan")
    assert is_project_value_empty("   ")
    assert not is_project_value_empty("N398-HRB-01.2-03.4")


def test_exemption_handles_pd_na_subthema_clean():
    row = {
        "subthema": "perron",
        "subthema_clean": pd.NA,
        "Onderhoudsproject": "",
    }
    assert is_maintenance_project_exempt(row)
