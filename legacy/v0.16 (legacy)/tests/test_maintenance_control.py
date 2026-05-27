import io

import pandas as pd

from iasset_tool.maintenance_control import (
    build_maintenance_control,
    normalize_project_name,
    read_maintenance_exports,
    summarize_maintenance_projects,
)


def test_normalize_project_name_keeps_content_but_fixes_spacing_and_dash():
    assert normalize_project_name(" N398 – HRB - 01.6 - 04.6 ") == "N398-HRB-01.6-04.6"


def test_read_maintenance_excel_detects_header_row_and_project_column():
    buffer = io.BytesIO()

    raw = pd.DataFrame(
        [
            ["", "Onderhoud, overzicht", ""],
            ["", "", ""],
            ["Object nr:", "Project", "Maatregel Omschrijving"],
            ["VV-1", "N398-HRB-01.6-04.6", "2L DGD A"],
        ]
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, header=False, sheet_name="Worksheet")

    result = read_maintenance_exports((("onderhoud.xlsx", buffer.getvalue()),))

    assert result.dataframe.shape[0] == 1
    assert result.dataframe.iloc[0]["Onderhoudsproject"] == "N398-HRB-01.6-04.6"
    assert result.dataframe.iloc[0]["maatregel"] == "2L DGD A"
    assert any("kopregel gevonden op rij 3" in warning for warning in result.warnings)



def test_read_maintenance_csv_detects_header_after_title_rows():
    csv_text = (
        "Onderhoud, overzicht\n"
        "\n"
        "Object nr:;Project;Maatregel Omschrijving\n"
        "VV-1;N398-HRB-01.0-02.0;DGD\n"
    )

    result = read_maintenance_exports((("onderhoud.csv", csv_text.encode("utf-8")),))

    assert result.dataframe.shape[0] == 1
    assert result.dataframe.iloc[0]["objectnummer"] == "VV-1"
    assert result.dataframe.iloc[0]["Onderhoudsproject"] == "N398-HRB-01.0-02.0"


def test_build_maintenance_control_finds_ok_missing_and_orphan_projects():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-1",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-01.0-02.0",
                "Metrering": "1,1",
            },
            {
                "nummer": "VV-2",
                "Wegnummer": "N398",
                "subthema": "bermverharding",
                "Onderhoudsproject": "N398-HRB-02.0-03.0",
                "Metrering": "2,1",
            },
        ]
    )

    maintenance_df = pd.DataFrame(
        [
            {
                "objectnummer": "VV-1",
                "Onderhoudsproject": "N398-HRB-01.0-02.0",
                "maatregel": "2L DGD A",
            },
            {
                "objectnummer": "VV-999",
                "Onderhoudsproject": "N398-HRB-03.0-04.0",
                "maatregel": "LVOv",
            },
        ]
    )

    result = build_maintenance_control(passport_df, maintenance_df, selected_road="N398")

    statuses = dict(zip(result.comparison["onderhoudsproject"], result.comparison["status"]))
    assert statuses["N398-HRB-01.0-02.0"] == "OK"
    assert statuses["N398-HRB-02.0-03.0"] == "ONTBREEKT_IN_ONDERHOUD"
    assert statuses["N398-HRB-03.0-04.0"] == "GEEN_PASPOORTOBJECTEN"
    assert result.summary["projecten_ok"] == 1
    assert result.summary["ontbreekt_in_onderhoud"] == 1
    assert result.summary["geen_paspoortobjecten"] == 1


def test_passport_summary_counts_primary_secondary_and_exempt_objects():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-1",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-01.0-02.0",
            },
            {
                "nummer": "VV-2",
                "Wegnummer": "N398",
                "subthema": "bermverharding",
                "Onderhoudsproject": "N398-HRB-01.0-02.0",
            },
            {
                "nummer": "VV-3",
                "Wegnummer": "N398",
                "subthema": "perron",
                "Onderhoudsproject": "N398-HRB-01.0-02.0",
            },
        ]
    )

    result = build_maintenance_control(
        passport_df,
        pd.DataFrame([{"Onderhoudsproject": "N398-HRB-01.0-02.0"}]),
        selected_road="N398",
    )

    row = result.passport_projects.iloc[0]
    assert row["paspoort_primaire_objecten"] == 1
    assert row["paspoort_secundaire_objecten"] == 2
    assert row["paspoort_uitzondering_objecten"] == 1


def test_maintenance_summary_filters_by_selected_road_from_project_name():
    df = pd.DataFrame(
        [
            {"Onderhoudsproject": "N398-HRB-01.0-02.0", "objectnummer": "VV-1"},
            {"Onderhoudsproject": "N354-HRB-01.0-02.0", "objectnummer": "VV-2"},
        ]
    )

    summary = summarize_maintenance_projects(df, selected_road="N398")

    assert len(summary) == 1
    assert summary.iloc[0]["onderhoudsproject"] == "N398-HRB-01.0-02.0"
