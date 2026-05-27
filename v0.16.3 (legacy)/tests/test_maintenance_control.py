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
    assert statuses["N398-HRB-01.0-02.0"] == "OK_VOLLEDIG"
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


def test_build_maintenance_control_detects_object_difference_and_wrong_road_object():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N398-26704",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "Metrering": "0,8",
            },
            {
                "nummer": "VV-N398-26707",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "Metrering": "1,0",
            },
        ]
    )

    maintenance_df = pd.DataFrame(
        [
            {
                "objectnummer": "VV-N398-26704",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "maatregel": "OAB",
            },
            {
                "objectnummer": "VV-N398-26707",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "maatregel": "OAB",
            },
            {
                "objectnummer": "VV-N389-00000008",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "maatregel": "OAB",
            },
        ]
    )

    result = build_maintenance_control(passport_df, maintenance_df, selected_road="N398")

    row = result.comparison.iloc[0]
    assert row["status"] == "OBJECT_WEGNUMMER_VERDACHT"
    assert row["alleen_in_onderhoud"] == 1
    assert row["onderhoud_object_wegnummer_verdacht"] == 1

    diff_types = set(result.object_differences["verschiltype"])
    assert "ALLEEN_IN_ONDERHOUD" in diff_types
    assert "OBJECT_WEGNUMMER_VERDACHT" in diff_types


def test_invalid_metrering_is_ignored_for_hm_range_but_reported():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N398-38213",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "Metrering": "4,9",
            },
            {
                "nummer": "VV-N398-38214",
                "Wegnummer": "N398",
                "subthema": "inrit en doorsteek",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "Metrering": "4,,9",
            },
        ]
    )

    maintenance_df = pd.DataFrame(
        [
            {
                "objectnummer": "VV-N398-38213",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "maatregel": "DGD",
            },
            {
                "objectnummer": "VV-N398-38214",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "maatregel": "DGD",
            },
        ]
    )

    result = build_maintenance_control(passport_df, maintenance_df, selected_road="N398")

    summary_row = result.passport_projects.iloc[0]
    assert summary_row["paspoort_hm_min"] == 4.9
    assert summary_row["paspoort_hm_max"] == 4.9
    assert summary_row["paspoort_ongeldige_metrering_aantal"] == 1

    comparison_row = result.comparison.iloc[0]
    assert comparison_row["status"] == "HM_BEREIK_VERDACHT"

    diff_types = set(result.object_differences["verschiltype"])
    assert "ONGELDIGE_METRERING_PASPOORT" in diff_types



def test_action_list_translates_missing_project_to_work_instruction():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N354-100",
                "Wegnummer": "N354",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N354-HRB-11.5-12.8",
                "Metrering": "11,5",
            }
        ]
    )

    result = build_maintenance_control(passport_df, pd.DataFrame(), selected_road="N354")

    assert result.summary["acties"] == 1
    action = result.action_list.iloc[0]
    assert action["status"] == "ONTBREEKT_IN_ONDERHOUD"
    assert action["controlecategorie"] == "Project ontbreekt in onderhoudsexport"
    assert "VV-N354-100" in action["betrokken_objecten"]
    assert "Zoek het onderhoudsproject exact op" in action["voorgestelde_actie"]


def test_action_list_explains_wrong_road_object_and_invalid_hm():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N398-26704",
                "Wegnummer": "N398",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "Metrering": "0,8",
            },
            {
                "nummer": "VV-N398-38214",
                "Wegnummer": "N398",
                "subthema": "inrit en doorsteek",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "Metrering": "4,,9",
            },
        ]
    )

    maintenance_df = pd.DataFrame(
        [
            {
                "objectnummer": "VV-N398-26704",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "maatregel": "OAB",
            },
            {
                "objectnummer": "VV-N389-00000008",
                "Onderhoudsproject": "N398-HRB-00.8-01.1",
                "maatregel": "OAB",
            },
            {
                "objectnummer": "VV-N398-38214",
                "Onderhoudsproject": "N398-HRB-04.8-04.9",
                "maatregel": "DGD",
            },
        ]
    )

    result = build_maintenance_control(passport_df, maintenance_df, selected_road="N398")

    assert set(result.action_list["status"]) == {"OBJECT_WEGNUMMER_VERDACHT", "HM_BEREIK_VERDACHT"}

    wrong_road_action = result.action_list[result.action_list["status"] == "OBJECT_WEGNUMMER_VERDACHT"].iloc[0]
    assert "VV-N389-00000008" in wrong_road_action["betrokken_objecten"]
    assert "andere N-weg" in wrong_road_action["controlecategorie"]

    hm_action = result.action_list[result.action_list["status"] == "HM_BEREIK_VERDACHT"].iloc[0]
    assert "VV-N398-38214" in hm_action["betrokken_objecten"]
    assert "corrigeer de metrering" in hm_action["voorgestelde_actie"]


def test_action_list_contains_practical_category_and_follow_up_columns():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N354-100",
                "Wegnummer": "N354",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N354-HRB-11.5-12.8",
                "Metrering": "11,5",
            }
        ]
    )

    result = build_maintenance_control(passport_df, pd.DataFrame(), selected_road="N354")
    action = result.action_list.iloc[0]

    assert action["praktische_categorie"] == "oude_of_ontbrekende_projectnaam_controleren"
    assert action["beoordeling_databeheerder"] == ""
    assert action["afhandelstatus"] == "nieuw"
    assert action["actiehouder"] == ""
    assert action["opmerking_afhandeling"] == ""


def test_action_list_uses_practical_category_for_wrong_road_and_object_difference():
    passport_df = pd.DataFrame(
        [
            {
                "nummer": "VV-N354-22551",
                "Wegnummer": "N354",
                "subthema": "rijstrook",
                "Onderhoudsproject": "N354-HRB-31.6-31.7",
                "Metrering": "31,6",
            },
            {
                "nummer": "VV-N354-300",
                "Wegnummer": "N354",
                "subthema": "fietspad",
                "Onderhoudsproject": "N354-FPR-25.7-28.0",
                "Metrering": "25,8",
            },
        ]
    )

    maintenance_df = pd.DataFrame(
        [
            {
                "objectnummer": "VV-N354-22551",
                "Onderhoudsproject": "N354-HRB-31.6-31.7",
                "maatregel": "DGD",
            },
            {
                "objectnummer": "VV-N354-22552",
                "Onderhoudsproject": "N354-HRB-31.6-31.7",
                "maatregel": "DGD",
            },
            {
                "objectnummer": "VV-N354-300",
                "Onderhoudsproject": "N354-FPR-25.7-28.0",
                "maatregel": "DGD",
            },
            {
                "objectnummer": "VV-N392-42740",
                "Onderhoudsproject": "N354-FPR-25.7-28.0",
                "maatregel": "DGD",
            },
        ]
    )

    result = build_maintenance_control(passport_df, maintenance_df, selected_road="N354")
    categories_by_status = dict(zip(result.action_list["status"], result.action_list["praktische_categorie"]))

    assert categories_by_status["OBJECT_WEGNUMMER_VERDACHT"] == "wegnummer_objectpaspoort_of_grensgeval_controleren"
    assert categories_by_status["OBJECTVERSCHIL"] == "objectset_of_projecttype_controleren"
