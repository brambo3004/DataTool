import io

import pandas as pd

from iasset_tool.maintenance_control import (
    action_work_queue_summary,
    build_maintenance_control,
    filter_action_work_queue,
    merge_action_work_queue_edits,
    merge_previous_action_follow_up,
    normalize_project_name,
    read_action_lists_safely,
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


def test_previous_action_list_follow_up_is_copied_to_matching_action():
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

    previous_action_list = pd.DataFrame(
        [
            {
                "onderhoudsproject": "N354-HRB-11.5-12.8",
                "project_norm": "N354-HRB-11.5-12.8",
                "status": "ONTBREEKT_IN_ONDERHOUD",
                "controlecategorie": "Project ontbreekt in onderhoudsexport",
                "praktische_categorie": "oude_of_ontbrekende_projectnaam_controleren",
                "beoordeling_databeheerder": "Oude projectnaam staat nog bij objecten.",
                "afhandelstatus": "te corrigeren in paspoort",
                "actiehouder": "Bram",
                "opmerking_afhandeling": "Na correctie opnieuw exporteren.",
            }
        ]
    )

    result = build_maintenance_control(
        passport_df,
        pd.DataFrame(),
        selected_road="N354",
        previous_action_list=previous_action_list,
    )

    action = result.action_list.iloc[0]
    assert action["beoordeling_databeheerder"] == "Oude projectnaam staat nog bij objecten."
    assert action["afhandelstatus"] == "te corrigeren in paspoort"
    assert action["actiehouder"] == "Bram"
    assert action["opmerking_afhandeling"] == "Na correctie opnieuw exporteren."
    assert result.summary["acties_met_overgenomen_beoordeling"] == 1
    assert any("eerdere beoordeling" in warning for warning in result.warnings)


def test_previous_action_list_does_not_copy_to_changed_status():
    action_list = pd.DataFrame(
        [
            {
                "onderhoudsproject": "N354-HRB-11.5-12.8",
                "project_norm": "N354-HRB-11.5-12.8",
                "status": "OBJECTVERSCHIL",
                "controlecategorie": "Objectsets verschillen",
                "praktische_categorie": "objectset_of_projecttype_controleren",
                "beoordeling_databeheerder": "",
                "afhandelstatus": "nieuw",
                "actiehouder": "",
                "opmerking_afhandeling": "",
            }
        ]
    )
    previous_action_list = pd.DataFrame(
        [
            {
                "onderhoudsproject": "N354-HRB-11.5-12.8",
                "project_norm": "N354-HRB-11.5-12.8",
                "status": "ONTBREEKT_IN_ONDERHOUD",
                "controlecategorie": "Project ontbreekt in onderhoudsexport",
                "praktische_categorie": "oude_of_ontbrekende_projectnaam_controleren",
                "beoordeling_databeheerder": "Deze oude melding mag niet meekomen.",
                "afhandelstatus": "afgehandeld",
            }
        ]
    )

    merged, copied = merge_previous_action_follow_up(action_list, previous_action_list)

    assert copied == 0
    assert merged.iloc[0]["afhandelstatus"] == "nieuw"
    assert merged.iloc[0]["beoordeling_databeheerder"] == ""


def test_read_action_lists_safely_reads_semicolon_csv():
    csv_text = (
        "onderhoudsproject;status;controlecategorie;praktische_categorie;"
        "beoordeling_databeheerder;afhandelstatus;actiehouder;opmerking_afhandeling\n"
        "N354-HRB-11.5-12.8;ONTBREEKT_IN_ONDERHOUD;Project ontbreekt in onderhoudsexport;"
        "oude_of_ontbrekende_projectnaam_controleren;Oude projectnaam;in onderzoek;Bram;Controle loopt\n"
    )

    df, warnings = read_action_lists_safely((("Fase4_Actielijst_N354.csv", csv_text.encode("utf-8-sig")),))

    assert len(df) == 1
    assert df.iloc[0]["afhandelstatus"] == "in onderzoek"
    assert any("1 regel" in warning for warning in warnings)


def test_previous_action_list_accepts_human_column_names_and_v0162_matching():
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
    csv_text = (
        "onderhoudsproject;project_norm;status;controlecategorie;beoordeling databeheerder;afhandel status\n"
        "N354-HRB-11.5-12.8;N354-HRB-11.5-12.8;ONTBREEKT_IN_ONDERHOUD;"
        "Project ontbreekt in onderhoudsexport;Projectnaam is vervallen.;te corrigeren in paspoort\n"
    )
    previous_action_list, warnings = read_action_lists_safely((("oude_actielijst.csv", csv_text.encode("utf-8")),))

    assert "beoordeling_databeheerder" in previous_action_list.columns
    assert "afhandelstatus" in previous_action_list.columns
    assert any("mist opvolgkolommen" in warning for warning in warnings)

    result = build_maintenance_control(
        passport_df,
        pd.DataFrame(),
        selected_road="N354",
        previous_action_list=previous_action_list,
    )

    action = result.action_list.iloc[0]
    assert action["beoordeling_databeheerder"] == "Projectnaam is vervallen."
    assert action["afhandelstatus"] == "te corrigeren in paspoort"


def test_action_work_queue_summary_counts_follow_up_statuses():
    action_list = pd.DataFrame(
        [
            {"ernst": "waarschuwing", "afhandelstatus": "nieuw"},
            {"ernst": "waarschuwing", "afhandelstatus": "in onderzoek"},
            {"ernst": "aandachtspunt", "afhandelstatus": "te corrigeren in paspoort"},
            {"ernst": "waarschuwing", "afhandelstatus": "afgehandeld"},
        ]
    )

    summary = action_work_queue_summary(action_list)

    assert summary["controlepunten"] == 4
    assert summary["waarschuwingen"] == 3
    assert summary["aandachtspunten"] == 1
    assert summary["nieuw"] == 1
    assert summary["in_onderzoek"] == 1
    assert summary["te_corrigeren"] == 1
    assert summary["afgehandeld"] == 1
    assert summary["open"] == 3


def test_filter_action_work_queue_filters_by_category_status_and_search():
    action_list = pd.DataFrame(
        [
            {
                "onderhoudsproject": "N354-HRB-11.5-12.8",
                "ernst": "waarschuwing",
                "status": "ONTBREEKT_IN_ONDERHOUD",
                "praktische_categorie": "oude_of_ontbrekende_projectnaam_controleren",
                "afhandelstatus": "nieuw",
                "actiehouder": "Bram",
                "betrokken_objecten": "VV-N354-100",
                "uitleg": "Project ontbreekt in onderhoudsexport",
            },
            {
                "onderhoudsproject": "N354-FPR-25.7-28.0",
                "ernst": "waarschuwing",
                "status": "OBJECT_WEGNUMMER_VERDACHT",
                "praktische_categorie": "wegnummer_objectpaspoort_of_grensgeval_controleren",
                "afhandelstatus": "in onderzoek",
                "actiehouder": "Willem",
                "betrokken_objecten": "VV-N392-42740",
                "uitleg": "Objectnummer lijkt bij andere N-weg te horen",
            },
        ]
    )

    result = filter_action_work_queue(
        action_list,
        praktische_categorie="wegnummer_objectpaspoort_of_grensgeval_controleren",
        afhandelstatus="in onderzoek",
        zoektekst="N392",
    )

    assert len(result) == 1
    assert result.iloc[0]["onderhoudsproject"] == "N354-FPR-25.7-28.0"


def test_merge_action_work_queue_edits_updates_only_follow_up_columns():
    base = pd.DataFrame(
        [
            {
                "onderhoudsproject": "N354-HRB-11.5-12.8",
                "project_norm": "N354-HRB-11.5-12.8",
                "status": "ONTBREEKT_IN_ONDERHOUD",
                "controlecategorie": "Project ontbreekt in onderhoudsexport",
                "praktische_categorie": "oude_of_ontbrekende_projectnaam_controleren",
                "ernst": "waarschuwing",
                "afhandelstatus": "nieuw",
                "beoordeling_databeheerder": "",
                "actiehouder": "",
                "opmerking_afhandeling": "",
            },
            {
                "onderhoudsproject": "N354-FPR-25.7-28.0",
                "project_norm": "N354-FPR-25.7-28.0",
                "status": "OBJECT_WEGNUMMER_VERDACHT",
                "controlecategorie": "Objectnummer lijkt bij andere N-weg te horen",
                "praktische_categorie": "wegnummer_objectpaspoort_of_grensgeval_controleren",
                "ernst": "waarschuwing",
                "afhandelstatus": "nieuw",
                "beoordeling_databeheerder": "",
                "actiehouder": "",
                "opmerking_afhandeling": "",
            },
        ]
    )

    edited_subset = base.iloc[[1]].copy()
    edited_subset.loc[edited_subset.index[0], "afhandelstatus"] = "verklaarbare uitzondering"
    edited_subset.loc[edited_subset.index[0], "beoordeling_databeheerder"] = "Grensgeval bij wegnummerovergang"
    edited_subset.loc[edited_subset.index[0], "actiehouder"] = "Bram"

    result = merge_action_work_queue_edits(base, edited_subset)

    unchanged = result[result["onderhoudsproject"] == "N354-HRB-11.5-12.8"].iloc[0]
    changed = result[result["onderhoudsproject"] == "N354-FPR-25.7-28.0"].iloc[0]

    assert unchanged["afhandelstatus"] == "nieuw"
    assert changed["afhandelstatus"] == "verklaarbare uitzondering"
    assert changed["beoordeling_databeheerder"] == "Grensgeval bij wegnummerovergang"
    assert changed["actiehouder"] == "Bram"
    assert changed["status"] == "OBJECT_WEGNUMMER_VERDACHT"
