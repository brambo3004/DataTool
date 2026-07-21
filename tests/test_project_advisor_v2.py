from __future__ import annotations

import pandas as pd

from iasset_tool.project_advisor_v2 import (
    build_project_advisor_proposal_table,
    build_project_advisor_worklist,
    summarize_project_advisor,
)


def test_project_advisor_splitst_datakwaliteit_van_actielijst() -> None:
    """Een lokaal ontbrekend besteknummer mag geen werklijstregel worden."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v1",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.6-04.6",
                "vergelijking_iasset_status": "ok",
                "status_voorstel": "aandacht",
                "datakwaliteit_signalen": "enkele objecten missen Besteknummer",
                "lokale_afwijkingen": "",
                "fysiek_lengte_m": 3003.0,
                "naam_begin": 1.6,
                "naam_eind": 4.6,
            }
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)
    worklist = build_project_advisor_worklist(advisor_table)

    assert advisor_table.iloc[0]["adviesstatus"] == "ok"
    assert advisor_table.iloc[0]["datakwaliteitstatus"] == "aandacht"
    assert advisor_table.iloc[0]["grensstatus"] == "ok"
    assert advisor_table.iloc[0]["eindadvies"] == "Akkoord; datakwaliteit controleren"
    assert worklist.empty


def test_project_advisor_iasset_verschil_komt_in_werklijst() -> None:
    """Een voorstel dat afwijkt van iASSET blijft een echte databeheeractie."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v2",
                "onderhoudsproject_voorgesteld": "N398-HRB-00.8-01.2",
                "vergelijking_iasset_status": "aandacht",
                "status_voorstel": "aandacht",
                "fysiek_lengte_m": 348.0,
                "naam_begin": 0.8,
                "naam_eind": 1.2,
            }
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)
    worklist = build_project_advisor_worklist(advisor_table)

    assert advisor_table.iloc[0]["adviesstatus"] == "aandacht"
    assert advisor_table.iloc[0]["iassetvergelijking"] == "aandacht"
    assert advisor_table.iloc[0]["eindadvies"] == "Controleer verschil met iASSET"
    assert list(worklist["voorstel_id"]) == ["v2"]


def test_project_advisor_microvoorstel_wordt_apart_gemarkeerd() -> None:
    """Een nulmeter- of microvoorstel hoort niet als regulier project te voelen."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v3",
                "onderhoudsproject_voorgesteld": "N398-HRB-06.3-06.3",
                "vergelijking_iasset_status": "ok",
                "status_voorstel": "controleer",
                "fysiek_lengte_m": 0.0,
                "naam_begin": 6.3,
                "naam_eind": 6.3,
            }
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)
    worklist = build_project_advisor_worklist(advisor_table)

    assert advisor_table.iloc[0]["voorstelcategorie"] == "micro-/eindzonevoorstel"
    assert advisor_table.iloc[0]["eindadvies"] == "Niet als regulier onderhoudsproject gebruiken"
    assert list(worklist["voorstel_id"]) == ["v3"]


def test_project_advisor_terminale_grens_komt_in_werklijst() -> None:
    """De laatste projectgrens bij een afwijkend hm-interval blijft een actiepunt."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v_ok",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.6-04.6",
                "vergelijking_iasset_status": "ok",
                "status_voorstel": "aandacht",
                "fysiek_lengte_m": 3003.0,
                "naam_begin": 1.6,
                "naam_eind": 4.6,
                "begin_hm_interval_afwijking_m": 28.86,
            },
            {
                "voorstel_id": "v_end",
                "onderhoudsproject_voorgesteld": "N398-HRB-04.9-06.3",
                "vergelijking_iasset_status": "ok",
                "status_voorstel": "controleer",
                "fysiek_lengte_m": 1357.0,
                "naam_begin": 4.9,
                "naam_eind": 6.3,
                "eind_hm_interval_afwijking_m": -26.96,
                "eind_grensdiagnose": "hectometerinterval 06.2-06.3 is fysiek 73.0 m in plaats van 100.0 m",
            },
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)
    worklist = build_project_advisor_worklist(advisor_table)

    assert advisor_table.loc[advisor_table["voorstel_id"] == "v_ok", "in_werklijst"].iloc[0] == False
    assert advisor_table.loc[advisor_table["voorstel_id"] == "v_end", "eindadvies"].iloc[0] == "Controleer eindgrens op kaart"
    assert list(worklist["voorstel_id"]) == ["v_end"]


def test_project_advisor_samenvatting_en_werklijst() -> None:
    """De samenvatting telt voorstellen, werklijstregels en hm-intervallen."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "ok",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.0-01.2",
                "vergelijking_iasset_status": "ok",
                "status_voorstel": "ok",
                "fysiek_lengte_m": 200.0,
                "naam_begin": 1.0,
                "naam_eind": 1.2,
            },
            {
                "voorstel_id": "controle",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.2-01.4",
                "vergelijking_iasset_status": "aandacht",
                "status_voorstel": "aandacht",
                "fysiek_lengte_m": 200.0,
                "naam_begin": 1.2,
                "naam_eind": 1.4,
            },
        ]
    )
    intervals = pd.DataFrame(
        [
            {"status": "ok"},
            {"status": "aandacht"},
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)
    summary = summarize_project_advisor(advisor_table, intervals)
    worklist = build_project_advisor_worklist(advisor_table)

    assert summary["voorstellen"] == 2
    assert summary["geen_directe_actie"] == 1
    assert summary["werklijstregels"] == 1
    assert summary["hm_intervallen_afwijkend"] == 1
    assert list(worklist["voorstel_id"]) == ["controle"]
