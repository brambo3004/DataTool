from __future__ import annotations

import pandas as pd

from iasset_tool.project_advisor_v2 import (
    build_project_advisor_proposal_table,
    build_project_advisor_worklist,
    summarize_project_advisor,
)


def test_project_advisor_splitst_datakwaliteit_van_adviesstatus() -> None:
    """Een lokaal ontbrekend besteknummer mag het inhoudelijke advies niet vervuilen."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v1",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.6-04.6",
                "status_voorstel": "ok",
                "datakwaliteit_signalen": "enkele objecten missen Besteknummer",
                "lokale_afwijkingen": "",
            }
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)

    assert advisor_table.iloc[0]["adviesstatus"] == "ok"
    assert advisor_table.iloc[0]["datakwaliteitstatus"] == "aandacht"
    assert advisor_table.iloc[0]["grensstatus"] == "ok"
    assert advisor_table.iloc[0]["eindadvies"] == "Inhoudelijk bruikbaar; datakwaliteit controleren"


def test_project_advisor_grensstatus_reageert_op_afwijkend_hm_interval() -> None:
    """Afwijkende hectometerintervallen horen in grensstatus, niet in datakwaliteit."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "v2",
                "onderhoudsproject_voorgesteld": "N398-HRB-06.2-06.3",
                "status_voorstel": "ok",
                "eind_hm_interval": "06.2-06.3",
                "eind_hm_interval_afwijking_m": -26.96,
                "eind_grensdiagnose": "eindgrens ligt in hectometerinterval 06.2-06.3",
            }
        ]
    )

    advisor_table = build_project_advisor_proposal_table(proposals)

    assert advisor_table.iloc[0]["adviesstatus"] == "ok"
    assert advisor_table.iloc[0]["datakwaliteitstatus"] == "ok"
    assert advisor_table.iloc[0]["grensstatus"] == "aandacht"
    assert advisor_table.iloc[0]["eindadvies"] == "Aandachtspunt controleren"


def test_project_advisor_samenvatting_en_werklijst() -> None:
    """De werklijst bevat alleen voorstellen met aandacht of controle."""
    proposals = pd.DataFrame(
        [
            {
                "voorstel_id": "ok",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.0-01.2",
                "status_voorstel": "ok",
            },
            {
                "voorstel_id": "controle",
                "onderhoudsproject_voorgesteld": "N398-HRB-01.2-01.4",
                "status_voorstel": "controleer",
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
    assert summary["advies_ok"] == 1
    assert summary["advies_controleer"] == 1
    assert summary["hm_intervallen_afwijkend"] == 1
    assert list(worklist["voorstel_id"]) == ["controle"]
