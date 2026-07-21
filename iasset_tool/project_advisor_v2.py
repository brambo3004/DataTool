"""
Presentatielaag voor Project Adviseur 2.0.

Deze module bevat bewust geen nieuwe GIS-rekenlogica. De functies ordenen de
bestaande v0.35.5-projectasuitkomsten tot een rustiger scherm voor
databeheer: gescheiden statussen, een compacte samenvatting en een werklijst.

Waarom apart van app.py?
- Streamlit blijft UI.
- De statuspresentatie is testbaar zonder browser.
- De rekenmotor in ``project_axis.py`` blijft ongemoeid.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .utils import clean_display_value


STATUS_ORDER: dict[str, int] = {
    "ok": 0,
    "akkoord": 0,
    "overzicht": 0,
    "projectie": 0,
    "aandacht": 1,
    "controleer": 2,
    "niet gebruiken zonder handmatige beoordeling": 3,
    "onbekend": 1,
}


PROJECT_ADVISOR_MAIN_COLUMNS: tuple[str, ...] = (
    "eindadvies",
    "adviesstatus",
    "datakwaliteitstatus",
    "grensstatus",
    "referentieasstatus",
    "onderhoudsproject_voorgesteld",
    "project_type",
    "fysiek_begin_km",
    "fysiek_eind_km",
    "naam_begin",
    "naam_eind",
    "aantal_primaire_objecten",
    "hoofdmelding",
    "contextmelding",
)


PROJECT_ADVISOR_WORKLIST_COLUMNS: tuple[str, ...] = (
    "eindadvies",
    "adviesstatus",
    "datakwaliteitstatus",
    "grensstatus",
    "onderhoudsproject_voorgesteld",
    "project_type",
    "fysiek_begin_km",
    "fysiek_eind_km",
    "hoofdmelding",
    "contextmelding",
    "datakwaliteit_signalen",
    "lokale_afwijkingen",
    "begin_grensdiagnose",
    "eind_grensdiagnose",
    "voorstel_id",
)


def _text(value: Any) -> str:
    """Geef een veilige tekstrepresentatie zonder 'nan' of None."""
    text = clean_display_value(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kolom terug."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _status_value(value: Any, default: str = "ok") -> str:
    """Normaliseer een statuswaarde voor sortering en tellingen."""
    text = _text(value).lower()
    return text if text else default


def _status_rank(value: Any) -> int:
    """Bepaal de zwaarte van een status."""
    return STATUS_ORDER.get(_status_value(value, "onbekend"), 1)


def _has_text_signal(row: pd.Series, columns: Iterable[str]) -> bool:
    """Controleer of één van de genoemde kolommen een inhoudelijk signaal bevat."""
    for column in columns:
        if column in row.index and _text(row.get(column)):
            return True
    return False


def _numeric_abs_ge(row: pd.Series, columns: Iterable[str], threshold: float) -> bool:
    """Controleer of één van de numerieke kolommen boven een absolute drempel zit."""
    for column in columns:
        if column not in row.index:
            continue
        try:
            value = float(row.get(column))
        except (TypeError, ValueError):
            continue
        if value == value and abs(value) >= threshold:
            return True
    return False


def _contains_any(row: pd.Series, columns: Iterable[str], words: Iterable[str]) -> bool:
    """Zoek sleutelwoorden in bestaande tekstkolommen."""
    haystack = " ".join(_text(row.get(column)).lower() for column in columns if column in row.index)
    return any(word in haystack for word in words)


def derive_datakwaliteitstatus(row: pd.Series) -> str:
    """
    Bepaal de aparte datakwaliteitstatus.

    Deze status mag het inhoudelijke projectadvies niet automatisch geel maken.
    Een voorstel kan dus adviesstatus ``ok`` hebben en tegelijk datakwaliteit
    ``aandacht``.
    """
    if _has_text_signal(row, ("datakwaliteit_signalen",)):
        return "aandacht"

    lokale_afwijkingen = _text(row.get("lokale_afwijkingen", "")).lower()
    if "datakwaliteit" in lokale_afwijkingen or "ontbrekend" in lokale_afwijkingen or "mist" in lokale_afwijkingen:
        return "aandacht"

    return "ok"


def derive_grensstatus(row: pd.Series) -> str:
    """Bepaal of begin- of eindgrens extra controle vraagt."""
    diagnostic_columns = (
        "grensdiagnose",
        "begin_grensdiagnose",
        "eind_grensdiagnose",
        "hoofdmelding",
        "contextmelding",
    )
    if _contains_any(
        row,
        diagnostic_columns,
        (
            "buiten ijkbereik",
            "geen bruikbare ijking",
            "geen bruikbare wegas",
            "einde referentieas",
            "0-meter",
            "nulmeter",
        ),
    ):
        return "controleer"

    # De drempels volgen de bestaande v0.35.5-presentatie: afwijkingen rond
    # 15 meter en groter zijn geen automatische fout, maar horen zichtbaar in
    # de grensbeoordeling.
    if _numeric_abs_ge(
        row,
        ("begin_hm_interval_afwijking_m", "eind_hm_interval_afwijking_m"),
        threshold=15.0,
    ):
        return "aandacht"

    if _has_text_signal(row, ("grensdiagnose", "begin_grensdiagnose", "eind_grensdiagnose")):
        return "aandacht"

    return "ok"


def derive_referentieasstatus(row: pd.Series) -> str:
    """Bepaal of de gebruikte as/ijking als presentatie extra aandacht vraagt."""
    diagnostic_columns = (
        "grensdiagnose",
        "begin_grensdiagnose",
        "eind_grensdiagnose",
        "hoofdmelding",
        "contextmelding",
    )
    if _contains_any(
        row,
        diagnostic_columns,
        (
            "geen bruikbare wegas",
            "geen bruikbare ijking",
            "buiten ijkbereik",
            "einde referentieas",
        ),
    ):
        return "controleer"

    if _has_text_signal(row, ("begin_hm_interval", "eind_hm_interval")) and _numeric_abs_ge(
        row,
        ("begin_hm_interval_afwijking_m", "eind_hm_interval_afwijking_m"),
        threshold=15.0,
    ):
        return "aandacht"

    return "ok"


def combine_eindadvies(
    adviesstatus: str,
    datakwaliteitstatus: str,
    grensstatus: str,
    referentieasstatus: str,
) -> str:
    """
    Maak een leesbaar eindadvies op basis van gescheiden statussen.

    Het eindadvies is bewust mensgericht. Het is geen iASSET-mutatieadvies en
    voert niets automatisch door.
    """
    if _status_rank(referentieasstatus) >= 2:
        return "Niet gebruiken zonder handmatige beoordeling"
    if _status_rank(adviesstatus) >= 2:
        return "Controleren voor gebruik"
    if _status_rank(grensstatus) >= 2:
        return "Controleer grens/ijking"
    if _status_rank(adviesstatus) == 1 or _status_rank(grensstatus) == 1:
        return "Aandachtspunt controleren"
    if _status_rank(datakwaliteitstatus) >= 1:
        return "Inhoudelijk bruikbaar; datakwaliteit controleren"
    return "Akkoord als kaartbeeld klopt"


def build_project_advisor_proposal_table(project_proposals: pd.DataFrame | None) -> pd.DataFrame:
    """
    Voeg presentatiestatussen toe aan de bestaande projectvoorstellen.

    De invoer wordt niet gemuteerd. Ontbrekende kolommen leveren veilige lege
    waarden op, zodat oude of incomplete diagnoseframes de UI niet laten crashen.
    """
    if project_proposals is None or project_proposals.empty:
        return pd.DataFrame(columns=[*PROJECT_ADVISOR_MAIN_COLUMNS, "voorstel_id"])

    result = project_proposals.copy()

    if "adviesstatus" not in result.columns:
        status_source = _first_existing_column(result, ("status_voorstel", "status", "adviesstatus"))
        result["adviesstatus"] = (
            result[status_source].map(lambda value: _status_value(value, "ok")) if status_source else "ok"
        )

    result["datakwaliteitstatus"] = result.apply(derive_datakwaliteitstatus, axis=1)
    result["grensstatus"] = result.apply(derive_grensstatus, axis=1)
    result["referentieasstatus"] = result.apply(derive_referentieasstatus, axis=1)
    result["eindadvies"] = result.apply(
        lambda row: combine_eindadvies(
            row.get("adviesstatus", "ok"),
            row.get("datakwaliteitstatus", "ok"),
            row.get("grensstatus", "ok"),
            row.get("referentieasstatus", "ok"),
        ),
        axis=1,
    )

    return result


def summarize_project_advisor(
    advisor_table: pd.DataFrame | None,
    hectometer_intervals: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Maak compacte tellers voor het hoofdscherm."""
    table = advisor_table if isinstance(advisor_table, pd.DataFrame) else pd.DataFrame()
    intervals = hectometer_intervals if isinstance(hectometer_intervals, pd.DataFrame) else pd.DataFrame()
    comparison_df = comparison if isinstance(comparison, pd.DataFrame) else pd.DataFrame()

    def count_status(column: str, status: str) -> int:
        if table.empty or column not in table.columns:
            return 0
        return int((table[column].fillna("").astype(str).str.lower() == status).sum())

    interval_status = (
        intervals["status"].fillna("ok").astype(str).str.lower()
        if not intervals.empty and "status" in intervals.columns
        else pd.Series(dtype="object")
    )

    comparison_status = (
        comparison_df["status"].fillna("").astype(str).str.lower()
        if not comparison_df.empty and "status" in comparison_df.columns
        else pd.Series(dtype="object")
    )

    return {
        "voorstellen": int(len(table)),
        "advies_ok": count_status("adviesstatus", "ok"),
        "advies_aandacht": count_status("adviesstatus", "aandacht"),
        "advies_controleer": count_status("adviesstatus", "controleer"),
        "datakwaliteit_aandacht": count_status("datakwaliteitstatus", "aandacht"),
        "grens_aandacht": count_status("grensstatus", "aandacht"),
        "grens_controleer": count_status("grensstatus", "controleer"),
        "referentieas_controleer": count_status("referentieasstatus", "controleer"),
        "hm_intervallen_afwijkend": int((interval_status != "ok").sum()) if not interval_status.empty else 0,
        "iasset_vergelijking_aandacht": int((comparison_status == "aandacht").sum()) if not comparison_status.empty else 0,
        "iasset_vergelijking_controleer": int((comparison_status == "controleer").sum()) if not comparison_status.empty else 0,
    }


def build_project_advisor_worklist(advisor_table: pd.DataFrame | None) -> pd.DataFrame:
    """Selecteer alleen voorstellen die actie of controle vragen."""
    if advisor_table is None or advisor_table.empty:
        return pd.DataFrame(columns=PROJECT_ADVISOR_WORKLIST_COLUMNS)

    table = advisor_table.copy()
    status_columns = [column for column in ("adviesstatus", "datakwaliteitstatus", "grensstatus", "referentieasstatus") if column in table.columns]
    if not status_columns:
        return pd.DataFrame(columns=[column for column in PROJECT_ADVISOR_WORKLIST_COLUMNS if column in table.columns])

    mask = pd.Series(False, index=table.index)
    for column in status_columns:
        mask = mask | (table[column].fillna("ok").astype(str).str.lower() != "ok")

    worklist = table[mask].copy()
    columns = [column for column in PROJECT_ADVISOR_WORKLIST_COLUMNS if column in worklist.columns]
    return worklist[columns] if columns else worklist
