"""
Presentatielaag voor Project Adviseur 2.0.

Deze module bevat bewust geen nieuwe GIS-rekenlogica. De functies vertalen de
bestaande projectas-uitkomsten naar een werkbaar databeheerbeeld:
een voorstellenlijst, een actielijst en begrijpelijke samenvattingen.

Waarom deze laag apart houden?
- ``project_axis.py`` blijft de centrale rekenmotor.
- De Streamlit-app blijft vooral schermopbouw.
- De vertaling naar dagelijks databeheer is testbaar zonder browser.
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
    "informatief": 0,
    "aandacht": 1,
    "controleer": 2,
    "niet gebruiken zonder handmatige beoordeling": 3,
    "onbekend": 1,
}

# Voorstellen korter dan deze drempel zijn voor de databeheerder geen normale
# onderhoudsprojecten. Ze blijven zichtbaar, maar komen als apart soort
# actiepunt in de werklijst.
MICRO_PROPOSAL_MAX_LENGTH_M = 25.0


PROJECT_ADVISOR_MAIN_COLUMNS: tuple[str, ...] = (
    "eindadvies",
    "werkadvies",
    "voorstelcategorie",
    "iassetvergelijking",
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
    "werklijst_reden",
    "hoofdmelding",
    "contextmelding",
)


PROJECT_ADVISOR_WORKLIST_COLUMNS: tuple[str, ...] = (
    "werkadvies",
    "werklijst_reden",
    "voorstelcategorie",
    "iassetvergelijking",
    "eindadvies",
    "adviesstatus",
    "datakwaliteitstatus",
    "grensstatus",
    "referentieasstatus",
    "onderhoudsproject_voorgesteld",
    "project_type",
    "fysiek_begin_km",
    "fysiek_eind_km",
    "fysiek_lengte_m",
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
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


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
    """Bepaal de urgentie van een status."""
    return STATUS_ORDER.get(_status_value(value, "onbekend"), 1)


def _number(value: Any) -> float | None:
    """Lees een getal robuust uit een iASSET-/CSV-waarde."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        text_value = str(value).replace(",", ".")
        number = float(text_value)
    except (TypeError, ValueError):
        return None

    return number if number == number else None


def _has_text_signal(row: pd.Series, columns: Iterable[str]) -> bool:
    """Controleer of één van de genoemde kolommen een inhoudelijk signaal bevat."""
    for column in columns:
        if column in row.index and _text(row.get(column)):
            return True
    return False


def _numeric_abs_ge(row: pd.Series, columns: Iterable[str], threshold: float) -> bool:
    """Controleer of één van de numerieke kolommen boven een absolute drempel zit."""
    for column in columns:
        value = _number(row.get(column)) if column in row.index else None
        if value is not None and abs(value) >= threshold:
            return True
    return False


def _contains_any(row: pd.Series, columns: Iterable[str], words: Iterable[str]) -> bool:
    """Zoek sleutelwoorden in bestaande tekstkolommen."""
    haystack = " ".join(_text(row.get(column)).lower() for column in columns if column in row.index)
    return any(word in haystack for word in words)


def derive_iassetvergelijking(row: pd.Series) -> str:
    """
    Vertaal de vergelijking met bestaande iASSET-projecten naar één status.

    Bestaande iASSET-projecten sturen de greenfield-voorstellen niet, maar voor
    dagelijks databeheer zijn ze wel een belangrijk controlesignaal.
    """
    direct_value = _status_value(row.get("vergelijking_iasset_status", ""), default="")
    if direct_value in {"ok", "aandacht", "controleer"}:
        return direct_value

    context = " ".join(
        _text(row.get(column)).lower()
        for column in ("hoofdmelding", "contextmelding", "bestaande_onderhoudsprojecten")
        if column in row.index
    )
    if "voorgestelde naam wijkt af" in context or "wijkt af van bestaande iasset" in context:
        return "aandacht"
    if "bestaande iasset-naam komt overeen" in context:
        return "ok"
    return "onbekend"


def derive_voorstelcategorie(row: pd.Series) -> str:
    """
    Maak onderscheid tussen normale projectvoorstellen en randgevallen.

    Een nulmeter- of microvoorstel kan technisch uit de engine komen, maar is
    voor databeheer geen normaal onderhoudsproject. Daarom krijgt het een aparte
    categorie in de presentatie en werklijst.
    """
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
        ),
    ):
        return "buiten ijkbereik"

    length_m = _number(row.get("fysiek_lengte_m"))
    if length_m is not None and length_m <= MICRO_PROPOSAL_MAX_LENGTH_M:
        return "micro-/eindzonevoorstel"

    begin_name = _number(row.get("naam_begin"))
    end_name = _number(row.get("naam_eind"))
    if begin_name is not None and end_name is not None and abs(begin_name - end_name) < 0.0001:
        return "micro-/eindzonevoorstel"

    proposed_name = _text(row.get("onderhoudsproject_voorgesteld", ""))
    if proposed_name:
        match = proposed_name.rsplit("-", 2)
        if len(match) == 3 and match[-1] == match[-2]:
            return "micro-/eindzonevoorstel"

    return "regulier projectvoorstel"


def derive_datakwaliteitstatus(row: pd.Series) -> str:
    """
    Bepaal de aparte datakwaliteitstatus.

    Deze status mag het inhoudelijke projectadvies niet automatisch in de
    werklijst zetten. Ontbrekende paspoortdata blijft zichtbaar, maar wordt pas
    een actiepunt als er daarnaast een echt project-, grens- of iASSET-signaal is.
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

    # Afwijkingen vanaf 15 meter zijn relevant voor de kaartcontrole. Ze maken
    # het projectvoorstel niet automatisch onbruikbaar.
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


def derive_adviesstatus(row: pd.Series) -> str:
    """
    Bepaal de inhoudelijke adviesstatus voor het projectvoorstel.

    v0.35.5 kon een voorstel op ``aandacht`` zetten door ondersteunende signalen
    zoals besteknummer of grenscontext. Voor de dagelijkse Project Adviseur is
    het inhoudelijke voorstel pas aandacht/controleren wanneer het afwijkt van
    iASSET, buiten ijkbereik valt of door de engine zelf zonder vergelijkingsdata
    als controlepunt is gemarkeerd.
    """
    source_status = _status_value(row.get("status_voorstel", row.get("status", "")), default="")
    iasset_status = _status_value(row.get("iassetvergelijking", ""), default="onbekend")
    category = _text(row.get("voorstelcategorie", "regulier projectvoorstel")).lower()

    if category == "buiten ijkbereik":
        return "controleer"
    if category == "micro-/eindzonevoorstel":
        return "controleer"

    if iasset_status == "controleer":
        return "controleer"
    if iasset_status == "aandacht":
        return "aandacht"

    # Als iASSET overeenkomt, beschouwen we ondersteunende bestek-/datakwaliteit-
    # en grenssignalen niet als inhoudelijk projectprobleem. Die blijven apart
    # zichtbaar in datakwaliteitstatus en grensstatus.
    if iasset_status == "ok":
        return "ok"

    return source_status if source_status else "ok"


def _mark_terminal_boundary_attention(result: pd.DataFrame) -> pd.Series:
    """
    Markeer voorstellen waarvan de eindgrens bij het laatste hm-bereik ligt.

    Waarom op tabelniveau?
    Of een grens aan het einde van de weg ligt kun je niet betrouwbaar uit één
    rij afleiden. We gebruiken daarom de hoogste naam-eindwaarde in de huidige
    wegselectie. Dit blijft presentatielogica: de engine zelf verandert niet.
    """
    if result.empty or "naam_eind" not in result.columns:
        return pd.Series(False, index=result.index)

    numeric_end = pd.to_numeric(result["naam_eind"], errors="coerce")
    if numeric_end.notna().sum() < 2:
        return pd.Series(False, index=result.index)

    max_end = numeric_end.max()
    at_last_name = numeric_end >= (max_end - 0.0001)

    end_interval_attention = pd.Series(False, index=result.index)
    if "eind_hm_interval_afwijking_m" in result.columns:
        end_interval_attention = (
            pd.to_numeric(result["eind_hm_interval_afwijking_m"], errors="coerce").abs() >= 15.0
        ).fillna(False)

    end_diagnosis = (
        result["eind_grensdiagnose"].fillna("").astype(str).str.strip() != ""
        if "eind_grensdiagnose" in result.columns
        else pd.Series(False, index=result.index)
    )

    return (at_last_name & (end_interval_attention | end_diagnosis)).fillna(False)


def combine_eindadvies(
    adviesstatus: str,
    datakwaliteitstatus: str,
    grensstatus: str,
    referentieasstatus: str,
) -> str:
    """
    Maak een algemeen leesbaar eindadvies op basis van gescheiden statussen.

    Deze functie blijft beschikbaar voor simpele situaties. De volledige
    Project Adviseur gebruikt daarnaast rijcontext, bijvoorbeeld iASSET-
    vergelijking en microvoorstellen.
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


def derive_eindadvies(row: pd.Series) -> str:
    """Maak een databeheergericht eindadvies voor één projectvoorstel."""
    category = _text(row.get("voorstelcategorie", "regulier projectvoorstel")).lower()
    iasset_status = _status_value(row.get("iassetvergelijking", ""), default="onbekend")

    if category == "buiten ijkbereik":
        return "Niet gebruiken zonder handmatige beoordeling"
    if category == "micro-/eindzonevoorstel":
        return "Niet als regulier onderhoudsproject gebruiken"
    if iasset_status in {"aandacht", "controleer"}:
        return "Controleer verschil met iASSET"
    if bool(row.get("eindgrens_bij_laatste_hm_aandacht", False)):
        return "Controleer eindgrens op kaart"
    if _status_rank(row.get("grensstatus", "ok")) >= 2 or _status_rank(row.get("referentieasstatus", "ok")) >= 2:
        return "Controleer grens/ijking"
    if _status_rank(row.get("datakwaliteitstatus", "ok")) >= 1:
        return "Akkoord; datakwaliteit controleren"
    if _status_rank(row.get("grensstatus", "ok")) == 1:
        return "Akkoord; grensmelding meenemen op kaart"
    return "Akkoord als kaartbeeld klopt"


def derive_worklist_reason(row: pd.Series) -> str:
    """
    Geef een concrete reden waarom iets in de werklijst moet komen.

    Een lege reden betekent: niet in de werklijst. Zo blijft de werklijst een
    actielijst en geen herhaling van alle diagnose-informatie.
    """
    category = _text(row.get("voorstelcategorie", "regulier projectvoorstel")).lower()
    iasset_status = _status_value(row.get("iassetvergelijking", ""), default="onbekend")

    if category == "buiten ijkbereik":
        return "Begin/eind ligt buiten het betrouwbare ijkbereik; beoordeel handmatig op kaart."
    if category == "micro-/eindzonevoorstel":
        return "Zeer kort of nulmeter voorstel; niet als normaal onderhoudsproject behandelen."
    if iasset_status in {"aandacht", "controleer"}:
        return "Voorgesteld onderhoudsproject wijkt af van bestaande iASSET-indeling."
    if bool(row.get("eindgrens_bij_laatste_hm_aandacht", False)):
        return "Eindgrens ligt bij het laatste hm-bereik met afwijkend interval; controleer op kaart."
    if _status_rank(row.get("referentieasstatus", "ok")) >= 2 or _status_rank(row.get("grensstatus", "ok")) >= 2:
        return "Grens of ijking vraagt handmatige beoordeling."
    if _status_rank(row.get("adviesstatus", "ok")) >= 2:
        return "Projectvoorstel vraagt handmatige beoordeling."

    return ""


def derive_work_advice(row: pd.Series) -> str:
    """Vertaal de werklijstreden naar een korte actie voor de databeheerder."""
    reason = _text(row.get("werklijst_reden", ""))
    if not reason:
        if _status_rank(row.get("datakwaliteitstatus", "ok")) >= 1:
            return "Geen directe actie; dataveld controleren wanneer je dit project verwerkt"
        if _status_rank(row.get("grensstatus", "ok")) >= 1:
            return "Geen directe actie; kaartbeeld meenemen bij beoordeling"
        return "Geen directe actie"
    if "iASSET" in reason:
        return "Controleer iASSET-verschil"
    if "kort" in reason or "nulmeter" in reason:
        return "Beoordeel micro/eindzone"
    if "ijkbereik" in reason or "Grens" in reason or "Eindgrens" in reason:
        return "Controleer grens op kaart"
    return "Handmatig beoordelen"


def build_project_advisor_proposal_table(project_proposals: pd.DataFrame | None) -> pd.DataFrame:
    """
    Voeg presentatiestatussen toe aan de bestaande projectvoorstellen.

    De invoer wordt niet gemuteerd. Ontbrekende kolommen leveren veilige lege
    waarden op, zodat oude of incomplete diagnoseframes de UI niet laten crashen.
    """
    if project_proposals is None or project_proposals.empty:
        return pd.DataFrame(columns=[*PROJECT_ADVISOR_MAIN_COLUMNS, "voorstel_id"])

    result = project_proposals.copy()

    result["iassetvergelijking"] = result.apply(derive_iassetvergelijking, axis=1)
    result["voorstelcategorie"] = result.apply(derive_voorstelcategorie, axis=1)
    result["datakwaliteitstatus"] = result.apply(derive_datakwaliteitstatus, axis=1)
    result["grensstatus"] = result.apply(derive_grensstatus, axis=1)
    result["referentieasstatus"] = result.apply(derive_referentieasstatus, axis=1)
    result["eindgrens_bij_laatste_hm_aandacht"] = _mark_terminal_boundary_attention(result)
    result["adviesstatus"] = result.apply(derive_adviesstatus, axis=1)
    result["eindadvies"] = result.apply(derive_eindadvies, axis=1)
    result["werklijst_reden"] = result.apply(derive_worklist_reason, axis=1)
    result["in_werklijst"] = result["werklijst_reden"].map(lambda value: bool(_text(value)))
    result["werkadvies"] = result.apply(derive_work_advice, axis=1)

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

    worklist_count = int(table["in_werklijst"].fillna(False).astype(bool).sum()) if "in_werklijst" in table.columns else 0
    iasset_attention = (
        int(table["iassetvergelijking"].fillna("").astype(str).str.lower().isin(["aandacht", "controleer"]).sum())
        if "iassetvergelijking" in table.columns
        else int((comparison_status.isin(["aandacht", "controleer"])).sum()) if not comparison_status.empty else 0
    )
    micro_count = (
        int((table["voorstelcategorie"].fillna("").astype(str).str.lower() == "micro-/eindzonevoorstel").sum())
        if "voorstelcategorie" in table.columns
        else 0
    )
    no_direct_action = int(len(table) - worklist_count) if not table.empty else 0

    return {
        "voorstellen": int(len(table)),
        "geen_directe_actie": no_direct_action,
        "werklijstregels": worklist_count,
        "iasset_verschillen": iasset_attention,
        "micro_eindzone": micro_count,
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
    """Selecteer alleen voorstellen waarvoor een concrete databeheeractie is benoemd."""
    if advisor_table is None or advisor_table.empty:
        return pd.DataFrame(columns=PROJECT_ADVISOR_WORKLIST_COLUMNS)

    table = advisor_table.copy()

    if "werklijst_reden" not in table.columns:
        table["werklijst_reden"] = table.apply(derive_worklist_reason, axis=1)
    if "in_werklijst" not in table.columns:
        table["in_werklijst"] = table["werklijst_reden"].map(lambda value: bool(_text(value)))
    if "werkadvies" not in table.columns:
        table["werkadvies"] = table.apply(derive_work_advice, axis=1)

    worklist = table[table["in_werklijst"].fillna(False).astype(bool)].copy()
    columns = [column for column in PROJECT_ADVISOR_WORKLIST_COLUMNS if column in worklist.columns]
    return worklist[columns] if columns else worklist
