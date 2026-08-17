"""
Zichtbare onderhoudscomplexlaag voor Project Adviseur.

Deze module verandert de ruwe projectas-engine niet. Zij vertaalt de ruwe
projectvoorstellen naar een werklaag die dichter bij het dagelijkse
databeheerproces ligt:

- dubbele projectnamen worden als één zichtbaar cluster gepresenteerd;
- micro-/eindzones blijven controlepunten en worden niet als normaal
  onderhoudscomplex in het concept-N-wegendocument gezet;
- zeer korte technische segmenten blijven controlepunten, tenzij ze onderdeel
  zijn van een samengevoegd cluster;
- objectfamilie-mismatches worden als controlepunt benoemd, niet als nieuwe
  projectbasis.

Waarom deze laag?
De ruwe projectvoorstellen blijven de technische onderbouwing. De zichtbare
onderhoudscomplexlaag is de werklaag voor Project Adviseur en het
concept-N-wegendocument.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .utils import clean_display_value, sanitize_filename


# Zeer korte reguliere voorstellen worden niet vanzelf als normaal
# onderhoudscomplex in het concept-N-wegendocument gezet. Ze blijven zichtbaar
# in de zichtbare laag en in de controlepunten.
VERY_SHORT_REGULAR_LIMIT_M = 100.0

VISIBLE_COMPLEX_COLUMNS: tuple[str, ...] = (
    "zichtbaar_complex_id",
    "zichtbare_status",
    "zichtbare_klasse",
    "in_concept_nwegendocument",
    "onderhoudsproject_voorgesteld",
    "project_type",
    "project_family",
    "situering",
    "fysiek_begin_km",
    "fysiek_eind_km",
    "fysiek_lengte_m",
    "naam_begin",
    "naam_eind",
    "aantal_ruwe_voorstellen",
    "bron_voorstel_ids",
    "bron_projectnamen",
    "ruwe_voorstelcategorieen",
    "iassetvergelijking",
    "werkadvies",
    "werklijst_reden",
    "objectfamilie_mismatch_aantal",
    "aantal_primaire_objecten",
    "bestaande_onderhoudsprojecten",
    "technisch_profiel",
    "bijzonderheden",
)


def _text(value: Any) -> str:
    """Geef een veilige tekstwaarde zonder NaN/None-ruis."""
    text = clean_display_value(value).strip()
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


def _number(value: Any) -> float | None:
    """Lees een getal robuust uit CSV-/DataFramewaarden."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _bool(value: Any) -> bool:
    """Lees een bool robuust uit echte bools en CSV-tekstwaarden."""
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    return text in {"true", "waar", "ja", "yes", "1"}


def _unique_join(values: Iterable[Any], *, max_items: int = 8, separator: str = "; ") -> str:
    """Maak een compacte unieke opsomming."""
    result: list[str] = []
    for value in values:
        text = _text(value)
        if not text or text in result:
            continue
        result.append(text)

    if not result:
        return ""
    if len(result) <= max_items:
        return separator.join(result)
    return separator.join(result[:max_items]) + f"; … (+{len(result) - max_items})"


def _first_non_empty(values: Iterable[Any]) -> str:
    """Geef de eerste niet-lege waarde uit een reeks."""
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _max_status(values: Iterable[Any]) -> str:
    """Vat iASSET-vergelijking of statuswaarden conservatief samen."""
    order = {
        "ok": 0,
        "akkoord": 0,
        "informatief": 0,
        "aandacht": 1,
        "controleer": 2,
        "niet gebruiken zonder handmatige beoordeling": 3,
    }
    best_text = ""
    best_rank = -1
    for value in values:
        text = _text(value).lower()
        rank = order.get(text, 1 if text else -1)
        if rank > best_rank:
            best_rank = rank
            best_text = _text(value)
    return best_text


def _proposal_family(row: pd.Series) -> str:
    """Normaliseer het voorgestelde projecttype naar een beheerfamilie."""
    project_type = _text(row.get("project_type", "")).upper()
    project_family = _text(row.get("project_family", "")).upper()
    proposed = _text(row.get("onderhoudsproject_voorgesteld", "")).upper()
    combined = " ".join(part for part in (project_type, project_family, proposed) if part)

    if project_type.startswith("HRB") or project_family == "HRB" or "-HRB" in proposed:
        return "HRB"
    if "FP" in combined:
        return "FP"
    if "BB" in combined or "BBLR" in combined:
        return "BB"
    if "LBP" in combined:
        return "LBP"
    if "PW" in combined:
        return "PW"
    return project_family or project_type or "ONBEKEND"


def _object_family_from_subtheme(row: pd.Series) -> str:
    """Bepaal de objectfamilie vanuit het objectsubthema."""
    subthema = _text(row.get("subthema", row.get("Subthema", ""))).lower()

    if "rijstrook" in subthema:
        return "HRB"
    if "fietspad" in subthema:
        return "FP"
    if "parallelweg" in subthema:
        return "PW"
    if "busbaan" in subthema:
        return "BB"
    if "landbouwpad" in subthema:
        return "LBP"

    project_family = _text(row.get("project_family", "")).upper()
    project_type = _text(row.get("project_type", "")).upper()
    return project_family or project_type or "ONBEKEND"


def _accepted_object_families_for_proposal(proposal_family: str) -> set[str]:
    """Geef objectfamilies die zonder controle bij een projectfamilie passen."""
    proposal_family = proposal_family.upper()
    if proposal_family == "HRB":
        return {"HRB"}
    if proposal_family == "FP":
        return {"FP"}
    if proposal_family == "PW":
        return {"PW", "BB", "LBP"}
    if proposal_family in {"BB", "LBP"}:
        return {proposal_family}
    return {proposal_family}


def _split_ids(value: Any) -> list[str]:
    """Lees voorstel-id's uit een enkele waarde of een samengestelde lijst."""
    text = _text(value)
    if not text:
        return []
    parts = []
    for part in text.replace(",", ";").replace("|", ";").split(";"):
        clean = part.strip()
        if clean and clean not in parts:
            parts.append(clean)
    return parts


def _proposal_objects_for_ids(proposal_objects: pd.DataFrame, proposal_ids: Iterable[str]) -> pd.DataFrame:
    """Selecteer objecttoewijzingen voor één of meer ruwe voorstel-id's."""
    ids = {proposal_id for proposal_id in proposal_ids if proposal_id}
    if proposal_objects.empty or "voorstel_id" not in proposal_objects.columns or not ids:
        return pd.DataFrame()
    mask = proposal_objects["voorstel_id"].map(_text).str.strip().isin(ids)
    return proposal_objects[mask].copy()


def _object_family_mismatch_count(proposal_rows: pd.DataFrame, proposal_objects: pd.DataFrame) -> int:
    """Tel objecten die niet bij de projectfamilie van hun voorstel passen."""
    if proposal_rows.empty or proposal_objects.empty:
        return 0

    mismatch_count = 0
    for _, proposal_row in proposal_rows.iterrows():
        proposal_id = _text(proposal_row.get("voorstel_id", ""))
        family = _proposal_family(proposal_row)
        accepted = _accepted_object_families_for_proposal(family)
        object_rows = _proposal_objects_for_ids(proposal_objects, [proposal_id])
        if object_rows.empty:
            continue

        for _, object_row in object_rows.iterrows():
            object_family = _object_family_from_subtheme(object_row)
            if object_family and object_family not in accepted and object_family != "ONBEKEND":
                mismatch_count += 1

    return mismatch_count


def _aggregate_numeric(proposal_rows: pd.DataFrame, column: str, method: str) -> float | None:
    """Agregeer een numerieke kolom robuust."""
    if column not in proposal_rows.columns:
        return None
    values = [_number(value) for value in proposal_rows[column]]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    if method == "sum":
        return sum(values)
    return values[0]


def _aggregate_length(proposal_rows: pd.DataFrame) -> float | None:
    """Bepaal de zichtbare lengte als trajectrange, met fallback op lengte-som."""
    begin_m = _aggregate_numeric(proposal_rows, "fysiek_begin_m", "min")
    end_m = _aggregate_numeric(proposal_rows, "fysiek_eind_m", "max")
    if begin_m is not None and end_m is not None and end_m >= begin_m:
        return round(end_m - begin_m, 3)

    begin_km = _aggregate_numeric(proposal_rows, "fysiek_begin_km", "min")
    end_km = _aggregate_numeric(proposal_rows, "fysiek_eind_km", "max")
    if begin_km is not None and end_km is not None and end_km >= begin_km:
        return round((end_km - begin_km) * 1000.0, 3)

    return _aggregate_numeric(proposal_rows, "fysiek_lengte_m", "sum")


def _derive_visible_class(
    proposal_rows: pd.DataFrame,
    *,
    duplicate_group: bool,
    mismatch_count: int,
) -> tuple[str, str, bool, str]:
    """
    Bepaal klasse/status voor de zichtbare onderhoudscomplexlaag.

    Retourneert:
    - zichtbare_status
    - zichtbare_klasse
    - in_concept_nwegendocument
    - toelichting
    """
    categories = [
        _text(value).lower()
        for value in proposal_rows.get("voorstelcategorie", pd.Series(dtype="object"))
    ]
    length = _aggregate_length(proposal_rows)
    work_reasons = " | ".join(
        _text(value)
        for value in proposal_rows.get("werklijst_reden", pd.Series(dtype="object"))
        if _text(value)
    ).lower()

    all_micro = bool(categories) and all("micro" in category or "eindzone" in category for category in categories)
    any_micro = any("micro" in category or "eindzone" in category for category in categories)
    any_outside = any("buiten ijkbereik" in category for category in categories)
    any_regular = any("regulier" in category for category in categories)

    if all_micro:
        return (
            "controlepunt",
            "micro-/eindzonecontrole",
            False,
            "Niet als regulier onderhoudscomplex opnemen; beoordeel als micro-/eindzonecontrole.",
        )

    if duplicate_group:
        return (
            "controle",
            "controlecluster - dubbele projectnaam samengevoegd",
            True,
            "Meerdere ruwe voorstellen met dezelfde projectnaam zijn samengevoegd in één zichtbare regel.",
        )

    if any_micro:
        return (
            "controlepunt",
            "micro-/eindzonecontrole",
            False,
            "Ruw voorstel is zeer kort of nulmeterachtig; niet als normaal onderhoudscomplex opnemen.",
        )

    if any_regular and length is not None and length < VERY_SHORT_REGULAR_LIMIT_M:
        return (
            "controlepunt",
            "kort technisch segment",
            False,
            f"Korter dan {VERY_SHORT_REGULAR_LIMIT_M:.0f} m; beoordeel als technisch segment bij aangrenzend traject.",
        )

    if any_outside:
        return (
            "controle",
            "onderhoudscomplex met ijkcontrole",
            True,
            "Traject blijft zichtbaar, maar grens/ijking vraagt kaart- en broncontrole.",
        )

    if mismatch_count:
        return (
            "controle",
            "onderhoudscomplex met objectfamiliecontrole",
            True,
            "Bevat objecten waarvan de objectfamilie niet vanzelf bij de projectfamilie past.",
        )

    if "iasset" in work_reasons:
        return (
            "controle",
            "onderhoudscomplex met iASSET-verschil",
            True,
            "Zichtbaar complex wijkt af van bestaande iASSET-indeling; beoordeel met actuele brondata.",
        )

    return (
        "concept",
        "regulier onderhoudscomplex",
        True,
        "Zichtbaar onderhoudscomplex op basis van actuele iASSET-objectdata en projectaslogica.",
    )


def _build_visible_row(
    proposal_rows: pd.DataFrame,
    proposal_objects: pd.DataFrame,
    visible_index: int,
    *,
    duplicate_group: bool,
) -> dict[str, Any]:
    """Bouw één zichtbare onderhoudscomplexregel uit één of meer ruwe voorstellen."""
    first = proposal_rows.iloc[0]
    proposal_ids = [
        _text(value)
        for value in proposal_rows.get("voorstel_id", pd.Series(dtype="object"))
        if _text(value)
    ]
    mismatch_count = _object_family_mismatch_count(proposal_rows, proposal_objects)

    visible_status, visible_class, in_nwegendoc, class_note = _derive_visible_class(
        proposal_rows,
        duplicate_group=duplicate_group,
        mismatch_count=mismatch_count,
    )

    project_type = _first_non_empty(proposal_rows.get("project_type", []))
    family = _first_non_empty(proposal_rows.get("project_family", []))
    visible_id_family = (_proposal_family(first) or project_type or family or "ONB").replace(" ", "_")

    begin_km = _aggregate_numeric(proposal_rows, "fysiek_begin_km", "min")
    end_km = _aggregate_numeric(proposal_rows, "fysiek_eind_km", "max")
    length_m = _aggregate_length(proposal_rows)

    notes = [
        class_note,
    ]

    raw_count = len(proposal_rows)
    if raw_count > 1:
        notes.append(f"Gebaseerd op {raw_count} ruwe projectvoorstellen.")

    if mismatch_count:
        notes.append(f"Objectfamiliecontrole: {mismatch_count} object(en) vragen controle.")

    work_advice = _unique_join(proposal_rows.get("werkadvies", []), max_items=4)
    work_reasons = _unique_join(proposal_rows.get("werklijst_reden", []), max_items=4)
    if work_advice:
        notes.append(f"Werkadvies: {work_advice}.")
    if work_reasons:
        notes.append(f"Onderbouwing: {work_reasons}.")

    visible_name = _first_non_empty(proposal_rows.get("onderhoudsproject_voorgesteld", []))

    return {
        "zichtbaar_complex_id": f"ZOC-{visible_id_family}-{visible_index:04d}",
        "voorstel_id": f"ZOC-{visible_id_family}-{visible_index:04d}",
        "zichtbare_status": visible_status,
        "zichtbare_klasse": visible_class,
        "in_concept_nwegendocument": bool(in_nwegendoc),
        "onderhoudsproject_voorgesteld": visible_name,
        "project_type": project_type,
        "project_family": family,
        "situering": _first_non_empty(proposal_rows.get("situering", [])),
        "fysiek_begin_m": _aggregate_numeric(proposal_rows, "fysiek_begin_m", "min"),
        "fysiek_eind_m": _aggregate_numeric(proposal_rows, "fysiek_eind_m", "max"),
        "fysiek_begin_km": begin_km,
        "fysiek_eind_km": end_km,
        "fysiek_lengte_m": length_m,
        "naam_begin": _aggregate_numeric(proposal_rows, "naam_begin", "min") or _first_non_empty(proposal_rows.get("naam_begin", [])),
        "naam_eind": _aggregate_numeric(proposal_rows, "naam_eind", "max") or _first_non_empty(proposal_rows.get("naam_eind", [])),
        "aantal_ruwe_voorstellen": raw_count,
        "bron_voorstel_ids": "; ".join(proposal_ids),
        "bron_projectnamen": _unique_join(proposal_rows.get("onderhoudsproject_voorgesteld", []), max_items=12),
        "ruwe_voorstelcategorieen": _unique_join(proposal_rows.get("voorstelcategorie", []), max_items=6),
        "iassetvergelijking": _max_status(proposal_rows.get("iassetvergelijking", [])),
        "werkadvies": work_advice,
        "werklijst_reden": work_reasons,
        "objectfamilie_mismatch_aantal": mismatch_count,
        "aantal_primaire_objecten": _aggregate_numeric(proposal_rows, "aantal_primaire_objecten", "sum"),
        "bestaande_onderhoudsprojecten": _unique_join(proposal_rows.get("bestaande_onderhoudsprojecten", []), max_items=8),
        "technisch_profiel": _unique_join(proposal_rows.get("technisch_profiel", []), max_items=3),
        "bijzonderheden": " ".join(note for note in notes if note),
    }


def build_visible_maintenance_complex_table(
    advisor_table: pd.DataFrame | None,
    proposal_objects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bouw een zichtbare onderhoudscomplexlaag uit de ruwe Project Adviseur-voorstellen.

    De invoer wordt niet gemuteerd. Ruwe projectvoorstellen blijven beschikbaar
    in ``Projectadvies_Voorstellen``; deze tabel is de werklaag voor het
    concept-N-wegendocument en snelle databeheerbeoordeling.
    """
    if advisor_table is None or advisor_table.empty:
        return pd.DataFrame(columns=VISIBLE_COMPLEX_COLUMNS)

    table = advisor_table.copy()
    objects = proposal_objects if isinstance(proposal_objects, pd.DataFrame) else pd.DataFrame()

    if "onderhoudsproject_voorgesteld" not in table.columns:
        table["onderhoudsproject_voorgesteld"] = ""

    # Sorteer ruimtelijk vóór groepsvorming, zodat zichtbare complex-id's stabiel zijn.
    sort_columns = [
        column
        for column in ("project_type", "fysiek_begin_km", "fysiek_eind_km", "onderhoudsproject_voorgesteld")
        if column in table.columns
    ]
    if sort_columns:
        table = table.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    name_counts = table["onderhoudsproject_voorgesteld"].map(_text).str.strip().value_counts()
    duplicate_names = {name for name, count in name_counts.items() if name and count > 1}

    rows: list[dict[str, Any]] = []
    consumed_indices: set[int] = set()
    visible_index = 1

    # Eerst dubbele projectnamen samenvoegen tot één zichtbare regel.
    for name in sorted(duplicate_names):
        group = table[table["onderhoudsproject_voorgesteld"].map(_text).str.strip() == name].copy()
        if group.empty:
            continue
        consumed_indices.update(int(index) for index in group.index)
        rows.append(
            _build_visible_row(
                group,
                objects,
                visible_index,
                duplicate_group=True,
            )
        )
        visible_index += 1

    # Daarna alle overige ruwe voorstellen als eigen zichtbare/controleregel.
    for index, row in table.iterrows():
        if int(index) in consumed_indices:
            continue
        proposal_rows = pd.DataFrame([row])
        rows.append(
            _build_visible_row(
                proposal_rows,
                objects,
                visible_index,
                duplicate_group=False,
            )
        )
        visible_index += 1

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=VISIBLE_COMPLEX_COLUMNS)

    sort_columns = [
        column
        for column in ("project_type", "fysiek_begin_km", "fysiek_eind_km", "onderhoudsproject_voorgesteld")
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    # Zorg dat de publieke kolommen eerst staan, maar bewaar extra technische
    # hulpkolommen zoals fysiek_begin_m/fysiek_eind_m voor exports.
    ordered_columns = [column for column in VISIBLE_COMPLEX_COLUMNS if column in result.columns]
    extra_columns = [column for column in result.columns if column not in ordered_columns]
    return result[ordered_columns + extra_columns]


def visible_maintenance_complexes_for_nwegendocument(visible_table: pd.DataFrame | None) -> pd.DataFrame:
    """Selecteer de zichtbare regels die in het concept-N-wegendocument thuishoren."""
    if visible_table is None or visible_table.empty:
        return pd.DataFrame()
    if "in_concept_nwegendocument" not in visible_table.columns:
        return visible_table.copy()
    mask = visible_table["in_concept_nwegendocument"].map(_bool)
    return visible_table[mask].copy()


def visible_maintenance_complex_export_filename(selected_road: str) -> str:
    """Maak een veilige bestandsnaam voor de zichtbare onderhoudscomplexlaag."""
    return f"Projectadvies_Zichtbare_Onderhoudscomplexen_{sanitize_filename(selected_road)}.csv"
