"""
Concept-export in het format van het N-wegendocument.

Deze module maakt géén mutatie in het bestaande N-wegendocument. De export is
een nieuw Excelbestand met concepttabbladen per wegdeeltype. De databeheerder kan
dit naast het handmatige N-wegendocument leggen en daarna bewust overnemen.

Waarom apart van Project Adviseur?
- Project Adviseur blijft de presentatielaag voor de Streamlit-app.
- Deze module vertaalt dezelfde voorstellen naar de werkvorm die databeheer
  dagelijks gebruikt: het N-wegendocument.
- De export is testbaar zonder browser en zonder iASSET-mutatie.
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .utils import clean_display_value, sanitize_filename


NWEGENDOC_HEADERS: tuple[str, ...] = (
    "onderhoudscomplex oud",
    "",
    "onderhoudscomplex nieuw",
    "knip (begin)",
    "knip (einde)",
    "objecten",
    "locatie",
    "besteknummer",
    "documentatie",
    "verharding (begin)",
    "verharding (einde)",
    "verhardingsoort",
    "conservering",
    "jaar aanleg",
    "jaar deklaag",
    "jaar conservering",
    "jaar herstrating",
    "bijzonderheden",
)

NWEGENDOC_DATA_COLUMNS: tuple[str, ...] = (
    "onderhoudscomplex_oud",
    "statuskolom",
    "onderhoudscomplex_nieuw",
    "knip_begin",
    "knip_einde",
    "objecten",
    "locatie",
    "besteknummer",
    "documentatie",
    "verharding_begin",
    "verharding_einde",
    "verhardingsoort",
    "conservering",
    "jaar_aanleg",
    "jaar_deklaag",
    "jaar_conservering",
    "jaar_herstrating",
    "bijzonderheden",
)

NWEGENDOC_SHEET_ORDER: tuple[str, ...] = ("HRB", "PW", "FP")


def _text(value: Any) -> str:
    """Geef een veilige tekstwaarde zonder NaN/None-ruis."""
    text = clean_display_value(value).strip()
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


def _first_non_empty(values: Iterable[Any]) -> str:
    """Geef de eerste inhoudelijke waarde uit een reeks."""
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _unique_join(values: Iterable[Any], *, max_items: int = 6, separator: str = " / ") -> str:
    """Voeg unieke waarden compact samen voor een Excelcel."""
    seen: list[str] = []
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.append(text)

    if not seen:
        return ""

    if len(seen) <= max_items:
        return separator.join(seen)

    visible = seen[:max_items]
    return separator.join(visible) + f" / … (+{len(seen) - max_items})"


def _most_common(values: Iterable[Any]) -> str:
    """Bepaal de meest voorkomende inhoudelijke waarde."""
    cleaned = [_text(value) for value in values]
    cleaned = [value for value in cleaned if value and value.lower() != "<leeg>"]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


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


def _as_meter_value(value: Any) -> int | float | str:
    """
    Schrijf fysieke knipwaarden in meters.

    Het N-wegendocument gebruikt in de handmatige tabbladen meestal fysieke
    meterwaarden, bijvoorbeeld 8605 in plaats van 8.605 km. Daarom gebruiken we
    ``fysiek_begin_m``/``fysiek_eind_m`` en ronden we af op hele meters als dat
    logisch is.
    """
    number = _number(value)
    if number is None:
        return ""
    rounded = round(number)
    if abs(number - rounded) < 0.001:
        return int(rounded)
    return round(number, 3)


def _parse_technisch_profiel(profile: Any) -> dict[str, str]:
    """Parseer 'Veld=waarde, Veld=waarde' uit de projectas-engine."""
    text = _text(profile)
    parsed: dict[str, str] = {}
    if not text:
        return parsed

    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()

    return parsed


def _value_from_objects_or_profile(
    proposal_row: pd.Series,
    object_rows: pd.DataFrame,
    object_column: str,
    profile_key: str | None = None,
) -> str:
    """Vul een N-wegendocumentwaarde vanuit objecttoewijzing met profiel-fallback."""
    if not object_rows.empty and object_column in object_rows.columns:
        value = _most_common(object_rows[object_column])
        if value:
            return value

    if profile_key:
        parsed = _parse_technisch_profiel(proposal_row.get("technisch_profiel", ""))
        value = _text(parsed.get(profile_key, ""))
        if value and value.lower() != "<leeg>":
            return value

    return ""


def _classify_sheet(row: pd.Series) -> str:
    """
    Bepaal op welk concepttabblad het voorstel hoort.

    In het N-wegendocument staan hoofdrijbanen, parallelwegen en fietspaden in
    eigen tabbladen. Busbanen en landbouwpaden worden voor deze concept-export
    bij PW geplaatst, omdat ze in het werkproces op het parallelweg-tabblad
    worden genoteerd.
    """
    project_type = _text(row.get("project_type", "")).upper()
    project_family = _text(row.get("project_family", "")).upper()
    proposed = _text(row.get("onderhoudsproject_voorgesteld", "")).upper()
    combined = " ".join(part for part in (project_type, project_family, proposed) if part)

    if "FP" in combined:
        return "FP"
    if project_type.startswith("HRB") or project_family == "HRB" or "-HRB" in proposed:
        return "HRB"

    # Parallelwegen, busbanen, landbouwpaden en overige niet-HRB/FP voorstellen
    # komen in dit concept bij het PW-tabblad.
    return "PW"


def _proposal_objects_for_id(proposal_objects: pd.DataFrame, proposal_id: str) -> pd.DataFrame:
    """Selecteer objecttoewijzingen voor één voorstel-id."""
    if proposal_objects.empty or "voorstel_id" not in proposal_objects.columns or not proposal_id:
        return pd.DataFrame()
    return proposal_objects[
        proposal_objects["voorstel_id"].map(_text).str.strip() == proposal_id
    ].copy()


def _object_summary(object_rows: pd.DataFrame) -> str:
    """Maak een compacte objectomschrijving voor het N-wegendocument."""
    if object_rows.empty:
        return ""

    candidate_columns = ["naam", "Object naam", "nummer", "Object nummer", "sys_id"]
    for column in candidate_columns:
        if column in object_rows.columns:
            summary = _unique_join(object_rows[column], max_items=8, separator=", ")
            if summary:
                return summary
    return ""


def _build_bijzonderheden(row: pd.Series) -> str:
    """Maak een korte, bruikbare toelichting voor de conceptregel."""
    parts: list[str] = []

    for label, column in (
        ("Werkadvies", "werkadvies"),
        ("Werklijst", "werklijst_reden"),
        ("Eindadvies", "eindadvies"),
        ("iASSET", "iassetvergelijking"),
        ("Categorie", "voorstelcategorie"),
        ("Datakwaliteit", "datakwaliteit_signalen"),
        ("Lokale afwijking", "lokale_afwijkingen"),
    ):
        value = _text(row.get(column, ""))
        if value:
            parts.append(f"{label}: {value}")

    return " | ".join(parts)


def build_nwegendocument_concept_rows(
    advisor_table: pd.DataFrame | None,
    proposal_objects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bouw conceptregels in dezelfde kolomvolgorde als het N-wegendocument.

    De uitvoer bevat extra hulpkolommen ``tabblad`` en ``voorstel_id`` voor de
    Excel-export. Die kolommen worden niet in de tabbladweergave zelf gezet.
    """
    if advisor_table is None or advisor_table.empty:
        columns = ["tabblad", "voorstel_id", *NWEGENDOC_DATA_COLUMNS]
        return pd.DataFrame(columns=columns)

    objects = proposal_objects if isinstance(proposal_objects, pd.DataFrame) else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for _, proposal_row in advisor_table.iterrows():
        proposal_id = _text(proposal_row.get("voorstel_id", ""))
        object_rows = _proposal_objects_for_id(objects, proposal_id)

        besteknummer = _unique_join(object_rows.get("Besteknummer", []), max_items=4) if not object_rows.empty else ""
        if not besteknummer and not object_rows.empty and "besteknummer_norm" in object_rows.columns:
            besteknummer = _unique_join(object_rows["besteknummer_norm"], max_items=4)

        verhardingsoort = _value_from_objects_or_profile(
            proposal_row,
            object_rows,
            "Soort deklaag specifiek",
            "Soort deklaag specifiek",
        )
        if not verhardingsoort:
            verhardingsoort = _value_from_objects_or_profile(
                proposal_row,
                object_rows,
                "Soort verharding_N",
                "Soort verharding_N",
            )

        row = {
            "tabblad": _classify_sheet(proposal_row),
            "voorstel_id": proposal_id,
            "onderhoudscomplex_oud": _text(proposal_row.get("bestaande_onderhoudsprojecten", "")),
            "statuskolom": "",
            "onderhoudscomplex_nieuw": _text(proposal_row.get("onderhoudsproject_voorgesteld", "")),
            "knip_begin": _as_meter_value(proposal_row.get("fysiek_begin_m", "")),
            "knip_einde": _as_meter_value(proposal_row.get("fysiek_eind_m", "")),
            "objecten": _object_summary(object_rows),
            "locatie": "",
            "besteknummer": besteknummer,
            "documentatie": "",
            "verharding_begin": _as_meter_value(proposal_row.get("fysiek_begin_m", "")),
            "verharding_einde": _as_meter_value(proposal_row.get("fysiek_eind_m", "")),
            "verhardingsoort": verhardingsoort,
            "conservering": _value_from_objects_or_profile(proposal_row, object_rows, "Soort conservering", "Soort conservering"),
            "jaar_aanleg": _value_from_objects_or_profile(proposal_row, object_rows, "Jaar aanleg", "Jaar aanleg"),
            "jaar_deklaag": _value_from_objects_or_profile(proposal_row, object_rows, "Jaar deklaag", "Jaar deklaag"),
            "jaar_conservering": _value_from_objects_or_profile(proposal_row, object_rows, "Jaar conservering", "Jaar conservering"),
            "jaar_herstrating": _value_from_objects_or_profile(proposal_row, object_rows, "Jaar herstrating", "Jaar herstrating"),
            "bijzonderheden": _build_bijzonderheden(proposal_row),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_columns = [column for column in ("tabblad", "knip_begin", "knip_einde", "onderhoudscomplex_nieuw") if column in result.columns]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return result[["tabblad", "voorstel_id", *NWEGENDOC_DATA_COLUMNS]]


def _safe_sheet_name(name: str) -> str:
    """Maak een Excel-tabbladnaam met maximaal 31 tekens."""
    invalid = "[]:*?/\\"
    safe = "".join("_" if char in invalid else char for char in name)
    return safe[:31] or "Sheet"


def _write_nwegendoc_sheet(writer: pd.ExcelWriter, sheet_name: str, rows: pd.DataFrame, selected_road: str) -> None:
    """Schrijf één concepttabblad in N-wegendocument-layout."""
    workbook = writer.book
    worksheet = workbook.create_sheet(_safe_sheet_name(sheet_name))
    writer.sheets[worksheet.title] = worksheet

    # Bovenblok, gebaseerd op de handmatige N-wegendocumenttabbladen.
    worksheet.append(["Status", "", "Concept-export Project Adviseur", ""])
    worksheet.append([])
    worksheet.append(list(NWEGENDOC_HEADERS))
    worksheet.append(["Oud onderhoudscomplex verwijderd", "", "Alleen paspoort gevuld"])
    worksheet.append(["", "", "Paspoort gevuld + nieuw onderhoudscomplex aangemaakt"])
    worksheet.append(["", "", "Maatregeltoetsdocument bijgewerkt"])
    worksheet.append([])

    display_rows = rows[list(NWEGENDOC_DATA_COLUMNS)] if not rows.empty else pd.DataFrame(columns=NWEGENDOC_DATA_COLUMNS)
    for values in display_rows.itertuples(index=False, name=None):
        worksheet.append(list(values))

    # Opmaak blijft bewust eenvoudig maar herkenbaar: kopregels, bevroren rijen,
    # filter en goed leesbare kolombreedtes.
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    status_fill = PatternFill("solid", fgColor="F2F2F2")
    note_fill = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = status_fill

    for row_idx in (3, 4, 5, 6):
        for cell in worksheet[row_idx]:
            cell.fill = header_fill if row_idx == 3 else status_fill
            cell.font = Font(bold=row_idx == 3)
            cell.border = border
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    max_row = worksheet.max_row
    max_col = len(NWEGENDOC_HEADERS)
    for row in worksheet.iter_rows(min_row=8, max_row=max_row, max_col=max_col):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    if max_row >= 8:
        worksheet.auto_filter.ref = f"A3:R{max_row}"

    worksheet.freeze_panes = "A8"

    widths = {
        "A": 24,
        "B": 4,
        "C": 24,
        "D": 12,
        "E": 12,
        "F": 34,
        "G": 22,
        "H": 20,
        "I": 28,
        "J": 13,
        "K": 13,
        "L": 24,
        "M": 18,
        "N": 12,
        "O": 12,
        "P": 16,
        "Q": 16,
        "R": 60,
    }
    for column_letter, width in widths.items():
        worksheet.column_dimensions[column_letter].width = width

    if max_row >= 8:
        for cell in worksheet[f"R8:R{max_row}"]:
            cell[0].fill = note_fill

    worksheet["D1"] = selected_road


def _write_dataframe_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    """Schrijf een ondersteunend tabblad met simpele tabelopmaak."""
    safe_name = _safe_sheet_name(sheet_name)
    df_to_write = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    df_to_write.to_excel(writer, sheet_name=safe_name, index=False)

    worksheet = writer.sheets[safe_name]
    if worksheet.max_row > 1:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

    from openpyxl.styles import Font, PatternFill, Alignment

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    for column_cells in worksheet.columns:
        column_letter = column_cells[0].column_letter
        max_length = max(len(_text(cell.value)) for cell in column_cells[:100])
        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 10), 60)


def build_nwegendocument_concept_workbook_bytes(
    advisor_table: pd.DataFrame | None,
    proposal_objects: pd.DataFrame | None = None,
    run_report: pd.DataFrame | None = None,
    *,
    selected_road: str = "",
    app_version: str = "",
) -> bytes:
    """
    Maak een Excelbestand met concepttabbladen in N-wegendocument-format.

    Dit bestand is bedoeld om naast het bestaande N-wegendocument te leggen. Het
    overschrijft geen bronbestand en is geen directe iASSET-mutatie.
    """
    concept_rows = build_nwegendocument_concept_rows(advisor_table, proposal_objects)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Bewaar het standaardblad tijdelijk. We verwijderen het pas nadat
        # pandas/openpyxl minimaal één echt werkblad heeft aangemaakt.
        default_sheet = writer.book.active

        summary_rows = [
            {"onderdeel": "Doel", "waarde": "Concepttabblad in N-wegendocument-format"},
            {"onderdeel": "Weg", "waarde": selected_road or "onbekend"},
            {"onderdeel": "App-versie", "waarde": app_version or "onbekend"},
            {
                "onderdeel": "Belangrijk",
                "waarde": "Concept-export: controleer altijd in kaart, iASSET en N-wegendocument voordat je gegevens overneemt.",
            },
            {
                "onderdeel": "Niet automatisch gevuld",
                "waarde": "locatie, documentatie en menselijke opmerkingen blijven grotendeels handmatig.",
            },
        ]
        _write_dataframe_sheet(writer, "Samenvatting", pd.DataFrame(summary_rows))

        if (
            default_sheet is not None
            and default_sheet.title in writer.book.sheetnames
            and default_sheet.max_row == 1
            and default_sheet.max_column == 1
            and default_sheet["A1"].value is None
            and len(writer.book.worksheets) > 1
        ):
            writer.book.remove(default_sheet)

        for tab in NWEGENDOC_SHEET_ORDER:
            rows_for_tab = concept_rows[concept_rows["tabblad"] == tab].copy() if not concept_rows.empty else concept_rows.copy()
            if rows_for_tab.empty and tab != "HRB":
                # Maak PW/FP alleen aan als ze echt voorkomen. HRB blijft bestaan
                # zodat een lege run voor een hoofdrijbaan herkenbaar is.
                continue
            _write_nwegendoc_sheet(writer, f"{selected_road or 'Weg'} ({tab})", rows_for_tab, selected_road)

        not_auto = pd.DataFrame(
            [
                {
                    "veld": "locatie",
                    "toelichting": "Herkenbare punten zoals brug, kruising of rotonde worden niet betrouwbaar uit de paspoortdata afgeleid.",
                },
                {
                    "veld": "documentatie",
                    "toelichting": "Dielplak-/besteklinks zitten niet altijd in de iASSET-export en blijven daarom leeg tenzij brondata dit bevat.",
                },
                {
                    "veld": "objecten",
                    "toelichting": "Compacte samenvatting uit objecttoewijzing; controleer bij complexe kruisingen of rotondes op kaart.",
                },
                {
                    "veld": "micro-/eindzone",
                    "toelichting": "Deze regels staan in bijzonderheden en zijn niet bedoeld als regulier onderhoudscomplex zonder beoordeling.",
                },
                {
                    "veld": "buiten ijkbereik",
                    "toelichting": "Niet overnemen zonder handmatige kaart- en broncontrole.",
                },
            ]
        )
        _write_dataframe_sheet(writer, "Niet automatisch gevuld", not_auto)

        if isinstance(run_report, pd.DataFrame) and not run_report.empty:
            _write_dataframe_sheet(writer, "Runrapport", run_report)

        if not concept_rows.empty:
            _write_dataframe_sheet(writer, "Conceptregels_data", concept_rows)

    output.seek(0)
    return output.getvalue()


def nwegendocument_export_filename(selected_road: str) -> str:
    """Maak een consistente bestandsnaam voor de concept-export."""
    road = sanitize_filename(selected_road or "weg")
    return f"Projectadvies_Nwegendocument_{road}.xlsx"
