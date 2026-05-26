"""
Controle tussen paspoortexport en onderhoudsexport.

Deze module bevat de eerste Fase-4-controle: objecten uit de paspoortexport
worden naast onderhoudsregels uit de onderhoudsexport gelegd.

Belangrijk ontwerpprincipe:
- Deze module voert géén mutaties uit.
- De controle is bedoeld als veilig overleg-/acceptatiepakket voordat iets in
  iASSET of het maatregeltoetsdocument wordt verwerkt.
- Inlezen is robuust: afwijkende kopregels of lege bestanden leveren
  waarschuwingen op in plaats van crashes.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import BACKBONE_TYPES
from .domain import is_maintenance_project_exempt
from .utils import clean_display_value, is_empty_value, normalize_text, parse_hm_sort


# Een bronbestand kan een pad zijn, of een tuple uit de Streamlit-uploader:
# ("bestandsnaam.csv", b"...inhoud...").
FileInput = str | Path | tuple[str, bytes]


@dataclass
class MaintenanceReadResult:
    """Resultaat van het veilig inlezen van onderhoudsexportbestanden."""

    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MaintenanceControlResult:
    """Alle tabellen die de Fase-4-controle oplevert."""

    summary: dict[str, int] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    passport_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    maintenance_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


MAINTENANCE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "objectnummer": (
        "object nr",
        "object nr:",
        "objectnummer",
        "object nummer",
        "nummer",
        "asset nummer",
    ),
    "objectnaam": (
        "object naam",
        "object naam:",
        "naam",
    ),
    "thema": ("thema",),
    "subthema": (
        "subthema",
        "sub thema",
        "subthema:",
    ),
    "project_id": (
        "project id",
        "projectid",
        "onderhoudsproject id",
    ),
    "Onderhoudsproject": (
        "project",
        "project:",
        "onderhoudsproject",
        "onderhoud project",
        "onderhoudscomplex",
        "onderhoud complex",
        "onderhoudscomplex nieuw",
        "onderhoudscomplex nieuw 2025",
        "onderhoudsproject nieuw",
    ),
    "project_type": (
        "type",
        "project type",
        "onderhoudstype",
    ),
    "maatregel": (
        "maatregel",
        "maatregel omschrijving",
        "maatregelomschrijving",
        "maatregel omschrijving:",
        "omschrijving maatregel",
    ),
    "eenheid": (
        "eenheid",
        "eenheid:",
        "hoeveelheid eenheid",
    ),
    "hoeveelheid": (
        "hoeveelh",
        "hoeveelh.",
        "hoeveelheid",
        "aantal",
    ),
    "prijs_per_eenheid": (
        "prijs/eenh",
        "prijs per eenheid",
        "prijs_per_eenheid",
    ),
    "totaalprijs": (
        "totaalprijs",
        "totaal prijs",
    ),
    "opmerking": (
        "opmerking",
        "opmerking:",
        "opmerking2",
        "opmerking 2",
    ),
    "project_jaar_gepland": (
        "project_jaar_gepland",
        "project jaar gepland",
        "jaar gepland",
        "gepland jaar",
        "planjaar",
    ),
}


_EXPECTED_MAINTENANCE_COLUMNS = set(MAINTENANCE_COLUMN_ALIASES)
_EXPECTED_HEADER_KEYS: set[str] = set()
for canonical, aliases in MAINTENANCE_COLUMN_ALIASES.items():
    _EXPECTED_HEADER_KEYS.add(canonical)
    _EXPECTED_HEADER_KEYS.update(aliases)


def _normalize_header_key(value: Any) -> str:
    """Normaliseer een kolomkop voor aliasvergelijking."""
    if is_empty_value(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("\ufeff", " ").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _alias_lookup() -> dict[str, str]:
    """Maak een lookup van genormaliseerde kolomkop naar canonieke naam."""
    lookup: dict[str, str] = {}

    for canonical, aliases in MAINTENANCE_COLUMN_ALIASES.items():
        lookup[_normalize_header_key(canonical)] = canonical
        for alias in aliases:
            lookup[_normalize_header_key(alias)] = canonical

    return lookup


_ALIAS_LOOKUP = _alias_lookup()


def _canonical_maintenance_column(value: Any) -> str:
    """Vertaal bekende onderhoudsexport-kolommen naar de appnaam."""
    key = _normalize_header_key(value)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    text = clean_display_value(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_unique_column_names(columns: Iterable[Any]) -> list[str]:
    """Maak kolomnamen uniek, zodat dubbele Excelkoppen geen data overschrijven."""
    result: list[str] = []
    seen: dict[str, int] = {}

    for index, value in enumerate(columns):
        name = clean_display_value(value)
        if not name or name.lower().startswith("unnamed"):
            name = f"onbenoemde_kolom_{index + 1}"

        canonical = _canonical_maintenance_column(name)
        count = seen.get(canonical, 0)
        seen[canonical] = count + 1

        if count:
            result.append(f"{canonical}_{count + 1}")
        else:
            result.append(canonical)

    return result


def _drop_fully_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Verwijder lege Excel-/CSV-regels."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    mask = df.apply(lambda row: all(is_empty_value(value) for value in row), axis=1)
    return df.loc[~mask].copy()


def _score_header_values(values: Iterable[Any]) -> int:
    """
    Score een mogelijke kopregel.

    De onderhoudsexport heeft vaak geen geometrie. Daarom scoren we op Project,
    Object nr en Maatregel in plaats van op WKT-kolommen.
    """
    keys = {_normalize_header_key(value) for value in values}
    canonical_hits = {_ALIAS_LOOKUP[key] for key in keys if key in _ALIAS_LOOKUP}

    score = 0
    if "Onderhoudsproject" in canonical_hits:
        score += 100
    if "objectnummer" in canonical_hits:
        score += 40
    if "maatregel" in canonical_hits:
        score += 30
    if "project_id" in canonical_hits:
        score += 15
    if "hoeveelheid" in canonical_hits:
        score += 10
    if "totaalprijs" in canonical_hits:
        score += 10

    score += len(canonical_hits)
    return score


def _table_from_raw_with_header_scan(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Zoek de meest waarschijnlijke kopregel in een onderhoudsexport.

    iASSET-overzichten hebben vaak titelregels boven de echte tabel. We scannen
    de eerste regels en kiezen de regel met Project/Object/Maatregel-kolommen.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), 0, -1

    max_header_scan_rows = min(40, len(raw_df))
    best_row = -1
    best_score = -1

    for row_index in range(max_header_scan_rows):
        values = list(raw_df.iloc[row_index])
        score = _score_header_values(values)

        if score > best_score:
            best_score = score
            best_row = row_index

    if best_row < 0 or best_score <= 0:
        return pd.DataFrame(), 0, -1

    columns = _make_unique_column_names(raw_df.iloc[best_row].tolist())
    data = raw_df.iloc[best_row + 1 :].copy()
    data.columns = columns
    data = _drop_fully_empty_rows(data)

    return data, best_score, best_row


def _resolve_input_file(input_file: FileInput | Any) -> tuple[str, Path | None, bytes | None]:
    """Zet pad/upload om naar naam, pad en optionele bytes."""
    if isinstance(input_file, tuple) and len(input_file) == 2:
        name, content = input_file
        return str(name), None, bytes(content)

    if hasattr(input_file, "name") and hasattr(input_file, "getvalue"):
        return str(input_file.name), None, bytes(input_file.getvalue())

    path = Path(input_file)
    return str(path), path, None


def _open_for_pandas(path: Path | None, content: bytes | None):
    """Geef pandas een pad of nieuwe BytesIO-stream."""
    if content is not None:
        return BytesIO(content)
    return path



def _read_raw_csv_loose(path: Path | None, content: bytes | None, encoding: str, separator: str | None) -> pd.DataFrame:
    """
    Lees CSV-regels zonder te crashen op titelregels met minder kolommen.

    Pandas behandelt wisselende kolomaantallen soms als "bad lines". Juist bij
    iASSET-overzichten hebben de eerste titelregels vaak minder kolommen dan de
    echte tabel. Daarom lezen we de regels hier zelf en vullen we kortere regels
    aan met lege cellen.
    """
    if content is not None:
        text = content.decode(encoding, errors="replace")
    else:
        text = Path(path).read_text(encoding=encoding, errors="replace") if path is not None else ""

    if separator is None:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ";"
    else:
        delimiter = separator

    rows = list(csv.reader(StringIO(text), delimiter=delimiter))
    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    if width == 0:
        return pd.DataFrame()

    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, dtype=str)


def read_maintenance_csv_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexport uit CSV, inclusief kopregelscan."""
    name, path, content = _resolve_input_file(input_file)
    warnings: list[str] = []

    if path is not None and not path.exists():
        return pd.DataFrame(), [f"Onderhoudsexport niet gevonden: {name}"]

    if content is not None and len(content) == 0:
        return pd.DataFrame(), [f"Onderhoudsexport {Path(name).name} is leeg en is overgeslagen."]

    candidates: list[tuple[int, int, pd.DataFrame, list[str]]] = []

    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        for separator in (";", ",", "\t", None):
            try:
                raw_df = _read_raw_csv_loose(path, content, encoding, separator)
            except Exception:
                continue

            table, score, header_row = _table_from_raw_with_header_scan(raw_df)
            if not table.empty or len(table.columns) > 0:
                candidates.append((score, len(table.columns), table, [f"CSV {Path(name).name}: kopregel gevonden op rij {header_row + 1}."]))

    if not candidates:
        return pd.DataFrame(), [f"CSV {Path(name).name} bevat geen herkenbare onderhoudsexport-kolommen."]

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def read_maintenance_excel_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexport uit Excel, inclusief tabblad- en kopregelscan."""
    name, path, content = _resolve_input_file(input_file)
    warnings: list[str] = []

    if path is not None and not path.exists():
        return pd.DataFrame(), [f"Onderhoudsexport niet gevonden: {name}"]

    if content is not None and len(content) == 0:
        return pd.DataFrame(), [f"Onderhoudsexport {Path(name).name} is leeg en is overgeslagen."]

    try:
        excel_file = pd.ExcelFile(_open_for_pandas(path, content))
    except Exception as exc:
        return pd.DataFrame(), [f"Kon onderhoudsexport {Path(name).name} niet lezen: {exc}"]

    candidates: list[tuple[int, int, int, str, pd.DataFrame, list[str]]] = []

    for sheet_name in excel_file.sheet_names:
        try:
            raw_preview = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None,
                dtype=str,
                keep_default_na=False,
                nrows=80,
            )
        except Exception as exc:
            warnings.append(f"Onderhoudsexport {Path(name).name}: tabblad '{sheet_name}' kon niet worden gescand ({exc}).")
            continue

        table_preview, score, header_row = _table_from_raw_with_header_scan(raw_preview)
        if table_preview.empty and len(table_preview.columns) == 0:
            continue

        candidates.append((score, len(table_preview), len(table_preview.columns), str(sheet_name), table_preview, [f"Onderhoudsexport {Path(name).name}: tabblad '{sheet_name}' gebruikt."]))
        candidates[-1][5].append(
            f"Onderhoudsexport {Path(name).name}: kopregel gevonden op rij {header_row + 1} van tabblad '{sheet_name}'."
        )

    if not candidates:
        warnings.append(f"Onderhoudsexport {Path(name).name} bevat geen tabblad met herkenbare projectkolommen.")
        return pd.DataFrame(), warnings

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _score, _rows, _cols, best_sheet_name, _preview, best_warnings = candidates[0]

    try:
        raw_selected = pd.read_excel(
            excel_file,
            sheet_name=best_sheet_name,
            header=None,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        return pd.DataFrame(), [*warnings, f"Kon tabblad '{best_sheet_name}' uit onderhoudsexport {Path(name).name} niet volledig lezen: {exc}"]

    table, _score, _header_row = _table_from_raw_with_header_scan(raw_selected)
    table = _drop_fully_empty_rows(table)
    return table, [*warnings, *best_warnings]


def read_maintenance_table_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexportbestand uit CSV of Excel."""
    name, path, content = _resolve_input_file(input_file)
    suffix = Path(name).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df, warnings = read_maintenance_excel_safely((name, content) if content is not None else path or name)
    elif suffix in {".csv", ".txt"}:
        df, warnings = read_maintenance_csv_safely((name, content) if content is not None else path or name)
    else:
        df, warnings = read_maintenance_csv_safely((name, content) if content is not None else path or name)
        if df.empty:
            excel_df, excel_warnings = read_maintenance_excel_safely((name, content) if content is not None else path or name)
            return excel_df, [*warnings, *excel_warnings]
        warnings.append(f"Bestandstype van {Path(name).name} is onbekend; succesvol als CSV gelezen.")

    if not df.empty:
        df = df.copy()
        df["bronbestand_onderhoud"] = Path(name).name

    return df, warnings


def read_maintenance_exports(input_files: Sequence[FileInput | Any]) -> MaintenanceReadResult:
    """Lees één of meer onderhoudsexportbestanden samen in."""
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for input_file in input_files:
        df_part, file_warnings = read_maintenance_table_safely(input_file)
        warnings.extend(file_warnings)
        if not df_part.empty:
            frames.append(df_part)

    if not frames:
        return MaintenanceReadResult(dataframe=pd.DataFrame(), warnings=warnings)

    combined = pd.concat(frames, ignore_index=True)
    return MaintenanceReadResult(dataframe=combined, warnings=warnings)


def normalize_project_name(value: Any) -> str:
    """
    Normaliseer onderhoudsprojectnamen voor vergelijking.

    We zijn bewust conservatief: spaties en streepjesvarianten worden gelijk
    getrokken, maar inhoudelijke verschillen zoals voorloopnullen blijven
    zichtbaar. Zo voorkomen we dat de controle verschillende projecten stil
    samenvoegt.
    """
    if is_empty_value(value):
        return ""

    text = clean_display_value(value).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kolom uit een lijst kandidaten terug."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _join_preview(values: Iterable[Any], max_items: int = 6) -> str:
    """Maak een korte, stabiele samenvatting van unieke waarden."""
    cleaned = []
    seen = set()
    for value in values:
        text = clean_display_value(value)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    if len(cleaned) > max_items:
        return ", ".join(cleaned[:max_items]) + f" (+{len(cleaned) - max_items})"
    return ", ".join(cleaned)



def _numeric_score(series: pd.Series) -> int:
    """Tel hoeveel waarden in een kolom numeriek leesbaar zijn."""
    if series is None or series.empty:
        return 0
    numeric = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return int(numeric.notna().sum())


def _best_numeric_column_near(df: pd.DataFrame, preferred_column: str | None, stop_column: str | None = None) -> str | None:
    """
    Kies de meest waarschijnlijke numerieke kolom rond een bekende kop.

    Sommige iASSET-overzichten hebben een samengestelde hoeveelheidkop: de
    eenheidskolom heeft dan de herkenbare kop, terwijl de echte hoeveelheid in
    de eerstvolgende naamloze kolom staat. Daarom zoeken we heel lokaal naar de
    beste numerieke kandidaat.
    """
    if df is None or df.empty or not preferred_column or preferred_column not in df.columns:
        return preferred_column

    columns = list(df.columns)
    start_index = columns.index(preferred_column)
    stop_index = columns.index(stop_column) if stop_column in columns else min(len(columns), start_index + 4)

    candidate_columns = columns[start_index:stop_index]
    if not candidate_columns:
        return preferred_column

    scored = [
        (_numeric_score(df[column]), -offset, column)
        for offset, column in enumerate(candidate_columns)
    ]
    scored.sort(reverse=True)

    if scored and scored[0][0] > 0:
        return scored[0][2]

    return preferred_column


def _safe_sum(series: pd.Series) -> float:
    """Tel een numerieke kolom veilig op, ook bij komma-decimalen."""
    if series.empty:
        return 0.0

    numeric = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return float(numeric.fillna(0).sum())


def summarize_passport_projects(passport_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Groepeer paspoortobjecten per onderhoudsproject."""
    if passport_df is None or passport_df.empty:
        return pd.DataFrame()

    if "Onderhoudsproject" not in passport_df.columns:
        return pd.DataFrame()

    working = passport_df.copy()
    if selected_road and "Wegnummer" in working.columns:
        selected_road_text = clean_display_value(selected_road)
        working = working[working["Wegnummer"].astype(str).str.strip() == selected_road_text].copy()

    working["project_norm"] = working["Onderhoudsproject"].apply(normalize_project_name)
    working = working[working["project_norm"] != ""].copy()

    if working.empty:
        return pd.DataFrame()

    object_column = _first_existing_column(working, ["nummer", "bron_id", "objectnummer", "sys_id"])
    subthema_series = working["subthema"].apply(normalize_text) if "subthema" in working.columns else pd.Series("", index=working.index)
    working["_is_primary"] = subthema_series.isin({normalize_text(value) for value in BACKBONE_TYPES})
    working["_is_exempt"] = working.apply(is_maintenance_project_exempt, axis=1)

    if "hm_sort" in working.columns:
        working["_hm_sort_control"] = pd.to_numeric(working["hm_sort"], errors="coerce")
    elif "Metrering" in working.columns:
        working["_hm_sort_control"] = working["Metrering"].apply(lambda value: parse_hm_sort(value, fallback=float("nan")))
    else:
        working["_hm_sort_control"] = float("nan")

    records: list[dict[str, Any]] = []

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group["Onderhoudsproject"] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm
        object_values = group[object_column] if object_column else group.index.to_series()

        record = {
            "onderhoudsproject": project_name,
            "project_norm": project_norm,
            "paspoort_objecten": int(len(group)),
            "paspoort_unieke_objecten": int(object_values.astype(str).nunique()),
            "paspoort_primaire_objecten": int(group["_is_primary"].sum()),
            "paspoort_secundaire_objecten": int((~group["_is_primary"]).sum()),
            "paspoort_uitzondering_objecten": int(group["_is_exempt"].sum()),
            "paspoort_subthema_samenvatting": _join_preview(group["subthema"]) if "subthema" in group.columns else "",
            "paspoort_objectvoorbeeld": _join_preview(object_values, max_items=8),
            "paspoort_naamvarianten": _join_preview(project_names),
        }

        hm_values = pd.to_numeric(group["_hm_sort_control"], errors="coerce").dropna()
        if not hm_values.empty:
            record["paspoort_hm_min"] = float(hm_values.min())
            record["paspoort_hm_max"] = float(hm_values.max())
        else:
            record["paspoort_hm_min"] = ""
            record["paspoort_hm_max"] = ""

        if "Wegnummer" in group.columns:
            record["paspoort_wegen"] = _join_preview(group["Wegnummer"])

        records.append(record)

    return pd.DataFrame(records).sort_values(["onderhoudsproject"]).reset_index(drop=True)


def summarize_maintenance_projects(maintenance_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Groepeer onderhoudsexportregels per onderhoudsproject."""
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame()

    project_column = _first_existing_column(maintenance_df, ["Onderhoudsproject", "project"])
    if project_column is None:
        return pd.DataFrame()

    working = maintenance_df.copy()
    working["project_norm"] = working[project_column].apply(normalize_project_name)
    working = working[working["project_norm"] != ""].copy()

    if selected_road:
        selected = clean_display_value(selected_road).upper()
        if selected:
            # Filter mild op projectnaam, omdat onderhoudsexporten vaak geen losse
            # Wegnummer-kolom hebben maar wel projectnamen als N398-HRB-...
            working = working[working["project_norm"].str.contains(re.escape(selected.upper()), na=False)].copy()

    if working.empty:
        return pd.DataFrame()

    object_column = _first_existing_column(working, ["objectnummer", "nummer", "bron_id"])
    measure_column = _first_existing_column(working, ["maatregel", "Maatregel Omschrijving"])
    amount_column = _first_existing_column(working, ["hoeveelheid", "Hoeveelh."])
    price_column = _first_existing_column(working, ["totaalprijs", "Totaalprijs"])
    amount_value_column = _best_numeric_column_near(working, amount_column, stop_column=price_column)

    records: list[dict[str, Any]] = []

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group[project_column] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm

        record = {
            "onderhoudsproject": project_name,
            "project_norm": project_norm,
            "onderhoud_regels": int(len(group)),
            "onderhoud_naamvarianten": _join_preview(project_names),
            "onderhoud_bronbestanden": _join_preview(group["bronbestand_onderhoud"]) if "bronbestand_onderhoud" in group.columns else "",
        }

        if object_column:
            object_values = group[object_column]
            record["onderhoud_unieke_objecten"] = int(object_values.astype(str).replace("", pd.NA).dropna().nunique())
            record["onderhoud_objectvoorbeeld"] = _join_preview(object_values, max_items=8)
        else:
            record["onderhoud_unieke_objecten"] = 0
            record["onderhoud_objectvoorbeeld"] = ""

        if measure_column:
            record["onderhoud_unieke_maatregelen"] = int(group[measure_column].astype(str).replace("", pd.NA).dropna().nunique())
            record["onderhoud_maatregel_samenvatting"] = _join_preview(group[measure_column], max_items=6)
        else:
            record["onderhoud_unieke_maatregelen"] = 0
            record["onderhoud_maatregel_samenvatting"] = ""

        record["onderhoud_hoeveelheid_totaal"] = _safe_sum(group[amount_value_column]) if amount_value_column else 0.0
        record["onderhoud_hoeveelheid_bronkolom"] = amount_value_column or ""
        record["onderhoud_totaalprijs"] = _safe_sum(group[price_column]) if price_column else 0.0

        records.append(record)

    return pd.DataFrame(records).sort_values(["onderhoudsproject"]).reset_index(drop=True)


def compare_passport_and_maintenance(
    passport_projects: pd.DataFrame,
    maintenance_projects: pd.DataFrame,
) -> pd.DataFrame:
    """Leg paspoortprojecten en onderhoudsprojecten naast elkaar."""
    passport_by_key = {
        str(row["project_norm"]): row.to_dict()
        for _, row in passport_projects.iterrows()
    } if passport_projects is not None and not passport_projects.empty else {}

    maintenance_by_key = {
        str(row["project_norm"]): row.to_dict()
        for _, row in maintenance_projects.iterrows()
    } if maintenance_projects is not None and not maintenance_projects.empty else {}

    keys = sorted(set(passport_by_key) | set(maintenance_by_key))
    records: list[dict[str, Any]] = []

    for key in keys:
        passport_record = passport_by_key.get(key, {})
        maintenance_record = maintenance_by_key.get(key, {})
        in_passport = bool(passport_record)
        in_maintenance = bool(maintenance_record)

        project_name = (
            passport_record.get("onderhoudsproject")
            or maintenance_record.get("onderhoudsproject")
            or key
        )

        if in_passport and in_maintenance:
            status = "OK"
            severity = "ok"
            message = "Project komt voor in paspoortexport én onderhoudsexport."
        elif in_passport:
            status = "ONTBREEKT_IN_ONDERHOUD"
            severity = "waarschuwing"
            message = "Project staat bij objecten in de paspoortexport, maar ontbreekt in de onderhoudsexport."
        else:
            status = "GEEN_PASPOORTOBJECTEN"
            severity = "waarschuwing"
            message = "Project staat in de onderhoudsexport, maar er zijn geen paspoortobjecten met deze projectnaam."

        record = {
            "status": status,
            "ernst": severity,
            "onderhoudsproject": project_name,
            "project_norm": key,
            "controle_bericht": message,
            "in_paspoortexport": in_passport,
            "in_onderhoudsexport": in_maintenance,
        }
        record.update({column: passport_record.get(column, "") for column in passport_projects.columns if column not in {"project_norm", "onderhoudsproject"}} if passport_projects is not None and not passport_projects.empty else {})
        record.update({column: maintenance_record.get(column, "") for column in maintenance_projects.columns if column not in {"project_norm", "onderhoudsproject"}} if maintenance_projects is not None and not maintenance_projects.empty else {})

        records.append(record)

    if not records:
        return pd.DataFrame()

    sort_order = {"waarschuwing": 0, "info": 1, "ok": 2}
    result = pd.DataFrame(records)
    result["_sort_ernst"] = result["ernst"].map(sort_order).fillna(9)
    result = result.sort_values(["_sort_ernst", "onderhoudsproject"]).drop(columns=["_sort_ernst"]).reset_index(drop=True)
    return result


def build_maintenance_control(
    passport_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    selected_road: str | None = None,
) -> MaintenanceControlResult:
    """
    Bouw de volledige Fase-4-controle.

    Deze functie is bewust pandas-only en kent Streamlit niet. Daardoor kunnen we
    de controle later ook als commandline- of batchcontrole gebruiken.
    """
    warnings: list[str] = []

    passport_projects = summarize_passport_projects(passport_df, selected_road=selected_road)
    maintenance_projects = summarize_maintenance_projects(maintenance_df, selected_road=selected_road)

    if passport_projects.empty:
        warnings.append("Geen onderhoudsprojecten gevonden in de paspoortexport voor deze selectie.")
    if maintenance_projects.empty:
        warnings.append("Geen onderhoudsprojecten gevonden in de onderhoudsexport voor deze selectie.")

    comparison = compare_passport_and_maintenance(passport_projects, maintenance_projects)

    if comparison.empty:
        summary = {
            "projecten_totaal": 0,
            "projecten_ok": 0,
            "ontbreekt_in_onderhoud": 0,
            "geen_paspoortobjecten": 0,
            "waarschuwingen": 0,
        }
    else:
        summary = {
            "projecten_totaal": int(len(comparison)),
            "projecten_ok": int((comparison["status"] == "OK").sum()),
            "ontbreekt_in_onderhoud": int((comparison["status"] == "ONTBREEKT_IN_ONDERHOUD").sum()),
            "geen_paspoortobjecten": int((comparison["status"] == "GEEN_PASPOORTOBJECTEN").sum()),
            "waarschuwingen": int((comparison["ernst"] == "waarschuwing").sum()),
        }

    return MaintenanceControlResult(
        summary=summary,
        comparison=comparison,
        passport_projects=passport_projects,
        maintenance_projects=maintenance_projects,
        warnings=warnings,
    )
